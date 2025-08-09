#include "afp/match/match.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "afp/util/util.hpp"
#include "afp/index/index.hpp"

namespace afp::match_ {
using afp::util::CandidateMatch;
using afp::util::Expected;
using afp::util::FingerprintKey;
using afp::util::FingerprintPair;
using afp::util::MatchResult;
using afp::util::Posting;
using afp::util::UtilError;

// ------------------------------------------------------
// Key packing assumptions (must match index build phase)
// Layout: [fa:11][ft:11][dt_q:10], dt_q = round(delta_ms / Δt_step_ms)
// ------------------------------------------------------
namespace {
constexpr std::uint32_t FA_MASK = 0x07FFu; // 11 bits -> 0..2047
constexpr std::uint32_t FT_MASK = 0x07FFu;
constexpr std::uint32_t DTQ_MASK = 0x03FFu; // 10 bits -> 0..1023
constexpr std::uint32_t FA_SHIFT = 21;
constexpr std::uint32_t FT_SHIFT = 10;
constexpr float DT_STEP_MS = 2.5f; // must match packer used at ingest

inline std::uint32_t pack_key(std::uint32_t fa, std::uint32_t ft,
                              std::uint32_t dt_q) noexcept {
  return ((fa & FA_MASK) << FA_SHIFT) | ((ft & FT_MASK) << FT_SHIFT) | (
           dt_q & DTQ_MASK);
}

inline std::uint32_t quantize_dt_ms(std::uint32_t delta_ms) noexcept {
  const float qf = static_cast<float>(delta_ms) / DT_STEP_MS;
  if (qf <= 0.f) return 0u;
  const std::uint32_t q = static_cast<std::uint32_t>(std::lround(qf));
  return (q > DTQ_MASK) ? DTQ_MASK : q;
}

inline std::int32_t bin_for_offset(std::int64_t offset_ms,
                                   std::uint32_t bin_width_ms) noexcept {
  // floor division for possibly negative offsets
  const double v = static_cast<double>(offset_ms) / static_cast<double>(
                     bin_width_ms);
  return static_cast<std::int32_t>(std::floor(v));
}

struct BinStats {
  std::uint32_t count{0};
  std::uint32_t min_q_ms{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t max_q_ms{0};
};
} // namespace

// ------------------------------------------------------
// Matcher implementation
// ------------------------------------------------------

class DefaultMatcher final : public IMatcher {
public:
  Expected<MatchResult>
  match(std::span<const FingerprintPair> query_pairs,
        const afp::index::IIndexReader& index,
        const MatchParams& mp) const override {
    if (mp.top_k == 0 || mp.offset_bin_ms == 0)
      return tl::unexpected(UtilError::InvalidArgument);
    if (query_pairs.empty()) return MatchResult{};

    // 1) Expand tolerant keys (deduplicate per key), track which query anchor times map to each key
    // key -> vector of query anchor times (ms) (dedup'd)
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> key_to_qtimes;
    key_to_qtimes.reserve(query_pairs.size() * 4);

    const int fb = static_cast<int>(mp.freq_tolerance_bins);
    const std::int32_t dt_tol_codes = static_cast<std::int32_t>(
      std::ceil(
          static_cast<double>(mp.delta_t_tolerance_ms) / static_cast<double>(
            DT_STEP_MS)));

    for (const auto& pair : query_pairs) {
      const std::uint32_t fa0 = pair.f_anchor;
      const std::uint32_t ft0 = pair.f_target;
      const std::uint32_t dt0 = pair.delta_ms;
      const std::uint32_t ta = pair.t_anchor_ms;

      const std::uint32_t dtq0 = quantize_dt_ms(dt0);

      for (int dfa = -fb; dfa <= fb; ++dfa) {
        const std::int32_t fa = static_cast<std::int32_t>(fa0) + dfa;
        if (fa < 0 || fa > static_cast<std::int32_t>(FA_MASK)) continue;

        for (int dft = -fb; dft <= fb; ++dft) {
          const std::int32_t ft = static_cast<std::int32_t>(ft0) + dft;
          if (ft < 0 || ft > static_cast<std::int32_t>(FT_MASK)) continue;

          for (std::int32_t dd = -dt_tol_codes; dd <= dt_tol_codes; ++dd) {
            const std::int32_t q = static_cast<std::int32_t>(dtq0) + dd;
            if (q < 0 || q > static_cast<std::int32_t>(DTQ_MASK)) continue;

            const std::uint32_t keyv = pack_key(static_cast<std::uint32_t>(fa),
                                                static_cast<std::uint32_t>(ft),
                                                static_cast<std::uint32_t>(q));
            auto& times = key_to_qtimes[keyv];
            times.push_back(ta);
          }
        }
      }
    }

    // Deduplicate the per-key query times (keep them sorted for reproducibility)
    for (auto& kv : key_to_qtimes) {
      auto& v = kv.second;
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    // 2) Batch lookup all unique keys
    std::vector<FingerprintKey> keys;
    keys.reserve(key_to_qtimes.size());
    for (const auto& kv : key_to_qtimes)
      keys.push_back(
          FingerprintKey{kv.first});

    auto lookupsE = index.lookup_batch(
        std::span<const FingerprintKey>(keys.data(), keys.size()));
    if (!lookupsE) return tl::unexpected(lookupsE.error());
    const auto& lists = *lookupsE; // vector< vector<Posting> >

    // 3) Vote into (track -> offset-bin -> stats)
    using Track = afp::util::TrackId;
    struct PairHash {
      std::size_t operator()(
          const std::pair<Track, std::int32_t>& x) const noexcept {
        const auto h1 = std::hash<Track>{}(x.first);
        const auto h2 = std::hash<std::int32_t>{}(x.second);
        return h1 ^ (h2 * 0x9E3779B1u);
      }
    };
    std::unordered_map<std::pair<Track, std::int32_t>, BinStats, PairHash> bins;
    bins.reserve(lists.size() * 2);

    for (std::size_t i = 0; i < keys.size(); ++i) {
      const auto& qtimes = key_to_qtimes[keys[i].value];
      const auto& postings = lists[i];
      if (qtimes.empty() || postings.empty()) continue;

      // For each posting and each query time, vote an offset bin
      for (const auto& p : postings) {
        for (auto ta : qtimes) {
          const std::int64_t off =
              static_cast<std::int64_t>(p.t_anchor_ms) - static_cast<
                std::int64_t>(ta);
          const std::int32_t bin = bin_for_offset(off, mp.offset_bin_ms);
          auto& st = bins[{p.track, bin}];
          ++st.count;
          if (ta < st.min_q_ms) st.min_q_ms = ta;
          if (ta > st.max_q_ms) st.max_q_ms = ta;
        }
      }
    }

    // 4) Reduce per-track best bin, build candidates
    struct TrackBest {
      std::int32_t bin;
      BinStats stats;
    };
    std::unordered_map<Track, TrackBest> best;
    best.reserve(bins.size());

    for (const auto& kv : bins) {
      const Track t = kv.first.first;
      const std::int32_t bin = kv.first.second;
      const auto& st = kv.second;
      auto it = best.find(t);
      if (it == best.end() || st.count > it->second.stats.count ||
          (st.count == it->second.stats.count && (st.max_q_ms - st.min_q_ms) > (
             it->second.stats.max_q_ms - it->second.stats.min_q_ms))) {
        best[t] = TrackBest{bin, st};
      }
    }

    std::vector<CandidateMatch> cand;
    cand.reserve(best.size());
    for (const auto& b : best) {
      const Track t = b.first;
      const auto bin = b.second.bin;
      const auto& st = b.second.stats;
      const std::uint32_t inliers = st.count;
      const std::uint32_t span = (st.max_q_ms > st.min_q_ms)
                                   ? (st.max_q_ms - st.min_q_ms)
                                   : 0u;
      if (inliers < mp.min_inliers || span < mp.min_span_ms) continue;

      CandidateMatch cm;
      cm.track = t;
      cm.offset_ms = static_cast<afp::util::TimeMs>(
        bin * static_cast<std::int32_t>(mp.offset_bin_ms));
      cm.inliers = inliers;
      cm.span_ms = span;
      cm.score = static_cast<float>(inliers) + 0.001f * static_cast<float>(
                   span); // simple composite
      cand.push_back(cm);
    }

    std::sort(cand.begin(), cand.end(),
              [](const CandidateMatch& a, const CandidateMatch& b) {
                if (a.score != b.score) return a.score > b.score;
                if (a.inliers != b.inliers) return a.inliers > b.inliers;
                return a.span_ms > b.span_ms;
              });

    // Apply margin rule (optional rejection if no clear winner)
    if (cand.size() >= 2) {
      const float top1 = cand[0].score;
      const float top2 = cand[1].score;
      if (!(top1 >= mp.min_margin_ratio * top2)) {
        // No clear winner – return empty result (policy choice).
        return MatchResult{};
      }
    }

    if (cand.size() > mp.top_k) cand.resize(mp.top_k);

    MatchResult mr;
    mr.topk = std::move(cand);
    return mr;
  }
};

// ------------------------------------------------------
// Factory
// ------------------------------------------------------

class DefaultMatchFactory final : public IMatchFactory {
public:
  std::unique_ptr<IMatcher> create_matcher() override {
    return std::make_unique<DefaultMatcher>();
  }
};

std::unique_ptr<IMatchFactory> make_default_match_factory() {
  return std::make_unique<DefaultMatchFactory>();
}
} // namespace afp::match_
