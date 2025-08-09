#include "afp/peaks/peaks.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace afp::peaks {
using afp::util::Expected;
using afp::util::UtilError;
using afp::util::Peak;
using afp::util::Spectrogram;

namespace {
// median via nth_element on a copy (O(n))
inline float median_of_span(std::span<const float> v) {
  if (v.empty()) return 0.0f;
  std::vector<float> tmp(v.begin(), v.end());
  const std::size_t mid = tmp.size() / 2;
  std::nth_element(tmp.begin(), tmp.begin() + mid, tmp.end());
  const float m0 = tmp[mid];
  if (tmp.size() & 1u) return m0;
  const float m1 = *std::max_element(tmp.begin(), tmp.begin() + mid);
  return 0.5f * (m0 + m1);
}

inline std::span<const float> row_span(const Spectrogram& S, std::uint32_t t) {
  const std::size_t off = static_cast<std::size_t>(t) * S.num_bins;
  return std::span<const float>(S.log_mag.data() + off, S.num_bins);
}

// Greedy NMS in a single frame: keep peaks sorted by strength, enforce min bin separation.
static void nms_per_frame(std::vector<Peak>& frame_peaks,
                          std::uint16_t min_sep_bins) {
  std::sort(frame_peaks.begin(), frame_peaks.end(),
            [](const Peak& a, const Peak& b) {
              return a.strength_db > b.strength_db;
            });
  std::vector<Peak> kept;
  for (const auto& p : frame_peaks) {
    bool ok = true;
    for (const auto& q : kept) {
      const auto df = (p.f > q.f) ? (p.f - q.f) : (q.f - p.f);
      if (df < min_sep_bins) {
        ok = false;
        break;
      }
    }
    if (ok) kept.push_back(p);
  }
  std::sort(kept.begin(), kept.end(),
            [](const Peak& a, const Peak& b) {
              return (a.t < b.t) || (a.t == b.t && a.f < b.f);
            });
  frame_peaks.swap(kept);
}
} // namespace

class PeakFinderKfr final : public IPeakFinder {
public:
  Expected<std::vector<Peak>>
  find(const Spectrogram& S, const PeakParams& pp) override {
    if (S.num_bins == 0 || S.num_frames == 0) return std::vector<Peak>{};
    if (S.log_mag.size() != static_cast<std::size_t>(S.num_bins) * S.num_frames)
      return tl::unexpected(UtilError::SizeMismatch);
    if (S.sample_rate_hz == 0 || S.hop_size == 0)
      return tl::unexpected(UtilError::InvalidArgument);

    const int Tr = static_cast<int>(pp.neighborhood_time_frames);
    const int Fr = static_cast<int>(pp.neighborhood_freq_bins);
    const float base_thresh_db = pp.threshold_db;

    // Per-frame target caps (peaks/frame)
    const double fps = static_cast<double>(S.sample_rate_hz) / static_cast<
                         double>(S.hop_size);
    const std::uint32_t min_per_frame = static_cast<std::uint32_t>(
      std::llround(pp.target_peak_density_per_sec_min / std::max(1.0, fps)));
    const std::uint32_t max_per_frame = std::max<std::uint32_t>(min_per_frame,
      static_cast<std::uint32_t>(std::llround(
          pp.target_peak_density_per_sec_max / std::max(1.0, fps))));

    std::vector<Peak> out;
    out.reserve(
        static_cast<std::size_t>(S.num_frames) * (max_per_frame
                                                    ? max_per_frame
                                                    : 8));

    // Working buffers
    kfr::univector<float> local_row(S.num_bins);
    kfr::univector<float> maxima_mask(S.num_bins); // 1.0 if local max else 0.0

    // Iterate frames
    for (std::uint32_t t = 0; t < S.num_frames; ++t) {
      // Copy frame row into KFR vector
      const auto row = row_span(S, t);
      std::memcpy(local_row.data(), row.data(), row.size() * sizeof(float));

      // Adaptive threshold: per-frame median
      const float med = median_of_span(row);
      const float thr = med + base_thresh_db;

      // Collect *all* local maxima (no threshold yet)
      std::vector<Peak> candidates;
      candidates.reserve(64);

      for (std::uint32_t f = 0; f < S.num_bins; ++f) {
        const float v = local_row[f];

        // Local neighborhood bounds in time/freq
        const int t0 = std::max<int>(0, static_cast<int>(t) - Tr);
        const int t1 = std::min<
          int>(S.num_frames - 1, static_cast<int>(t) + Tr);
        const int f0 = std::max<int>(0, static_cast<int>(f) - Fr);
        const int f1 = std::min<int>(S.num_bins - 1, static_cast<int>(f) + Fr);

        bool is_max = true;
        // Check within (t0..t1) × (f0..f1)
        for (int tt = t0; tt <= t1 && is_max; ++tt) {
          const auto nrow = row_span(S, static_cast<std::uint32_t>(tt));
          for (int ff = f0; ff <= f1; ++ff) {
            if (tt == static_cast<int>(t) && ff == static_cast<int>(f))
              continue
                  ;
            if (nrow[static_cast<std::size_t>(ff)] > v) {
              is_max = false;
              break;
            }
          }
        }
        if (is_max) {
          candidates.push_back(Peak{t, static_cast<afp::util::BinIndex>(f), v});
        }
      }

      // Split candidates into above/below threshold
      std::vector<Peak> hi, lo;
      hi.reserve(candidates.size());
      lo.reserve(candidates.size());
      for (const auto& p : candidates) {
        ((p.strength_db >= thr) ? hi : lo).push_back(p);
      }

      // Per-frame NMS on high-confidence set
      nms_per_frame(hi, pp.min_freq_separation_bins);

      // Cap at max_per_frame if set (>0)
      if (max_per_frame > 0 && hi.size() > max_per_frame) {
        // Keep strongest
        std::sort(hi.begin(), hi.end(),
                  [](const Peak& a, const Peak& b) {
                    return a.strength_db > b.strength_db;
                  });
        hi.resize(max_per_frame);
        std::sort(hi.begin(), hi.end(),
                  [](const Peak& a, const Peak& b) { return a.f < b.f; });
      }

      // If too few peaks, backfill from "lo" (below threshold) — strongest first, with NMS.
      if (hi.size() < min_per_frame && !lo.empty()) {
        // Sort lo by strength desc, then greedily add if respects min separation.
        std::sort(lo.begin(), lo.end(),
                  [](const Peak& a, const Peak& b) {
                    return a.strength_db > b.strength_db;
                  });
        for (const auto& p : lo) {
          bool ok = true;
          for (const auto& q : hi) {
            const auto df = (p.f > q.f) ? (p.f - q.f) : (q.f - p.f);
            if (df < pp.min_freq_separation_bins) {
              ok = false;
              break;
            }
          }
          if (ok) hi.push_back(p);
          if (hi.size() >= std::max<std::size_t>(
                  min_per_frame,
                  max_per_frame ? max_per_frame : hi.size() + 1))
            break;
        }
        // Keep natural order by frequency
        std::sort(hi.begin(), hi.end(),
                  [](const Peak& a, const Peak& b) { return a.f < b.f; });
      }

      // Append to output
      out.insert(out.end(), hi.begin(), hi.end());
    }

    // Final sort by (t,f)
    std::sort(out.begin(), out.end(),
              [](const Peak& a, const Peak& b) {
                return (a.t < b.t) || (a.t == b.t && a.f < b.f);
              });
    return out;
  }
};

class DefaultPeaksFactory final : public IPeaksFactory {
public:
  std::unique_ptr<IPeakFinder> create_peak_finder() override {
    return std::make_unique<PeakFinderKfr>();
  }
};

std::unique_ptr<IPeaksFactory> make_default_peaks_factory() {
  return std::make_unique<DefaultPeaksFactory>();
}
} // namespace afp::peaks
