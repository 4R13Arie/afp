#include "afp/fingerprint/fingerprint.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace afp::fingerprint {
using afp::util::Expected;
using afp::util::UtilError;
using afp::util::Peak;
using afp::util::FingerprintPair;
using afp::util::FingerprintKey;
using afp::util::TimeMs;
using afp::util::SampleRateHz;

// -----------------------------
// Builder
// -----------------------------

namespace {
inline bool params_ok(const PairingParams& pp) noexcept {
  return pp.sample_rate_hz > 0 && pp.hop_size > 0 &&
         pp.delta_min_ms <= pp.delta_max_ms;
}

// Map frame index -> time (ms) using scale = 1000 * hop / sr
inline double frame_to_ms_scale(SampleRateHz sr, std::uint32_t hop) noexcept {
  return 1000.0 * static_cast<double>(hop) / static_cast<double>(sr);
}
} // namespace

class FingerprintBuilderKfr final : public IFingerprintBuilder {
public:
  Expected<std::vector<FingerprintPair>>
  make_pairs(std::span<const Peak> peaks, const PairingParams& pp) override {
    if (!params_ok(pp)) return tl::unexpected(UtilError::InvalidArgument);
    if (peaks.empty() || pp.max_targets_per_anchor == 0)
      return std::vector<FingerprintPair>{};

    // Precompute t_ms for each peak (vectorized temporary with KFR)
    kfr::univector<float> t_ms_vec(peaks.size());
    const double s = frame_to_ms_scale(pp.sample_rate_hz, pp.hop_size);
    for (std::size_t i = 0; i < peaks.size(); ++i)
      t_ms_vec[i] = static_cast<float>(static_cast<double>(peaks[i].t) * s);

    const TimeMs dmin = pp.delta_min_ms;
    const TimeMs dmax = pp.delta_max_ms;

    std::vector<FingerprintPair> out;
    out.reserve(
        peaks.size() * std::min<std::uint8_t>(pp.max_targets_per_anchor, 4));

    // Sliding window over targets per anchor.
    std::size_t j_start = 0;
    for (std::size_t i = 0; i < peaks.size(); ++i) {
      const auto& a = peaks[i];
      const auto t_a_ms = static_cast<TimeMs>(std::llround(t_ms_vec[i]));

      // Advance j_start to first candidate with Δt >= dmin
      while (j_start < peaks.size()) {
        const TimeMs dt = diff_ms(i, j_start, t_ms_vec);
        if (j_start <= i || dt < dmin) {
          ++j_start;
          continue;
        }
        break;
      }
      if (j_start >= peaks.size()) break;

      // Collect up to max_targets_per_anchor within Δt ≤ dmax
      std::uint8_t taken = 0;
      for (std::size_t j = j_start; j < peaks.size(); ++j) {
        const TimeMs dt = diff_ms(i, j, t_ms_vec);
        if (dt < dmin) continue;
        if (dt > dmax) break; // window exhausted for this anchor

        const auto& b = peaks[j];
        out.push_back(FingerprintPair{
            /*f_anchor*/ a.f,
                         /*f_target*/ b.f,
                         /*delta_ms*/ dt,
                         /*t_anchor_ms*/ t_a_ms
        });
        if (++taken >= pp.max_targets_per_anchor) break;
      }
    }

    // Ensure non-decreasing t_anchor_ms
    std::sort(out.begin(), out.end(),
              [](const FingerprintPair& x, const FingerprintPair& y) {
                return x.t_anchor_ms < y.t_anchor_ms;
              });

    return out;
  }

private:
  static inline TimeMs diff_ms(std::size_t i, std::size_t j,
                               const kfr::univector<float>& tms) noexcept {
    const float dt = tms[j] - tms[i];
    return (dt <= 0.f) ? 0u : static_cast<TimeMs>(std::llround(dt));
  }
};

// -----------------------------
// Key packer
// -----------------------------

class KeyPacker32 final : public IKeyPacker {
public:
  Expected<std::vector<FingerprintKey>>
  pack(std::span<const FingerprintPair> pairs) override {
    // Layout: [fa:11][ft:11][dt_q:10]
    // dt_q = round(delta_ms / 2.5), covers ~0..2557.5 ms
    constexpr std::uint32_t FA_MASK = 0x07FFu; // 11 bits -> 0..2047
    constexpr std::uint32_t FT_MASK = 0x07FFu; // 11 bits -> 0..2047
    constexpr std::uint32_t DTQ_MASK = 0x03FFu; // 10 bits -> 0..1023
    constexpr std::uint32_t FA_SHIFT = 21;
    constexpr std::uint32_t FT_SHIFT = 10;
    constexpr float DT_STEP_MS = 2.5f; // must match packer used at ingest

    std::vector<FingerprintKey> keys;
    keys.reserve(pairs.size());

    for (const auto& p : pairs) {
      const std::uint32_t fa = static_cast<std::uint32_t>(p.f_anchor);
      const std::uint32_t ft = static_cast<std::uint32_t>(p.f_target);

      if (fa > FA_MASK || ft > FT_MASK) continue;

      // Quantize Δt (ms) to 10-bit code
      const float qf = static_cast<float>(p.delta_ms) / DT_STEP_MS;
      if (qf < 0.0f) continue;
      const std::uint32_t dt_q = static_cast<std::uint32_t>(std::lround(qf));
      if (dt_q > DTQ_MASK) continue;

      const std::uint32_t packed =
          (fa << FA_SHIFT) |
          (ft << FT_SHIFT) |
          (dt_q & DTQ_MASK);

      keys.push_back(FingerprintKey{packed});
    }
    return keys;
  }
};

// -----------------------------
// Factory
// -----------------------------

class DefaultFingerprintFactory final : public IFingerprintFactory {
public:
  std::unique_ptr<IFingerprintBuilder> create_builder() override {
    return std::make_unique<FingerprintBuilderKfr>();
  }

  std::unique_ptr<IKeyPacker> create_packer() override {
    return std::make_unique<KeyPacker32>();
  }
};

std::unique_ptr<IFingerprintFactory> make_default_fingerprint_factory() {
  return std::make_unique<DefaultFingerprintFactory>();
}
} // namespace afp::fingerprint
