#include "afp/features/features.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace afp::features {
using afp::util::Expected;
using afp::util::UtilError;
using afp::util::Spectrogram;
using afp::util::ComplexSpectra;

// -----------------------------
// Helpers
// -----------------------------

namespace {
inline std::uint32_t hz_to_bin(float hz, afp::util::SampleRateHz sr,
                               std::uint32_t fft_size) noexcept {
  const double binf = (static_cast<double>(hz) * static_cast<double>(fft_size))
                      /
                      static_cast<double>(sr);
  const double b = std::round(binf);
  // inclusive band selection: we’ll clamp below
  if (b < 0.0) return 0;
  const auto maxb = static_cast<std::uint32_t>(fft_size / 2);
  const auto bi = static_cast<std::uint32_t>(b);
  return (bi > maxb) ? maxb : bi;
}

inline float safe_log20(float x, float eps) noexcept {
  // 20*log10(x + eps)
  const float v = std::max(x, 0.0f) + eps;
  return 20.0f * std::log10(v);
}

inline float median_of_span(std::span<float> v) {
  if (v.empty()) return 0.0f;
  const std::size_t mid = v.size() / 2;
  // nth_element partially sorts: O(n)
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  const float m0 = v[mid];
  if ((v.size() & 1u) == 1u) return m0;
  // even length → need the lower of upper half; nth_element doesn’t guarantee neighbors,
  // so compute the second median value explicitly.
  const float m1 = *std::max_element(v.begin(), v.begin() + mid);
  return 0.5f * (m0 + m1);
}

// Whitening via moving average (box filter) along frequency bins.
// Uses kfr::convolution_filter with uniform kernel; then trims to original length.
inline void whiten_inplace_frame(std::span<float> frame_db,
                                 std::uint16_t radius_bins) {
  if (radius_bins == 0 || frame_db.size() <= 2) return;
  const std::size_t L = static_cast<std::size_t>(2 * radius_bins + 1);

  kfr::univector<float> x(frame_db.size());
  std::memcpy(x.data(), frame_db.data(), frame_db.size() * sizeof(float));

  kfr::univector<float> kernel(L, 1.0f / static_cast<float>(L));
  kfr::univector<float> y = kfr::convolve(x, kernel);
  if (y.size() > x.size()) y.resize(x.size());
  // center/causal mismatch—trim to input length

  // Subtract smoothed envelope (simple whitening)
  for (std::size_t i = 0; i < frame_db.size(); ++i)
    frame_db[i] = frame_db[i] - y[i];
}
} // namespace

// -----------------------------
// SpectrogramBuilder (KFR)
// -----------------------------

class SpectrogramBuilderKfr final : public ISpectrogramBuilder {
public:
  Expected<Spectrogram> build(const ComplexSpectra& spec_cpx,
                              const LogSpecParams& params) override {
    // Basic parameter checks
    if (params.sample_rate_hz == 0 || params.fft_size == 0)
      return tl::unexpected(UtilError::InvalidArgument);
    const std::uint32_t expected_bins = params.fft_size / 2 + 1;
    if (spec_cpx.num_bins != expected_bins)
      return tl::unexpected(UtilError::SizeMismatch);

    // Band selection
    const std::uint32_t b0 = std::min(
        hz_to_bin(params.band_low_hz, params.sample_rate_hz, params.fft_size),
        expected_bins - 1);
    const std::uint32_t b1 = std::min(
        hz_to_bin(params.band_high_hz, params.sample_rate_hz, params.fft_size),
        expected_bins - 1);
    if (b1 < b0) return tl::unexpected(UtilError::InvalidArgument);
    const std::uint32_t out_bins = b1 - b0 + 1;

    // Prepare output container
    Spectrogram out;
    out.num_bins = static_cast<std::uint16_t>(out_bins);
    out.num_frames = spec_cpx.num_frames;
    out.sample_rate_hz = params.sample_rate_hz;
    out.hop_size = params.hop_size;
    out.fft_size = params.fft_size;
    out.log_mag.resize(static_cast<std::size_t>(out.num_frames) * out_bins);

    // Temporary per-frame buffers
    kfr::univector<float> mag(spec_cpx.num_bins);
    kfr::univector<float> dB(spec_cpx.num_bins);

    // Process each frame
    for (std::uint32_t t = 0; t < spec_cpx.num_frames; ++t) {
      const std::size_t src_off =
          static_cast<std::size_t>(t) * spec_cpx.num_bins;

      // 1) Magnitude -> dB
      for (std::uint32_t k = 0; k < spec_cpx.num_bins; ++k) {
        const auto z = spec_cpx.bins[src_off + k];
        const float m = std::hypot(z.real(), z.imag()); // |X|
        mag[k] = m;
      }
      for (std::uint32_t k = 0; k < spec_cpx.num_bins; ++k) {
        dB[k] = safe_log20(mag[k], params.epsilon);
      }

      // 2) Per-frame normalization (median subtract) — across all bins
      if (params.per_frame_median_subtract) {
        // Copy to scratch for median (mutable span)
        std::vector<float> scratch(dB.begin(), dB.end());
        float med = median_of_span(
            std::span<float>(scratch.data(), scratch.size()));
        for (std::uint32_t k = 0; k < spec_cpx.num_bins; ++k) dB[k] -= med;
      }

      // 3) Spectral whitening (moving-average) — along frequency
      if (params.whiten_radius_bins > 0) {
        std::span<float> frame_db(dB.data(), dB.size());
        whiten_inplace_frame(frame_db, params.whiten_radius_bins);
      }

      // 4) Clip dynamic range to ±clip_db
      const float lo = -std::abs(params.clip_db);
      const float hi = std::abs(params.clip_db);
      for (std::uint32_t k = 0; k < spec_cpx.num_bins; ++k) {
        dB[k] = std::clamp(dB[k], lo, hi);
      }

      // 5) Band-limit and store to output
      const std::size_t dst_off = static_cast<std::size_t>(t) * out_bins;
      std::memcpy(out.log_mag.data() + dst_off, dB.data() + b0,
                  out_bins * sizeof(float));
    }

    return out;
  }
};

// -----------------------------
// Factory
// -----------------------------

class DefaultFeaturesFactory final : public IFeaturesFactory {
public:
  std::unique_ptr<ISpectrogramBuilder> create_spectrogram_builder() override {
    return std::make_unique<SpectrogramBuilderKfr>();
  }
};

std::unique_ptr<IFeaturesFactory> make_default_features_factory() {
  return std::make_unique<DefaultFeaturesFactory>();
}
} // namespace afp::features
