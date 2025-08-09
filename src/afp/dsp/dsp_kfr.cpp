#include "afp/dsp/dsp.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace afp::dsp {
using afp::util::Expected;
using afp::util::PcmBuffer;
using afp::util::PcmSpan;
using afp::util::SampleRateHz;
using afp::util::UtilError;

// ==============================
// Helpers
// ==============================

namespace {
inline bool valid_cutoff(float cutoff_hz, SampleRateHz sr) noexcept {
  return (sr > 0) && (cutoff_hz > 0.0f) && (
           cutoff_hz < 0.5f * static_cast<float>(sr));
}
} // namespace

// ==============================
// High-pass (KFR biquad + iir)
// ==============================

class HighPassKfr final : public IHighPass {
public:
  Expected<PcmBuffer>
  process(const PcmSpan& in, const BiquadParams& hp) override {
    if (!valid_cutoff(hp.cutoff_hz, in.sample_rate_hz) || hp.Q <= 0.0f)
      return tl::unexpected(UtilError::InvalidArgument);

    try {
      const auto sr = static_cast<double>(in.sample_rate_hz);
      const double nyq = std::max(1.0, sr * 0.5);
      const double cutoff = std::clamp(static_cast<double>(hp.cutoff_hz), 1.0,
                                       nyq * 0.99);
      const auto norm = static_cast<float>(cutoff / sr);
      const auto q = static_cast<float>(std::max(
          1e-6, static_cast<double>(hp.Q)));

      using T = float;
      const auto bq = kfr::biquad_highpass<T>(norm, q);
      kfr::iir_params params{bq};

      kfr::univector<T> in_uv(in.samples.begin(), in.samples.end());
      kfr::univector<T> out_uv = kfr::iir<T>(in_uv, params);

      PcmBuffer out;
      out.sample_rate_hz = in.sample_rate_hz;
      out.samples = std::move(out_uv);
      return out;
    } catch (...) {
      return tl::unexpected(UtilError::DspError);
    }
  }

  void reset() noexcept override {
    /* stateless */
  }
};

// ==============================
// Low-pass (KFR biquad + iir)
// ==============================

class LowPassKfr final : public ILowPass {
public:
  Expected<PcmBuffer>
  process(const PcmSpan& in, const BiquadParams& lp) override {
    if (!valid_cutoff(lp.cutoff_hz, in.sample_rate_hz) || lp.Q <= 0.0f)
      return tl::unexpected(UtilError::InvalidArgument);

    try {
      const auto sr = static_cast<double>(in.sample_rate_hz);
      const double nyq = std::max(1.0, sr * 0.5);
      const double cutoff = std::clamp(static_cast<double>(lp.cutoff_hz), 1.0,
                                       nyq * 0.999);
      const auto norm = static_cast<float>(cutoff / sr);
      const auto q = static_cast<float>(std::max(
          1e-6, static_cast<double>(lp.Q)));

      using T = float;
      const auto bq = kfr::biquad_lowpass<T>(norm, q);
      kfr::iir_params params{bq};

      kfr::univector<T> in_uv(in.samples.begin(), in.samples.end());
      kfr::univector<T> out_uv = kfr::iir<T>(in_uv, params);

      PcmBuffer out;
      out.sample_rate_hz = in.sample_rate_hz;
      out.samples = std::move(out_uv);
      return out;
    } catch (...) {
      return tl::unexpected(UtilError::DspError);
    }
  }

  void reset() noexcept override {
    /* stateless */
  }
};

// ==============================
// Pre-emphasis
// ==============================

class PreEmphasisKfr final : public IPreEmphasis {
public:
  Expected<PcmBuffer>
  process(const PcmSpan& in, const PreEmphasisParams& p) override {
    if (in.sample_rate_hz == 0 || p.alpha < 0.0f || p.alpha > 1.0f)
      return tl::unexpected(UtilError::InvalidArgument);

    try {
      using T = float;
      kfr::univector<T> in_uv(in.samples.begin(), in.samples.end());
      kfr::univector<T> y(in_uv.size());

      if (!in_uv.empty()) {
        const T a = static_cast<T>(p.alpha);
        y[0] = in_uv[0]; // x[0] - a*x[-1] (x[-1]=0) => x[0]
        for (size_t n = 1; n < in_uv.size(); ++n) {
          y[n] = in_uv[n] - a * in_uv[n - 1]; // causal pre-emphasis
        }
      }

      PcmBuffer out;
      out.sample_rate_hz = in.sample_rate_hz;
      out.samples = std::move(y);
      return out;
    } catch (...) {
      return tl::unexpected(UtilError::DspError);
    }
  }

  void reset() noexcept override {
    /* stateless */
  }
};

// ==============================
// Resampler (KFR resampler class)
// ==============================
class ResamplerKfr final : public IResampler {
public:
  Expected<PcmBuffer>
  resample(const PcmSpan& in, const ResampleParams& rp) override {
    if (in.sample_rate_hz == 0 || rp.target_hz == 0)
      return tl::unexpected(UtilError::InvalidArgument);

    try {
      using T = float;

      // Copy input to KFR vector
      kfr::univector<T> in_uv(in.samples.begin(), in.samples.end());

      PcmBuffer out;
      out.sample_rate_hz = rp.target_hz;
      if (in_uv.empty()) return out; // empty in → empty out

      // Clamp quality to valid range (KFR supports discrete levels, 0..8 is safe)
      const int q = std::clamp(rp.quality, 0, 8);

      // Build resampler: (quality, out_sr, in_sr)
      auto r = kfr::resampler<T>(
          static_cast<kfr::sample_rate_conversion_quality>(q),
          static_cast<int>(rp.target_hz),
          static_cast<int>(in.sample_rate_hz));

      // Compute expected output length and pre-size output buffer
      const double ratio = double(rp.target_hz) / double(in.sample_rate_hz);
      const size_t out_len = static_cast<size_t>(std::llround(
          in_uv.size() * ratio));
      kfr::univector<T> out_uv(out_len);

      // KFR: writes exactly out_uv.size() samples; returns number of input samples consumed
      (void)r.process(out_uv, in_uv);

      out.samples = std::move(out_uv);
      return out;
    } catch (...) {
      return tl::unexpected(UtilError::DspError);
    }
  }

  void reset() noexcept override {
  }
};


// ==============================
// Factory
// ==============================

class DefaultDspFactory final : public IDspFactory {
public:
  [[nodiscard]] std::unique_ptr<IHighPass> create_hpf() const override {
    return std::make_unique<HighPassKfr>();
  }

  [[nodiscard]] std::unique_ptr<ILowPass> create_lpf() const override {
    return std::make_unique<LowPassKfr>();
  }

  [[nodiscard]] std::unique_ptr<IPreEmphasis>
  create_preemphasis() const override {
    return std::make_unique<PreEmphasisKfr>();
  }

  [[nodiscard]] std::unique_ptr<IResampler> create_resampler() const override {
    return std::make_unique<ResamplerKfr>();
  }
};

std::unique_ptr<IDspFactory> make_default_dsp_factory() {
  return std::make_unique<DefaultDspFactory>();
}
} // namespace afp::dsp