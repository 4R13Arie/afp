#pragma once
#include <cstdint>
#include <memory>

#include <kfr/all.hpp>     // univector, complex, convolution
#include "afp/util/util.hpp"

namespace afp::features {
  /**
   * Parameters for log-spectrogram building.
   * Units:
   *  - epsilon: linear magnitude floor (unitless)
   *  - clip_db: dB range clipping after normalization, ±clip_db
   *  - sample_rate_hz: Hz
   *  - hop_size: samples (STFT hop)
   *  - fft_size: samples (STFT size)
   *  - band_low_hz / band_high_hz: Hz inclusive
   *  - whiten_radius_bins: bins radius for moving-average whitening (0 disables)
   */
  struct LogSpecParams {
    float epsilon{1e-8f};
    bool per_frame_median_subtract{true};
    std::uint16_t whiten_radius_bins{20}; // 0 = disabled
    float clip_db{6.0f}; // clamp post-normalization to ±clip_db
    afp::util::SampleRateHz sample_rate_hz{};
    std::uint32_t hop_size{}; // samples
    std::uint32_t fft_size{}; // samples
    // Band-limiting (inclusive)
    float band_low_hz{300.f};
    float band_high_hz{3500.f};
  };

  /**
   * Build a band-limited, log-compressed, normalized spectrogram from complex STFT.
   * Thread-safety: YES (stateless; no shared mutable state).
   */
  class ISpectrogramBuilder {
  public:
    virtual ~ISpectrogramBuilder() = default;

    /**
     * Purpose:
     *  - Convert complex spectra (frames × bins) to a log-magnitude spectrogram in dB,
     *    apply per-frame normalization (median subtract), optional spectral whitening,
     *    clip dynamic range, and band-limit to [band_low_hz, band_high_hz].
     *
     * Preconditions:
     *  - spec_cpx.num_frames >= 0
     *  - spec_cpx.num_bins   == fft_size/2 + 1 (from params)
     *  - params.sample_rate_hz > 0, params.fft_size > 0
     *  - 0 < band_low_hz < band_high_hz <= sample_rate_hz/2 (will be clamped)
     *
     * Postconditions:
     *  - Returns afp::util::Spectrogram with:
     *      num_bins   = (# bins in requested band)
     *      num_frames = spec_cpx.num_frames
     *      log_mag[i] = dB values after normalization/whitening/clipping
     *      metadata set: sample_rate_hz, hop_size, fft_size
     *
     * Units:
     *  - Input: complex spectrum (linear magnitude)
     *  - Output: dB (20·log10(|X| + epsilon))
     *
     * Complexity:
     *  - O(T * F) for magnitude + log + normalization
     *  - O(T * F * R) for whitening (R = 2*whiten_radius_bins + 1) if enabled
     *
     * Thread-safety:
     *  - YES (no internal state; uses only stack/local buffers).
     */
    virtual afp::util::Expected<afp::util::Spectrogram>
    build(const afp::util::ComplexSpectra &spec_cpx,
          const LogSpecParams &params) = 0;
  };

  class IFeaturesFactory {
  public:
    virtual ~IFeaturesFactory() = default;

    virtual std::unique_ptr<ISpectrogramBuilder> create_spectrogram_builder() = 0;
  };

  /** Default KFR-based features factory. */
  std::unique_ptr<IFeaturesFactory> make_default_features_factory();
} // namespace afp::features
