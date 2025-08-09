#pragma once
#include <memory>
#include <span>
#include <cstdint>

#include <kfr/all.hpp>   // univector, biquads, convolution_filter, resampler
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>
#include "afp/util/util.hpp"

namespace afp::dsp {
//==============================
// Parameters
//==============================

/**
 * Biquad filter configuration.
 * Units:
 *  - cutoff_hz: Hz (0 < cutoff_hz < 0.5 * sample_rate_hz)
 *  - Q: unitless (default ~0.707 for Butterworth-like response)
 */
struct BiquadParams {
  float cutoff_hz{}; ///< Hz
  float Q{0.707f}; ///< unitless
};

/**
 * Pre-emphasis filter parameters.
 * y[n] = x[n] - alpha * x[n-1]
 * Typical alpha ~ 0.95 .. 0.98
 */
struct PreEmphasisParams {
  float alpha{0.97f}; ///< unitless, 0..1
};

/**
 * Fractional resampling configuration (KFR resampler).
 * Units:
 *  - target_hz: Hz (output sample rate)
 *  - quality: implementation-defined quality level (KFR ctor expects an int)
 */
struct ResampleParams {
  afp::util::SampleRateHz target_hz{}; ///< Hz
  int quality{8};
  ///< KFR resampler quality (0=fast .. N=best; depends on KFR build)
};

//==============================
// High-pass biquad (KFR)
//==============================

/**
 * Stateless wrapper around KFR biquad high-pass.
 * Thread-safe: YES (no internal state).
 */
class IHighPass {
public:
  virtual ~IHighPass() = default;

  /**
   * Purpose: Apply 2nd-order HPF via KFR (one-shot, zero-state).
   * Preconditions:
   *  - in.sample_rate_hz > 0
   *  - 0 < hp.cutoff_hz < 0.5 * in.sample_rate_hz
   *  - hp.Q > 0
   * Postconditions:
   *  - Returns mono PCM with same sample rate; size == in.samples.size()
   * Units:
   *  - cutoff_hz in Hz; Q unitless; PCM normalized to [-1, 1]
   * Complexity: O(N) over input sample count.
   * Thread-safety: YES (stateless; uses only stack temporaries).
   */
  virtual afp::util::Expected<afp::util::PcmBuffer>
  process(const afp::util::PcmSpan& in, const BiquadParams& hp) = 0;

  /** No-op (stateless). */
  virtual void reset() noexcept = 0;
};

//==============================
// Low-pass biquad (KFR)
//==============================

/**
 * Stateless wrapper around KFR biquad low-pass.
 * Thread-safe: YES (no internal state).
 */
class ILowPass {
public:
  virtual ~ILowPass() = default;

  /**
   * Purpose: Apply 2nd-order LPF via KFR (one-shot, zero-state).
   * Preconditions:
   *  - in.sample_rate_hz > 0
   *  - 0 < lp.cutoff_hz < 0.5 * in.sample_rate_hz
   *  - lp.Q > 0
   * Postconditions:
   *  - Returns mono PCM with same sample rate; size == in.samples.size()
   * Units: Hz; Q unitless; PCM normalized to [-1, 1]
   * Complexity: O(N).
   * Thread-safety: YES (stateless).
   */
  virtual afp::util::Expected<afp::util::PcmBuffer>
  process(const afp::util::PcmSpan& in, const BiquadParams& lp) = 0;

  /** No-op (stateless). */
  virtual void reset() noexcept = 0;
};

//==============================
// Pre-emphasis (KFR convolution)
//==============================

/**
 * Stateless 1st-order pre-emphasis via KFR convolution_filter with FIR {1, -alpha}.
 * Thread-safe: YES (no internal state).
 */
class IPreEmphasis {
public:
  virtual ~IPreEmphasis() = default;

  /**
   * Purpose: Emphasize high frequencies via y[n] = x[n] - alpha * x[n-1].
   * Preconditions:
   *  - in.sample_rate_hz > 0
   *  - 0 <= alpha <= 1
   * Postconditions:
   *  - Returns mono PCM with same sample rate; size == in.samples.size()
   *    (result of convolution is trimmed to input length)
   * Units: alpha unitless.
   * Complexity: O(N) (tiny FIR with 2 taps).
   * Thread-safety: YES (stateless).
   */
  virtual afp::util::Expected<afp::util::PcmBuffer>
  process(const afp::util::PcmSpan& in, const PreEmphasisParams& p) = 0;

  /** No-op (stateless). */
  virtual void reset() noexcept = 0;
};

//==============================
// Fractional resampler (KFR)
//==============================

/**
 * Stateless one-shot resampling using KFR resampler (constructed per call).
 * Thread-safe: YES (no shared state across calls).
 */
class IResampler {
public:
  virtual ~IResampler() = default;

  /**
   * Purpose: Resample mono PCM to target sample rate (Hz) using KFR.
   * Preconditions:
   *  - in.sample_rate_hz > 0
   *  - rp.target_hz > 0
   * Postconditions:
   *  - Returns mono PCM with sample_rate_hz == rp.target_hz
   *  - Output length ≈ in.samples.size() * (target_hz / in.sample_rate_hz)
   * Units: Hz for sample rates.
   * Complexity: O(N) (as implemented by KFR).
   * Thread-safety: YES (stateless; constructs KFR resampler locally).
   */
  virtual afp::util::Expected<afp::util::PcmBuffer>
  resample(const afp::util::PcmSpan& in, const ResampleParams& rp) = 0;

  /** No-op (stateless). */
  virtual void reset() noexcept = 0;
};

//==============================
// Factory
//==============================

class IDspFactory {
public:
  virtual ~IDspFactory() = default;

  /** Create high-pass (KFR) */
  [[nodiscard]] virtual std::unique_ptr<IHighPass> create_hpf() const = 0;

  /** Create low-pass  (KFR) */
  [[nodiscard]] virtual std::unique_ptr<ILowPass> create_lpf() const = 0;

  /** Create pre-emphasis (KFR convolution) */
  [[nodiscard]] virtual std::unique_ptr<IPreEmphasis> create_preemphasis() const
  = 0;

  /** Create resampler (KFR) */
  [[nodiscard]] virtual std::unique_ptr<IResampler> create_resampler() const =
  0;
};

/** Default KFR-only DSP factory. */
std::unique_ptr<IDspFactory> make_default_dsp_factory();
} // namespace afp::dsp
