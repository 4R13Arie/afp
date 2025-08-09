#pragma once
#include <cstdint>
#include <memory>

#include <kfr/all.hpp>     // univector, complex
#include <kfr/dft.hpp>     // kfr::dft_plan_real<float>
#include "afp/util/util.hpp"

namespace afp::stft {
//==============================
// Configuration
//==============================

/**
 * STFT framing configuration.
 * Units:
 *  - frame_size: samples (must be > 0)
 *  - hop_size:   samples (0 < hop_size <= frame_size)
 */
struct FramerConfig {
  std::uint32_t frame_size{1024}; // samples
  std::uint32_t hop_size{256}; // samples
  bool pad_end{true}; // if true, zero-pad tail to include a final partial frame
};

/** Supported window types for STFT. */
enum class WindowType { kHann };

/** FFT configuration (real → complex). Units: samples. */
struct FftConfig {
  std::uint32_t fft_size{1024}; // must equal frame_size
};

//==============================
// Interfaces
//==============================

/**
 * Split a mono PCM stream into overlapping frames.
 * Thread-safety: YES (stateless; no shared state).
 */
class IFramer {
public:
  virtual ~IFramer() = default;

  /**
   * Purpose: Segment input PCM into frames according to FramerConfig.
   * Preconditions:
   *  - in.sample_rate_hz > 0
   *  - cfg.frame_size > 0
   *  - 0 < cfg.hop_size <= cfg.frame_size
   * Postconditions:
   *  - Returns FrameBlock with:
   *      frame_size == cfg.frame_size
   *      hop_size   == cfg.hop_size
   *      num_frames >= 0
   *      data size  == num_frames * frame_size (zero-padded if pad_end)
   * Units: samples.
   * Complexity: O(N) over input sample count.
   * Thread-safety: YES.
   */
  virtual afp::util::Expected<afp::util::FrameBlock>
  segment(const afp::util::PcmSpan& in, const FramerConfig& cfg) = 0;
};

/**
 * Apply an analysis window to each frame.
 * Thread-safety: YES (stateless; precomputes per call).
 */
class IWindow {
public:
  virtual ~IWindow() = default;

  /**
   * Purpose: Apply the chosen window (e.g., Hann) to every frame.
   * Preconditions:
   *  - frames.frame_size > 0
   * Postconditions:
   *  - Returns a new FrameBlock with the same sizes; data windowed.
   * Units: samples (frame units).
   * Complexity: O(F * N) where F is num_frames and N is frame_size.
   * Thread-safety: YES.
   */
  virtual afp::util::Expected<afp::util::FrameBlock>
  apply(const afp::util::FrameBlock& frames, WindowType wt) = 0;
};

/**
 * Real→Complex FFT driver using kfr::dft_plan_real<float>.
 * Thread-safety: NOT thread-safe (plan & scratch owned by instance).
 */
class IFFT {
public:
  virtual ~IFFT() = default;

  /**
   * Purpose: Compute FFT on each frame (real→complex).
   * Preconditions:
   *  - cfg.fft_size == frames.frame_size
   *  - frames.num_frames >= 0
   * Postconditions:
   *  - ComplexSpectra with:
   *      num_bins   == fft_size/2 + 1
   *      num_frames == frames.num_frames
   *      bins.size  == num_bins * num_frames
   * Units:
   *  - fft_size: samples; frequency bins map to Hz via bin * fs / fft_size.
   * Complexity: O(F * N log N)
   * Thread-safety: NO (mutable internal plan/scratch).
   */
  virtual afp::util::Expected<afp::util::ComplexSpectra>
  forward_r2c(const afp::util::FrameBlock& frames, const FftConfig& cfg) = 0;

  /**
   * Purpose: Pre-build/prepare the plan and scratch for the given FFT size.
   * Preconditions:
   *  - cfg.fft_size > 0
   * Postconditions:
   *  - Subsequent calls to forward_r2c with same size avoid re-planning.
   * Units: samples.
   * Complexity: O(N log N) one-time for planning, implementation-defined.
   * Thread-safety: NO.
   */
  virtual void warmup(const FftConfig& cfg, std::uint32_t max_frames) = 0;
};

/**
 * Convenience orchestrator for STFT (framing + windowing + FFT).
 * Thread-safety: NOT thread-safe (holds an internal FFT instance).
 */
class IStftDriver {
public:
  virtual ~IStftDriver() = default;

  /**
   * Purpose: One-shot STFT (frames → window → real→complex FFT).
   * Preconditions:
   *  - in.sample_rate_hz > 0
   *  - fr.frame_size == fft.fft_size
   * Postconditions:
   *  - ComplexSpectra as per IFFT::forward_r2c.
   * Units: samples; Hz derived from caller context.
   * Complexity: O(N) for framing + O(F*N) for window + O(F*N log N) for FFT.
   * Thread-safety: NO (internal plan).
   */
  virtual afp::util::Expected<afp::util::ComplexSpectra>
  run(const afp::util::PcmSpan& in,
      const FramerConfig& fr, WindowType wt, const FftConfig& fft) = 0;
};

/** Factory for STFT components. */
class IStftFactory {
public:
  virtual ~IStftFactory() = default;

  virtual std::unique_ptr<IFramer> create_framer() = 0;

  virtual std::unique_ptr<IWindow> create_window() = 0;

  virtual std::unique_ptr<IFFT> create_fft() = 0;

  virtual std::unique_ptr<IStftDriver> create_driver() = 0;
};

/** Default KFR-based STFT factory. */
std::unique_ptr<IStftFactory> make_default_stft_factory();
} // namespace afp::stft
