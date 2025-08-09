#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <span>

#include <kfr/all.hpp>          // KFR: univector, complex, math, FFT backends
#include <tl/expected.hpp>      // tl::expected

namespace afp::util {
// -----------------------------
// Error domain (no exceptions)
// -----------------------------

/**
 * Recoverable errors for util-level operations.
 * No exceptions are thrown by this module.
 *
 * Meanings:
 *  - InvalidArgument: argument value/range preconditions violated
 *  - SizeMismatch: shape/length/buffer size mismatch
 *  - AlignmentError: buffer alignment unsuitable for SIMD/KFR assumptions
 *  - OutOfMemory: allocation failed or growth could not be satisfied
 *  - IOError: file or device I/O failure
 *  - DecodeError: codec/parse error at I/O boundary
 *  - UnsupportedFormat: format/container not supported by current build
 *  - DspError: generic DSP failure (numerical or backend)
 *  - IndexCorrupt: on-disk index corruption detected
 *  - NotFound: resource/key missing (not a hard failure in many lookups)
 *  - Unavailable: subsystem not initialized/open or currently unavailable
 *  - Timeout: operation exceeded allowed time
 *  - Internal: invariant broken or unexpected state (bug)
 */
enum class UtilError : std::uint16_t {
  None = 0,
  InvalidArgument,
  SizeMismatch,
  AlignmentError,
  OutOfMemory,
  IOError,
  DecodeError,
  UnsupportedFormat,
  DspError,
  IndexCorrupt,
  NotFound,
  Unavailable,
  Timeout,
  Internal,
  ResourceExhausted
};

/** Short, stable name for an error. Thread-safe. */
std::string_view error_name(UtilError) noexcept;

/** Human-friendly description. Thread-safe. */
std::string_view error_description(UtilError) noexcept;

/** Project-wide expected alias. Prefer returning this in APIs that can fail. */
template <typename T>
using Expected = tl::expected<T, UtilError>;

// -----------------------------
// Scalar/time/index aliases
// -----------------------------

using TimeMs = std::uint32_t; // milliseconds
using SampleRateHz = std::uint32_t; // Hertz
using FrameIndex = std::uint32_t; // frame number
using BinIndex = std::uint16_t; // frequency bin index
using TrackId = std::uint64_t; // catalog track identifier

// -----------------------------
// Buffer/view conventions
// -----------------------------
//
// Policy:
//  * Own DSP-sized buffers with kfr::univector<T> (SIMD-friendly).
//  * Accept std::span<T> / std::span<const T> on public boundaries (zero copy).
//  * Use std::span for non-owning slices (no KFR view types).
//  * No global state; all types are trivially movable; thread-safe when treated as immutable.
//

/** Owning mono PCM buffer (float32 normalized to [-1, 1]). */
struct PcmBuffer {
  kfr::univector<float> samples; // owning, contiguous, SIMD-capable
  SampleRateHz sample_rate_hz{};
  // Thread-safety: safe for concurrent const access; mutations must be external.
};

/** Non-owning API boundary view of mono PCM. */
struct PcmSpan {
  SampleRateHz sample_rate_hz{};
  std::span<const float> samples; // no ownership; lifetime managed by caller
};

/** Compatibility alias for “PCM view” (previously a KFR view). */
using PcmView = std::span<const float>;

/** STFT framing block (owning, row-major [frame][sample]). */
struct FrameBlock {
  std::uint32_t frame_size{}; // samples per frame
  std::uint32_t hop_size{}; // samples
  std::uint32_t num_frames{}; // number of frames
  kfr::univector<float> data; // size = num_frames * frame_size

  // Access contract: frame i maps to contiguous subrange of length frame_size.
  // Thread-safety: safe for concurrent const access; not thread-safe for mutation.
};

/** Complex spectra container (owning). Prefer AoS with kfr::complex<float>. */
struct ComplexSpectra {
  std::uint16_t num_bins{}; // = fft_size/2 + 1
  std::uint32_t num_frames{};
  kfr::univector<kfr::complex<float>> bins; // size = num_frames * num_bins
  // Thread-safety: safe for concurrent const access.
};

/** Log spectrogram (owning, compressed/normalized units e.g., dB-like). */
struct Spectrogram {
  std::uint16_t num_bins{};
  std::uint32_t num_frames{};
  kfr::univector<float> log_mag;
  // row-major [frame][bin], size = num_frames * num_bins
  SampleRateHz sample_rate_hz{};
  std::uint32_t hop_size{}; // samples (time resolution)
  std::uint32_t fft_size{}; // samples (freq resolution)
  // Thread-safety: safe for concurrent const access.
};

// -----------------------------
// Lightweight feature/fp/index types (non-DSP sized)
// -----------------------------

struct Peak {
  FrameIndex t{}; // frame index
  BinIndex f{}; // frequency bin (0..num_bins-1)
  float strength_db{}; // relative strength post-normalization
  // Trivially copyable; thread-safe by value semantics.
};

struct FingerprintPair {
  BinIndex f_anchor{};
  BinIndex f_target{};
  TimeMs delta_ms{}; // target_time - anchor_time
  TimeMs t_anchor_ms{}; // absolute time of anchor from track start
};

struct FingerprintKey {
  std::uint32_t value{}; // packed (f_a<<23)|(f_t<<14)|delta_ms
};

struct Posting {
  TrackId track{};
  TimeMs t_anchor_ms{};
};

struct CandidateMatch {
  TrackId track{};
  TimeMs offset_ms{}; // best alignment offset
  std::uint32_t inliers{}; // votes in consensus cluster
  float score{}; // implementation-defined composite
  TimeMs span_ms{}; // temporal span of inliers
};

struct MatchResult {
  std::vector<CandidateMatch> topk; // small vector; not a DSP buffer
};

// -----------------------------
// Zero-copy helpers (header-only, noexcept)
// -----------------------------

/** Make a view from a PCM span (no copy). Caller ensures span lifetime. */
inline PcmView as_kfr_view(PcmSpan s) noexcept {
  // Kept name for source compatibility; now returns std::span<const float>.
  return s.samples;
}

/** Convenience: view over PcmBuffer samples (const). */
inline std::span<const float> as_span(const PcmBuffer& b) noexcept {
  return std::span<const float>(b.samples.data(), b.samples.size());
}

/** Convenience: view over PcmBuffer samples (mutable). */
inline std::span<float> as_span(PcmBuffer& b) noexcept {
  return std::span<float>(b.samples.data(), b.samples.size());
}

/** Slice a frame i from FrameBlock as a const span. Preconditions: i < num_frames. */
inline std::span<const float> frame_view(const FrameBlock& fb,
                                         std::uint32_t i) noexcept {
  const std::size_t offset = static_cast<std::size_t>(i) * fb.frame_size;
  return std::span<const float>(fb.data.data() + offset, fb.frame_size);
}

/** Mutable slice of a frame i from FrameBlock. Preconditions: i < num_frames. */
inline std::span<float> frame_view(FrameBlock& fb, std::uint32_t i) noexcept {
  const std::size_t offset = static_cast<std::size_t>(i) * fb.frame_size;
  return std::span<float>(fb.data.data() + offset, fb.frame_size);
}

/** Slice complex spectra row i (const). Preconditions: i < num_frames. */
inline std::span<const kfr::complex<float>>
spectrum_row(const ComplexSpectra& S, std::uint32_t i) noexcept {
  const std::size_t offset = static_cast<std::size_t>(i) * S.num_bins;
  return {S.bins.data() + offset, S.num_bins};
}

/** Slice complex spectra row i (mutable). Preconditions: i < num_frames. */
inline std::span<kfr::complex<float>>
spectrum_row(ComplexSpectra& S, std::uint32_t i) noexcept {
  const std::size_t offset = static_cast<std::size_t>(i) * S.num_bins;
  return {S.bins.data() + offset, S.num_bins};
}

/** Slice spectrogram row i (const). Preconditions: i < num_frames. */
inline std::span<const float> spectrogram_row(const Spectrogram& G,
                                              std::uint32_t i) noexcept {
  const std::size_t offset = static_cast<std::size_t>(i) * G.num_bins;
  return {G.log_mag.data() + offset, G.num_bins};
}

/** Slice spectrogram row i (mutable). Preconditions: i < num_frames. */
inline std::span<float> spectrogram_row(Spectrogram& G,
                                        std::uint32_t i) noexcept {
  const std::size_t offset = static_cast<std::size_t>(i) * G.num_bins;
  return std::span(G.log_mag.data() + offset, G.num_bins);
}
} // namespace afp::util
