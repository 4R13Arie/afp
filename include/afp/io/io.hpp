#pragma once
#include <cstddef>
#include <memory>
#include <span>
#include <string_view>

#include <tl/expected.hpp>

#include "afp/util/util.hpp"

namespace afp::io {
  /**
   * Parameters controlling decode-time processing.
   * All units are annotated in field comments.
   */
  struct DecodeParams {
    bool normalize_rms = true; ///< If true, scale to target RMS.
    float target_rms_dbfs = -20.0f; ///< dBFS target for RMS normalization.
    bool dither = false; ///< Ignored for float32 output; reserved.
  };

  /**
   * Downmix multi-channel PCM to mono (energy-preserving).
   * Implementations should average channels or use energy weights.
   */
  class IDownmixer {
  public:
    virtual ~IDownmixer() = default;

    /**
     * Purpose: Convert >=1 channel PCM streams to mono.
     * Preconditions:
     *  - channels.size() >= 1
     *  - All channels have identical sample_rate_hz and length
     *  - Input spans remain valid for the duration of the call
     * Postconditions:
     *  - Returned PcmBuffer contains mono float32 samples in [-1, 1]
     *  - sample_rate_hz preserved from inputs
     * Units:
     *  - Samples: unitless normalized floats
     *  - Sample rate: Hz
     * Complexity: O(N * C) over total samples (N frames, C channels)
     * Thread-safety: YES (stateless, no shared mutable state)
     */
    [[nodiscard]] virtual util::Expected<util::PcmBuffer>
    to_mono(std::span<const util::PcmSpan> channels) const = 0;
  };

  /**
   * Decode audio from file or memory into mono float32 PCM (owning).
   * Implementations may internally downmix using IDownmixer.
   */
  class IAudioDecoder {
  public:
    virtual ~IAudioDecoder() = default;

    /**
     * Purpose: Decode an entire file to mono float32 PCM.
     * Preconditions:
     *  - path is a readable file (UTF-8 or native encoding)
     *  - params are within documented ranges (e.g., reasonable target RMS)
     * Postconditions:
     *  - On success, returns mono PCM in [-1, 1] with sample_rate_hz set
     *  - If params.normalize_rms, overall RMS ~= target_rms_dbfs (clamped if needed)
     * Units:
     *  - Sample rate: Hz; Time: frames/s
     * Complexity: O(N) over decoded sample frames
     * Thread-safety: NO (decoder instances maintain internal scratch); use one instance per thread
     */
    virtual util::Expected<util::PcmBuffer>
    decode_file(std::string_view path, const DecodeParams &params) = 0;

    /**
     * Purpose: Decode from an in-memory container (entire encoded stream).
     * Preconditions:
     *  - data holds a full, valid encoded stream (container+codec)
     *  - params as per decode_file
     * Postconditions:
     *  - Same as decode_file
     * Complexity: O(N)
     * Thread-safety: NO (per-instance scratch); use separate instances per thread
     */
    virtual util::Expected<util::PcmBuffer>
    decode_bytes(std::span<const std::byte> data, const DecodeParams &params) = 0;
  };

  /**
   * Abstract factory for decoders. Concrete factories may choose specific backends.
   */
  class IDecoderFactory {
  public:
    virtual ~IDecoderFactory() = default;

    /**
     * Purpose: Create a decoder suitable for WAV/MP3/FLAC based on internal strategy.
     * Preconditions: None.
     * Postconditions: Returned pointer is non-null on success; each instance is independent.
     * Thread-safety: YES (factory is stateless or uses const-only state).
     * Complexity: O(1).
     */
    [[nodiscard]] virtual std::unique_ptr<IAudioDecoder> create_decoder() const = 0;
  };

  /**
   * Create a default, composite-decoder factory that supports:
   *  - WAV via dr_wav (PCM 8/16/24/32-bit, float32)
   *  - MP3 via dr_mp3 (MPEG-1/2 Layer III)
   *  - FLAC via dr_flac (16/24-bit)
   * The decoder auto-detects format by header sniffing or filename extension fallback.
   */
  std::unique_ptr<IDecoderFactory> make_default_decoder_factory();

  /**
   * Create a default downmixer that averages channels (energy-preserving).
   */
  std::unique_ptr<IDownmixer> make_default_downmixer();
} // namespace afp::io
