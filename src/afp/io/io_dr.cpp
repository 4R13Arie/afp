#include "afp/io/io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>

#define DR_WAV_IMPLEMENTATION
#include "../../../lib/dr_wav.h"

#define DR_MP3_IMPLEMENTATION
#include "../../../lib/dr_mp3.h"

#define DR_FLAC_IMPLEMENTATION
#include "../../../lib/dr_flac.h"

namespace afp::io {
using afp::util::Expected;
using afp::util::PcmBuffer;
using afp::util::PcmSpan;
using afp::util::SampleRateHz;
using afp::util::TimeMs;
using afp::util::TrackId;
using afp::util::UtilError;

// -------------------------------
// Helpers (internal, file-scope)
// -------------------------------

namespace {

inline bool iequals_ascii(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    char ca = a[i], cb = b[i];
    if ('A' <= ca && ca <= 'Z') ca = static_cast<char>(ca - 'A' + 'a');
    if ('A' <= cb && cb <= 'Z') cb = static_cast<char>(cb - 'A' + 'a');
    if (ca != cb) return false;
  }
  return true;
}

enum class SniffedFormat { Wav, Mp3, Flac, Unknown };

SniffedFormat sniff_header(std::span<const std::byte> data) {
  if (data.size() >= 12) {
    const char* p = reinterpret_cast<const char*>(data.data());
    if (std::memcmp(p, "RIFF", 4) == 0 && std::memcmp(p + 8, "WAVE", 4) == 0) return SniffedFormat::Wav;
  }
  if (data.size() >= 4) {
    const char* p = reinterpret_cast<const char*>(data.data());
    if (std::memcmp(p, "fLaC", 4) == 0) return SniffedFormat::Flac;
    if (std::memcmp(p, "ID3", 3) == 0) return SniffedFormat::Mp3;
    // Frame sync 0xFFEx: quick heuristic for MP3 (MPEG audio frame).
    const unsigned char b0 = static_cast<unsigned char>(p[0]);
    const unsigned char b1 = static_cast<unsigned char>(p[1]);
    if (b0 == 0xFF && (b1 & 0xE0) == 0xE0) return SniffedFormat::Mp3;
  }
  return SniffedFormat::Unknown;
}

SniffedFormat sniff_extension(std::string_view path) {
  std::error_code ec;
  auto ext = std::filesystem::path(path).extension().string();
  if (ext.empty()) return SniffedFormat::Unknown;
  if (iequals_ascii(ext, ".wav"))  return SniffedFormat::Wav;
  if (iequals_ascii(ext, ".mp3"))  return SniffedFormat::Mp3;
  if (iequals_ascii(ext, ".flac")) return SniffedFormat::Flac;
  return SniffedFormat::Unknown;
}

// Compute RMS (root mean square) for mono signal.
float compute_rms(std::span<const float> x) {
  if (x.empty()) return 0.0f;
  double acc = 0.0;
  for (float v : x) acc += static_cast<double>(v) * static_cast<double>(v);
  return static_cast<float>(std::sqrt(acc / static_cast<double>(x.size())));
}

// Apply RMS normalization to target dBFS (clamped to avoid extreme gains).
void normalize_rms_inplace(kfr::univector<float>& mono, float target_dbfs) {
  const float target_lin = std::pow(10.0f, target_dbfs * 0.05f); // 20*log10(s)=dB → s=10^(dB/20)
  const float rms = compute_rms(std::span<const float>(mono.data(), mono.size()));
  if (rms <= std::numeric_limits<float>::min()) return;
  float gain = target_lin / rms;

  // Optional gentle clamp to prevent clipping explosions under very low RMS.
  if (gain > 16.0f) gain = 16.0f;

  // Scale in-place.
  for (auto& v : mono) v *= gain;

  // Safety clamp to [-1, 1] just in case.
  for (auto& v : mono) {
    if (v > 1.0f) v = 1.0f;
    else if (v < -1.0f) v = -1.0f;
  }
}

} // namespace

// -------------------------------
// Default Downmixer
// -------------------------------

class EnergyPreservingDownmixer final : public IDownmixer {
public:
  [[nodiscard]] Expected<PcmBuffer> to_mono(std::span<const PcmSpan> channels) const override {
    if (channels.empty()) return tl::unexpected(UtilError::InvalidArgument);

    const SampleRateHz sr = channels[0].sample_rate_hz;
    const size_t N = channels[0].samples.size();
    for (const auto& ch : channels) {
      if (ch.sample_rate_hz != sr) return tl::unexpected(UtilError::SizeMismatch);
      if (ch.samples.size() != N)   return tl::unexpected(UtilError::SizeMismatch);
    }

    PcmBuffer out;
    out.sample_rate_hz = sr;
    out.samples.resize(N);

    const float invC = 1.0f / static_cast<float>(channels.size());
    for (size_t i = 0; i < N; ++i) {
      float acc = 0.0f;
      for (const auto& ch : channels) acc += ch.samples[i];
      out.samples[i] = acc * invC;
    }
    return out;
  }
};

std::unique_ptr<IDownmixer> make_default_downmixer() {
  return std::make_unique<EnergyPreservingDownmixer>();
}

// -------------------------------
// WAV Decoder (dr_wav)
// -------------------------------
//
// Supported:
//  - Container: RIFF/WAVE
//  - Formats: PCM 8/16/24/32-bit, IEEE float32
//  - Channels: 1..8 (downmixed to mono)
//  - Sample rates: as provided by file
//
// Thread-safety: instance NOT thread-safe (internal scratch during decode).
//

class DrWavDecoder final : public IAudioDecoder {
public:
  explicit DrWavDecoder(std::unique_ptr<IDownmixer> dm)
    : downmixer_(std::move(dm)) {}

  Expected<PcmBuffer> decode_file(std::string_view path, const DecodeParams& params) override {
    drwav wav{};
    if (!drwav_init_file(&wav, std::string(path).c_str(), nullptr))
      return tl::unexpected(UtilError::DecodeError);

    Expected<PcmBuffer> res = decode_impl(wav, params);
    drwav_uninit(&wav);
    return res;
  }

  Expected<PcmBuffer> decode_bytes(std::span<const std::byte> data, const DecodeParams& params) override {
    drwav wav{};
    if (!drwav_init_memory(&wav, data.data(), data.size(), nullptr))
      return tl::unexpected(UtilError::DecodeError);

    Expected<PcmBuffer> res = decode_impl(wav, params);
    drwav_uninit(&wav);
    return res;
  }

private:
  Expected<PcmBuffer> decode_impl(drwav& wav, const DecodeParams& params) {
    const uint32_t channels = wav.channels;
    const uint32_t sr       = wav.sampleRate;
    if (channels == 0 || sr == 0) return tl::unexpected(UtilError::DecodeError);

    // Read all frames as float32 interleaved.
    drwav_uint64 total_frames = wav.totalPCMFrameCount;
    if (total_frames == 0) {
      // Some WAVs may not report; read in chunks.
      total_frames = 0;
    }

    kfr::univector<float> interleaved;
    if (total_frames > 0) {
      interleaved.resize(static_cast<size_t>(total_frames) * channels);
      drwav_uint64 read = drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, interleaved.data());
      interleaved.resize(static_cast<size_t>(read) * channels);
    } else {
      // Fallback streaming read.
      constexpr drwav_uint64 CHUNK = 1u << 16;
      kfr::univector<float> chunk;
      for (;;) {
        chunk.resize(static_cast<size_t>(CHUNK) * channels);
        drwav_uint64 got = drwav_read_pcm_frames_f32(&wav, CHUNK, chunk.data());
        if (got == 0) break;
        chunk.resize(static_cast<size_t>(got) * channels);
        const size_t prev = interleaved.size();
        interleaved.resize(prev + chunk.size());
        std::memcpy(interleaved.data() + prev, chunk.data(), sizeof(float) * chunk.size());
      }
    }

    // Deinterleave → channel spans
    const size_t frames = (channels > 0) ? interleaved.size() / channels : 0;
    if (frames == 0) return tl::unexpected(UtilError::DecodeError);

    std::vector<kfr::univector<float>> ch_data(channels);
    for (uint32_t c = 0; c < channels; ++c) ch_data[c].resize(frames);
    for (size_t i = 0; i < frames; ++i) {
      const float* row = interleaved.data() + i * channels;
      for (uint32_t c = 0; c < channels; ++c) ch_data[c][i] = row[c];
    }

    std::vector<PcmSpan> spans;
    spans.reserve(channels);
    for (uint32_t c = 0; c < channels; ++c) {
      spans.push_back(PcmSpan{ sr, std::span<const float>(ch_data[c].data(), ch_data[c].size()) });
    }

    auto mono = downmixer_->to_mono(spans);
    if (!mono) return tl::unexpected(mono.error());

    if (params.normalize_rms) normalize_rms_inplace(mono->samples, params.target_rms_dbfs);
    // Dither ignored for float32.

    mono->sample_rate_hz = sr;
    return mono;
  }

  std::unique_ptr<IDownmixer> downmixer_;
};

// -------------------------------
// MP3 Decoder (dr_mp3)
// -------------------------------
//
// Supported:
//  - Container/codec: MPEG-1/2 Layer III (CBR/VBR)
//  - Output: float32 interleaved
//  - Channels: 1..2 typical (downmixed to mono)
//  - Sample rates: per stream (e.g., 32000/44100/48000 Hz)
//
// Thread-safety: instance NOT thread-safe.
//

class DrMp3Decoder final : public IAudioDecoder {
public:
  explicit DrMp3Decoder(std::unique_ptr<IDownmixer> dm)
    : downmixer_(std::move(dm)) {}

  Expected<PcmBuffer> decode_file(std::string_view path, const DecodeParams& params) override {
    drmp3 mp3{};
    if (!drmp3_init_file(&mp3, std::string(path).c_str(), nullptr))
      return tl::unexpected(UtilError::DecodeError);

    Expected<PcmBuffer> res = decode_impl(mp3, params);
    drmp3_uninit(&mp3);
    return res;
  }

  Expected<PcmBuffer> decode_bytes(std::span<const std::byte> data, const DecodeParams& params) override {
    drmp3 mp3{};
    if (!drmp3_init_memory(&mp3, data.data(), data.size(), nullptr))
      return tl::unexpected(UtilError::DecodeError);

    Expected<PcmBuffer> res = decode_impl(mp3, params);
    drmp3_uninit(&mp3);
    return res;
  }

private:
  Expected<PcmBuffer> decode_impl(drmp3& mp3, const DecodeParams& params) {
    const uint32_t channels = mp3.channels;
    const uint32_t sr       = mp3.sampleRate;
    if (channels == 0 || sr == 0) return tl::unexpected(UtilError::DecodeError);

    kfr::univector<float> interleaved;
    constexpr drmp3_uint64 CHUNK = 1u << 14; // frames per read
    kfr::univector<float> chunk;
    for (;;) {
      chunk.resize(static_cast<size_t>(CHUNK) * channels);
      drmp3_uint64 got = drmp3_read_pcm_frames_f32(&mp3, CHUNK, chunk.data());

      if (got == 0) break;
      chunk.resize(static_cast<size_t>(got) * channels);
      const size_t prev = interleaved.size();
      interleaved.resize(prev + chunk.size());
      std::memcpy(interleaved.data() + prev, chunk.data(), sizeof(float) * chunk.size());
    }

    const size_t frames = (channels > 0) ? interleaved.size() / channels : 0;
    if (frames == 0) return tl::unexpected(UtilError::DecodeError);

    std::vector<kfr::univector<float>> ch_data(channels);
    for (uint32_t c = 0; c < channels; ++c) ch_data[c].resize(frames);
    for (size_t i = 0; i < frames; ++i) {
      const float* row = interleaved.data() + i * channels;
      for (uint32_t c = 0; c < channels; ++c) ch_data[c][i] = row[c];
    }

    std::vector<PcmSpan> spans;
    spans.reserve(channels);
    for (uint32_t c = 0; c < channels; ++c) {
      spans.push_back(PcmSpan{ sr, std::span<const float>(ch_data[c].data(), ch_data[c].size()) });
    }

    auto mono = downmixer_->to_mono(spans);
    if (!mono) return tl::unexpected(mono.error());

    if (params.normalize_rms) normalize_rms_inplace(mono->samples, params.target_rms_dbfs);
    mono->sample_rate_hz = sr;
    return mono;
  }

  std::unique_ptr<IDownmixer> downmixer_;
};

// -------------------------------
// FLAC Decoder (dr_flac)
// -------------------------------
//
// Supported:
//  - Container/codec: FLAC
//  - Bit depths: 16/24-bit typical
//  - Channels: 1..8 (downmixed to mono)
//  - Output: float32
//
// Thread-safety: instance NOT thread-safe.
//

class DrFlacDecoder final : public IAudioDecoder {
public:
  explicit DrFlacDecoder(std::unique_ptr<IDownmixer> dm)
    : downmixer_(std::move(dm)) {}

  Expected<PcmBuffer> decode_file(std::string_view path, const DecodeParams& params) override {
    drflac* flac = drflac_open_file(std::string(path).c_str(), nullptr);
    if (!flac) return tl::unexpected(UtilError::DecodeError);
    Expected<PcmBuffer> res = decode_impl(*flac, params);
    drflac_close(flac);
    return res;
  }

  Expected<PcmBuffer> decode_bytes(std::span<const std::byte> data, const DecodeParams& params) override {
    drflac* flac = drflac_open_memory(data.data(), data.size(), nullptr);
    if (!flac) return tl::unexpected(UtilError::DecodeError);
    Expected<PcmBuffer> res = decode_impl(*flac, params);
    drflac_close(flac);
    return res;
  }

private:
  Expected<PcmBuffer> decode_impl(drflac& flac, const DecodeParams& params) {
    const uint32_t channels = flac.channels;
    const uint32_t sr       = flac.sampleRate;
    if (channels == 0 || sr == 0) return tl::unexpected(UtilError::DecodeError);

    kfr::univector<float> interleaved;
    interleaved.resize(static_cast<size_t>(flac.totalPCMFrameCount) * channels);
    drflac_uint64 read = drflac_read_pcm_frames_f32(&flac, flac.totalPCMFrameCount, interleaved.data());
    interleaved.resize(static_cast<size_t>(read) * channels);

    const size_t frames = (channels > 0) ? interleaved.size() / channels : 0;
    if (frames == 0) return tl::unexpected(UtilError::DecodeError);

    std::vector<kfr::univector<float>> ch_data(channels);
    for (uint32_t c = 0; c < channels; ++c) ch_data[c].resize(frames);
    for (size_t i = 0; i < frames; ++i) {
      const float* row = interleaved.data() + i * channels;
      for (uint32_t c = 0; c < channels; ++c) ch_data[c][i] = row[c];
    }

    std::vector<PcmSpan> spans;
    spans.reserve(channels);
    for (uint32_t c = 0; c < channels; ++c) {
      spans.push_back(PcmSpan{ sr, std::span<const float>(ch_data[c].data(), ch_data[c].size()) });
    }

    auto mono = downmixer_->to_mono(spans);
    if (!mono) return tl::unexpected(mono.error());

    if (params.normalize_rms) normalize_rms_inplace(mono->samples, params.target_rms_dbfs);
    mono->sample_rate_hz = sr;
    return mono;
  }

  std::unique_ptr<IDownmixer> downmixer_;
};

// -------------------------------
/* Composite decoder & Factory */
// -------------------------------

class CompositeDecoder final : public IAudioDecoder {
public:
  explicit CompositeDecoder(std::unique_ptr<IDownmixer> dm)
  {
    decoders_.reserve(3);
    // Each concrete decoder gets its own downmixer instance.
    decoders_.push_back(std::make_unique<DrWavDecoder>(std::make_unique<EnergyPreservingDownmixer>()));
    decoders_.push_back(std::make_unique<DrMp3Decoder>(std::make_unique<EnergyPreservingDownmixer>()));
    decoders_.push_back(std::make_unique<DrFlacDecoder>(std::make_unique<EnergyPreservingDownmixer>()));
    downmixer_ = std::move(dm); // kept only to adhere to signature symmetry; not used here
  }

  Expected<PcmBuffer> decode_file(std::string_view path, const DecodeParams& params) override {
    // Try preferred decoder by extension; fallback to trial.
    const SniffedFormat ext = sniff_extension(path);
    if (auto out = try_by_kind(ext, path, params)) return out;

    // Fallback: try each decoder.
    for (auto& d : decoders_) {
      if (auto out = d->decode_file(path, params)) return out;
    }
    return tl::unexpected(UtilError::DecodeError);
  }

  Expected<PcmBuffer> decode_bytes(std::span<const std::byte> data, const DecodeParams& params) override {
    const SniffedFormat fmt = sniff_header(data);
    if (auto out = try_by_kind(fmt, data, params)) return out;

    for (auto& d : decoders_) {
      if (auto out = d->decode_bytes(data, params)) return out;
    }
    return tl::unexpected(UtilError::DecodeError);
  }

private:
  Expected<PcmBuffer> try_by_kind(SniffedFormat kind, std::string_view path, const DecodeParams& params) {
    switch (kind) {
      case SniffedFormat::Wav:  return decoders_[0]->decode_file(path, params);
      case SniffedFormat::Mp3:  return decoders_[1]->decode_file(path, params);
      case SniffedFormat::Flac: return decoders_[2]->decode_file(path, params);
      default: break;
    }
    return tl::unexpected(UtilError::Unavailable);
  }

  Expected<PcmBuffer> try_by_kind(SniffedFormat kind, std::span<const std::byte> data, const DecodeParams& params) {
    switch (kind) {
      case SniffedFormat::Wav:  return decoders_[0]->decode_bytes(data, params);
      case SniffedFormat::Mp3:  return decoders_[1]->decode_bytes(data, params);
      case SniffedFormat::Flac: return decoders_[2]->decode_bytes(data, params);
      default: break;
    }
    return tl::unexpected(UtilError::Unavailable);
  }

private:
  std::vector<std::unique_ptr<IAudioDecoder>> decoders_;
  std::unique_ptr<IDownmixer> downmixer_; // unused in composite but kept for symmetry
};

class DefaultDecoderFactory final : public IDecoderFactory {
public:
  [[nodiscard]] std::unique_ptr<IAudioDecoder> create_decoder() const override {
    // Composite with internal WAV/MP3/FLAC decoders
    return std::make_unique<CompositeDecoder>(std::make_unique<EnergyPreservingDownmixer>());
  }
};

std::unique_ptr<IDecoderFactory> make_default_decoder_factory() {
  return std::make_unique<DefaultDecoderFactory>();
}

} // namespace afp::io
