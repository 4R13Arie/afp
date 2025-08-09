#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

namespace testio {
  // --- deterministic RNG ---
  inline std::mt19937 &rng() {
    static std::mt19937 gen(12345u);
    return gen;
  }

  // --- sine/impulse/silence generators (interleaved or planar) ---
  inline std::vector<float> make_sine(float freq_hz, float sr, size_t frames, float amp = 0.5f) {
    std::vector<float> x(frames);
    const float w = 2.0f * float(M_PI) * (freq_hz / sr);
    for (size_t n = 0; n < frames; ++n) x[n] = amp * std::sin(w * float(n));
    return x;
  }

  inline std::vector<float> make_silence(size_t frames) { return std::vector<float>(frames, 0.0f); }

  inline std::vector<float> make_impulsive(size_t frames, float amp = 1.0f, size_t every = 16) {
    std::vector<float> x(frames, 0.0f);
    for (size_t i = 0; i < frames; i += every) x[i] = amp;
    return x;
  }

  // interleave planar channels into interleaved buffer
  inline std::vector<float> interleave(const std::vector<std::vector<float> > &ch) {
    if (ch.empty()) return {};
    const size_t C = ch.size();
    const size_t N = ch[0].size();
    std::vector<float> out(C * N);
    for (size_t n = 0; n < N; ++n)
      for (size_t c = 0; c < C; ++c)
        out[n * C + c] = ch[c][n];
    return out;
  }

  // --- RMS helpers ---
  inline float compute_rms(std::span<const float> x) {
    if (x.empty()) return 0.0f;
    double acc = 0.0;
    for (float v: x) acc += double(v) * double(v);
    return float(std::sqrt(acc / double(x.size())));
  }

  inline float rms_dbfs(std::span<const float> x) {
    float r = compute_rms(x);
    if (r <= 1e-30f) return -120.0f;
    return 20.0f * std::log10(r);
  }

  // --- temp file helpers ---
  class TempFile {
  public:
    explicit TempFile(std::string stem = "afp_test") {
      auto dir = std::filesystem::temp_directory_path();
      path_ = (dir / (stem + "-" + std::to_string(::getpid()) + "-" + std::to_string(counter_++) + ".bin")).string();
    }

    ~TempFile() {
      std::error_code ec;
      std::filesystem::remove(path_, ec);
    }

    const std::string &path() const { return path_; }

    void write(const void *data, size_t bytes) {
      std::ofstream ofs(path_, std::ios::binary | std::ios::trunc);
      ofs.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    }

  private:
    inline static std::atomic<uint64_t> counter_{0};
    std::string path_;
  };

  // --- tiny WAV writers (PCM16 / float32 interleaved) ---
#pragma pack(push, 1)
  struct RiffHeader {
    char riff[4];
    uint32_t size;
    char wave[4];
  };

  struct FmtChunk {
    char id[4];
    uint32_t size;
    uint16_t audio_fmt;
    uint16_t ch;
    uint32_t sr;
    uint32_t br;
    uint16_t ba;
    uint16_t bits;
  };

  struct DataChunk {
    char id[4];
    uint32_t size;
  };
#pragma pack(pop)

  inline std::vector<uint8_t> write_wav_f32(const std::vector<float> &interleaved, uint16_t ch, uint32_t sr) {
    const uint32_t frames = ch ? (uint32_t) (interleaved.size() / ch) : 0u;
    const uint32_t data_bytes = frames * ch * 4u;
    RiffHeader rh{
      {'R', 'I', 'F', 'F'}, 4 + 8 + static_cast<uint32_t>(sizeof(FmtChunk)) + 8 + data_bytes, {'W', 'A', 'V', 'E'}
    };
    FmtChunk fmt{{'f', 'm', 't', ' '}, 16, 3 /*IEEE float*/, ch, sr, sr * ch * 4u, (uint16_t) (ch * 4u), 32};
    DataChunk dc{{'d', 'a', 't', 'a'}, data_bytes};

    std::vector<uint8_t> out(sizeof(rh) + sizeof(fmt) + sizeof(dc) + data_bytes);
    uint8_t *p = out.data();
    std::memcpy(p, &rh, sizeof(rh));
    p += sizeof(rh);
    std::memcpy(p, &fmt, sizeof(fmt));
    p += sizeof(fmt);
    std::memcpy(p, &dc, sizeof(dc));
    p += sizeof(dc);
    std::memcpy(p, interleaved.data(), data_bytes);
    return out;
  }

  inline std::vector<uint8_t> write_wav_s16(const std::vector<float> &interleaved, uint16_t ch, uint32_t sr) {
    const uint32_t frames = ch ? (uint32_t) (interleaved.size() / ch) : 0u;
    const uint32_t data_bytes = frames * ch * 2u;
    RiffHeader rh{
      {'R', 'I', 'F', 'F'}, 4 + 8 + static_cast<uint32_t>(sizeof(FmtChunk)) + 8 + data_bytes, {'W', 'A', 'V', 'E'}
    };
    FmtChunk fmt{{'f', 'm', 't', ' '}, 16, 1 /*PCM*/, ch, sr, sr * ch * 2u, (uint16_t) (ch * 2u), 16};
    DataChunk dc{{'d', 'a', 't', 'a'}, data_bytes};

    std::vector<uint8_t> out(sizeof(rh) + sizeof(fmt) + sizeof(dc) + data_bytes);
    uint8_t *p = out.data();
    std::memcpy(p, &rh, sizeof(rh));
    p += sizeof(rh);
    std::memcpy(p, &fmt, sizeof(fmt));
    p += sizeof(fmt);
    std::memcpy(p, &dc, sizeof(dc));
    p += sizeof(dc);
    // convert float [-1,1] → int16
    for (size_t i = 0; i < frames * ch; ++i) {
      float v = std::clamp(interleaved[i], -1.0f, 1.0f);
      int16_t s = (int16_t) std::lrintf(v * 32767.0f);
      std::memcpy(p + i * 2, &s, 2);
    }
    return out;
  }

  // Load asset file into memory; return empty vector if missing.
  inline std::vector<uint8_t> load_asset(std::string_view name) {
#ifndef TEST_ASSETS_DIR
  (void)name;
  return {};
#else
    auto p = std::filesystem::path(TEST_ASSETS_DIR) / std::string(name);
    if (!std::filesystem::exists(p)) return {};
    std::ifstream ifs(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(ifs)), {});
#endif
  }
} // namespace testio
