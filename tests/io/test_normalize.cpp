#include <gtest/gtest.h>
#include "afp/io/io.hpp"
#include "afp/util/util.hpp"
#include "test_utils_io.hpp"

using afp::io::DecodeParams;

static std::unique_ptr<afp::io::IAudioDecoder> new_wav() {
  auto f = afp::io::make_default_decoder_factory();
  auto d = f->create_decoder();
  return d;
}

TEST(Normalize, TargetHitMinus20dBFS) {
  const uint32_t sr = 44100;
  auto s = testio::make_sine(1000.f, float(sr), 2048,
                             10.0f /* intentionally hot */);
  auto wav = testio::write_wav_f32(testio::interleave({s}), 1, sr);
  DecodeParams p;
  p.normalize_rms = true;
  p.target_rms_dbfs = -20.f;

  auto dec = new_wav();
  auto outE = dec->decode_bytes(
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(wav.data()),
                                 wav.size()), p);
  ASSERT_TRUE(outE.has_value());
  const auto& y = outE->samples;

  const float rms_db = testio::rms_dbfs(
      std::span<const float>(y.data(), y.size()));
  ASSERT_NEAR(rms_db, -20.0f, 0.5f);
  for (float v : y) {
    ASSERT_LE(v, 1.0f);
    ASSERT_GE(v, -1.0f);
  };
}

TEST(Normalize, ClampAndNoNaNOnSilence) {
  const uint32_t sr = 16000;
  auto sil = testio::make_silence(1024);
  auto wav = testio::write_wav_s16(testio::interleave({sil}), 1, sr);
  DecodeParams p;
  p.normalize_rms = true;
  p.target_rms_dbfs = -20.f;

  auto dec = new_wav();
  auto outE = dec->decode_bytes(
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(wav.data()),
                                 wav.size()), p);
  ASSERT_TRUE(outE.has_value());
  const auto& y = outE->samples;
  // Remains silence
  for (float v : y) {
    ASSERT_FALSE(std::isnan(v));
    ASSERT_NEAR(v, 0.0f, 1e-7f);
  }
}
