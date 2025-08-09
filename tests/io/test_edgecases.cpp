#include <gtest/gtest.h>
#include "afp/io/io.hpp"
#include "afp/util/util.hpp"
#include "test_utils_io.hpp"

TEST(Edge, ZeroFramesWav) {
  const uint32_t sr = 48000;
  std::vector<float> empty; // zero frames
  auto wav = testio::write_wav_f32(empty, 1, sr);
  auto fac = afp::io::make_default_decoder_factory();
  auto dec = fac->create_decoder();
  auto out = dec->decode_bytes(
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(wav.data()),
                                 wav.size()),
      {});
  // dr_wav returns DecodeError for zero frames in our pipeline
  ASSERT_FALSE(out.has_value());
  ASSERT_EQ(out.error(), afp::util::UtilError::DecodeError);
}

TEST(Edge, EightChannelsDownmix) {
  const uint32_t sr = 32000;
  const size_t N = 64;
  std::vector<std::vector<float>> ch(8, std::vector<float>(N));
  for (size_t c = 0; c < ch.size(); ++c)
    for (size_t i = 0; i < N; ++i) ch[c][i] = float(c + 1) * 0.1f;
  auto inter = testio::interleave(ch);
  auto wav = testio::write_wav_s16(inter, 8, sr);
  auto fac = afp::io::make_default_decoder_factory();
  auto dec = fac->create_decoder();
  auto out = dec->decode_bytes(
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(wav.data()),
                                 wav.size()),
      {
          .normalize_rms = false
      });
  ASSERT_TRUE(out.has_value());
  ASSERT_EQ(out->samples.size(), N);
  // average of 0.1..0.8 = 0.45
  for (float v : out->samples)
    ASSERT_NEAR(v, 0.45f, 0.02f);
}

TEST(Edge, VeryHighSampleRateMetadata) {
  const uint32_t sr = 192000;
  auto mono = testio::make_sine(1000.f, float(sr), 256, 0.2f);
  auto wav = testio::write_wav_f32(testio::interleave({mono}), 1, sr);
  auto fac = afp::io::make_default_decoder_factory();
  auto dec = fac->create_decoder();
  auto out = dec->decode_bytes(
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(wav.data()),
                                 wav.size()),
      {});
  ASSERT_TRUE(out.has_value());
  ASSERT_EQ(out->sample_rate_hz, sr);
}

TEST(Edge, NonCanonicalWavChunks) {
  // Create a WAV then insert a junk chunk between fmt and data; our decoder should either skip or fail cleanly.
  const uint32_t sr = 16000;
  auto mono = testio::make_sine(440.f, float(sr), 128, 0.2f);
  auto clean = testio::write_wav_s16(testio::interleave({mono}), 1, sr);

  // Inject a fake "JUNK" chunk of 8 bytes after fmt (naive, but enough to exercise code-paths)
  // This is simplistic and may not fool dr_wav on all inputs; test accepts either success or clean DecodeError.
  std::vector<uint8_t> withjunk = clean;
  std::array<uint8_t, 12> junkHdr =
      {'J', 'U', 'N', 'K', 8, 0, 0, 0, 1, 2, 3, 4};
  withjunk.insert(withjunk.begin() + 12 + 24 /*roughly after fmt*/,
                  junkHdr.begin(), junkHdr.end());

  auto fac = afp::io::make_default_decoder_factory();
  auto dec = fac->create_decoder();
  auto out = dec->decode_bytes(
      std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(withjunk.data()), withjunk.size()),
      {});
  if (out) {
    EXPECT_GT(out->samples.size(), 0u);
  } else {
    EXPECT_EQ(out.error(), afp::util::UtilError::DecodeError);
  }
}
