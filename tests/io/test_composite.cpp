#include <gtest/gtest.h>
#include <thread>
#include "afp/io/io.hpp"
#include "afp/util/util.hpp"
#include "test_utils_io.hpp"

using afp::io::DecodeParams;

TEST(Composite, ExtensionRoutingAndFallback) {
  auto fac = afp::io::make_default_decoder_factory();
  auto dec = fac->create_decoder();

  // Valid WAV by bytes but unknown extension path → fallback should still decode
  const uint32_t sr = 16000;
  auto mono = testio::make_sine(440.f, float(sr), 128, 0.2f);
  auto wav = testio::write_wav_s16(testio::interleave({mono}), 1, sr);
  testio::TempFile tf("unknown_ext");
  tf.write(wav.data(), wav.size());
  // rename to ".bin" to bypass extension routing
  std::filesystem::path p = tf.path();
  auto p2 = p.parent_path() / (p.stem().string() + ".bin");
  std::error_code ec;
  std::filesystem::rename(p, p2, ec);

  DecodeParams par{};
  auto out = dec->decode_file(p2.string(), par);
  ASSERT_TRUE(out.has_value()) <<
 "fallback trial decoders should handle valid WAV contents";
  ASSERT_EQ(out->sample_rate_hz, sr);
}

TEST(Composite, SniffByHeaderMemory) {
  auto fac = afp::io::make_default_decoder_factory();
  auto dec = fac->create_decoder();

  const uint32_t sr = 22050;
  auto mono = testio::make_sine(220.f, float(sr), 64, 0.3f);
  auto wav = testio::write_wav_f32(testio::interleave({mono}), 1, sr);
  auto out = dec->decode_bytes(
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(wav.data()),
                                 wav.size()),
      {});
  ASSERT_TRUE(out.has_value());
}

TEST(Composite, UnknownHeaderUnavailable) {
  auto fac = afp::io::make_default_decoder_factory();
  auto dec = fac->create_decoder();
  std::array<uint8_t, 64> junk{};
  auto out = dec->decode_bytes(
      std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(junk.data()), junk.size()), {});
  // Composite tries all decoders; they should return DecodeError, composite maps to DecodeError.
  ASSERT_FALSE(out.has_value());
  ASSERT_EQ(out.error(), afp::util::UtilError::DecodeError);
}

TEST(Composite, ConcurrentInstancesNoDataRaces) {
  auto fac = afp::io::make_default_decoder_factory();
  const uint32_t sr = 44100;
  auto mono = testio::make_sine(880.f, float(sr), 256, 0.25f);
  auto wav = testio::write_wav_s16(testio::interleave({mono}), 1, sr);
  auto bytes = std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(wav.data()), wav.size());

  auto worker = [&](int) {
    auto dec = fac->create_decoder();
    for (int i = 0; i < 10; ++i) {
      auto out = dec->decode_bytes(bytes, {});
      ASSERT_TRUE(out.has_value());
    }
  };

  std::thread t1(worker, 1), t2(worker, 2), t3(worker, 3), t4(worker, 4);
  t1.join();
  t2.join();
  t3.join();
  t4.join();
  // Note: run under TSAN to verify no races (decoders are per-instance).
}
