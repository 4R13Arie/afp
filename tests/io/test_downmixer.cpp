#include <gtest/gtest.h>
#include "afp/io/io.hpp"
#include "afp/util/util.hpp"
#include "test_utils_io.hpp"

using afp::util::PcmSpan;
using afp::util::SampleRateHz;

TEST(Downmixer, MonoPassthrough) {
  auto dm = afp::io::make_default_downmixer();
  const SampleRateHz sr = 16000;
  auto x = testio::make_sine(440.f, float(sr), 256, 0.5f);

  PcmSpan ch{sr, std::span<const float>(x.data(), x.size())};
  std::array<PcmSpan, 1> chans{ch};
  auto outE = dm->to_mono(chans);
  ASSERT_TRUE(outE.has_value());
  const auto& out = *outE;
  ASSERT_EQ(out.sample_rate_hz, sr);
  ASSERT_EQ(out.samples.size(), x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_NEAR(out.samples[i], x[i], 1e-6f);
  }
}

TEST(Downmixer, StereoAverage) {
  auto dm = afp::io::make_default_downmixer();
  const SampleRateHz sr = 44100;
  const size_t N = 128;
  std::vector<float> L(N, 1.0f), R(N, -1.0f);
  PcmSpan chL{sr, std::span<const float>(L.data(), L.size())};
  PcmSpan chR{sr, std::span<const float>(R.data(), R.size())};

  std::array<PcmSpan, 2> chans{chL, chR};
  auto outE = dm->to_mono(chans);
  ASSERT_TRUE(outE.has_value());
  for (float v : outE->samples)
    ASSERT_NEAR(v, 0.0f, 1e-6f);
}

TEST(Downmixer, EnergyPreservingAverageRandomStereo) {
  auto dm = afp::io::make_default_downmixer();
  const SampleRateHz sr = 48000;
  const size_t N = 1000;
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> L(N), R(N);
  auto& gen = testio::rng();
  for (size_t i = 0; i < N; ++i) {
    L[i] = dist(gen);
    R[i] = dist(gen);
  }
  PcmSpan chL{sr, std::span<const float>(L.data(), L.size())};
  PcmSpan chR{sr, std::span<const float>(R.data(), R.size())};
  std::array<PcmSpan, 2> chans{chL, chR};
  auto outE = dm->to_mono(chans);
  ASSERT_TRUE(outE.has_value());
  for (size_t i = 0; i < N; ++i) {
    ASSERT_NEAR(outE->samples[i], 0.5f*(L[i]+R[i]), 1e-6f);
  }
}

TEST(Downmixer, ShapeAndMetadata) {
  auto dm = afp::io::make_default_downmixer();
  const SampleRateHz sr = 96000;
  auto a = testio::make_sine(220.f, float(sr), 200);
  auto b = testio::make_sine(330.f, float(sr), 200);
  PcmSpan A{sr, std::span<const float>(a.data(), a.size())};
  PcmSpan B{sr, std::span<const float>(b.data(), b.size())};
  std::array<PcmSpan, 2> chans{A, B};
  auto outE = dm->to_mono(chans);
  ASSERT_TRUE(outE.has_value());
  ASSERT_EQ(outE->sample_rate_hz, sr);
  ASSERT_EQ(outE->samples.size(), a.size());
}

TEST(Downmixer, Errors) {
  auto dm = afp::io::make_default_downmixer();
  // empty
  std::array<PcmSpan, 0> none{};
  auto e = dm->to_mono(none);
  ASSERT_FALSE(e.has_value());
  ASSERT_EQ(e.error(), afp::util::UtilError::InvalidArgument);

  // mismatched lengths
  const afp::util::SampleRateHz sr = 44100;
  auto a = testio::make_sine(440.f, float(sr), 100);
  auto b = testio::make_sine(440.f, float(sr), 120);
  PcmSpan A{sr, std::span<const float>(a.data(), a.size())};
  PcmSpan B{sr, std::span<const float>(b.data(), b.size())};
  std::array<PcmSpan, 2> chans1{A, B};
  auto e1 = dm->to_mono(chans1);
  ASSERT_FALSE(e1.has_value());
  ASSERT_EQ(e1.error(), afp::util::UtilError::SizeMismatch);

  // mismatched sample rates
  PcmSpan B2{sr + 1, std::span<const float>(b.data(), b.size())};
  std::array<PcmSpan, 2> chans2{A, B2};
  auto e2 = dm->to_mono(chans2);
  ASSERT_FALSE(e2.has_value());
  ASSERT_EQ(e2.error(), afp::util::UtilError::SizeMismatch);
}
