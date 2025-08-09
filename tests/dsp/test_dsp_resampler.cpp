#include <gtest/gtest.h>
#include "afp/dsp/dsp.hpp"
#include "test_utils_signal.hpp"

using namespace afp::dsp;
using afp::util::UtilError;

static std::unique_ptr<IDspFactory> F() { return make_default_dsp_factory(); }

TEST(Resampler, RateChangeAndLength) {
  const auto sr_in = 48000u;
  const auto sr_out = 16000u;
  const size_t N = 48000; // 1s
  auto x = testdsp::sine(N, float(sr_in), 1000.0f, 0.7f);

  auto r = F()->create_resampler();
  auto yE = r->resample(testdsp::span_of(testdsp::make_buffer(sr_in, x)),
                        {sr_out, 8});
  ASSERT_TRUE(yE.has_value());
  EXPECT_EQ(yE->sample_rate_hz, sr_out);

  const size_t exp_len = size_t(
      std::llround(double(N) * double(sr_out) / double(sr_in)));
  const auto got = yE->samples.size();
  ASSERT_LE(std::llabs((long long)got - (long long)exp_len), 1ll);
}

TEST(Resampler, ToneFrequencyPreservation) {
  const auto sr_in = 48000u;
  const auto sr_out = 16000u;
  const size_t N = 48000; // analyze reasonable length
  const float f = 1000.0f;
  auto x = testdsp::sine(N, float(sr_in), f, 0.9f);

  auto r = F()->create_resampler();
  auto yE = r->resample(testdsp::span_of(testdsp::make_buffer(sr_in, x)),
                        {sr_out, 8});
  ASSERT_TRUE(yE.has_value());

  // measure dominant frequency
  std::vector<float> yvec(yE->samples.begin(), yE->samples.end());
  const float f_dom = testdsp::dominant_bin_freq(
      std::span<const float>(yvec.data(), yvec.size()), float(sr_out));

  // Bin resolution ~ sr_out / N_out; allow ±30 Hz slack
  EXPECT_NEAR(f_dom, f, 30.0f);
}

TEST(Resampler, QualityMonotoneAliasRejectionDownsample) {
  // Downsample 48k -> 16k; measure high-frequency energy above 0.45*Nyq(out)
  const auto sr_in = 48000u;
  const auto sr_out = 16000u;
  const size_t N = 48000;
  // broadband-ish: sum of two high tones + some low tone
  auto x = testdsp::sine(N, float(sr_in), 500.0f, 0.5f);
  auto x2 = testdsp::sine(N, float(sr_in), 7000.0f, 0.3f);
  auto x3 = testdsp::sine(N, float(sr_in), 9000.0f, 0.3f);
  for (size_t i = 0; i < N; ++i) x[i] = x[i] + x2[i] + x3[i];

  auto r = F()->create_resampler();
  auto do_one = [&](int quality)-> std::vector<float> {
    auto y = r->resample(testdsp::span_of(testdsp::make_buffer(sr_in, x)),
                         {sr_out, quality});
    EXPECT_TRUE(y.has_value());
    return std::vector<float>(y->samples.begin(), y->samples.end());
  };

  auto yq0 = do_one(0);
  auto yq4 = do_one(4);
  auto yq8 = do_one(8);

  auto sp0 = testdsp::mag_spectrum(
      std::span<const float>(yq0.data(), yq0.size()));
  auto sp4 = testdsp::mag_spectrum(
      std::span<const float>(yq4.data(), yq4.size()));
  auto sp8 = testdsp::mag_spectrum(
      std::span<const float>(yq8.data(), yq8.size()));

  const float nyq = 0.5f * float(sr_out);
  const float cutoff = 0.45f * nyq; // near top end
  auto sum_above = [&](const std::vector<float>& sp)-> double {
    const size_t N2 = sp.size();
    double s = 0.0;
    for (size_t k = 0; k < N2; ++k) {
      const float f = float(k) * float(sr_out) / float(2 * (N2 - 1));
      // approx for r2c bins
      if (f >= cutoff) s += sp[k];
    }
    return s;
  };

  const double a0 = sum_above(sp0);
  const double a4 = sum_above(sp4);
  const double a8 = sum_above(sp8);

  // Expect monotone improvement (allow some slack)
  EXPECT_GT(a0, a4 * 0.9); // ~10% or more reduction
  EXPECT_GT(a4, a8 * 0.9);
}

TEST(Resampler, InvalidArgsAndRepeatability) {
  auto r = F()->create_resampler();
  auto in = testdsp::span_of(testdsp::make_buffer(0, {})); // sr=0

  auto e1 = r->resample(in, {16000u, 8});
  ASSERT_FALSE(e1.has_value());
  EXPECT_EQ(e1.error(), UtilError::InvalidArgument);
  auto e2 = r->resample({44100u, std::span<const float>()}, {0u, 8});
  ASSERT_FALSE(e2.has_value());
  EXPECT_EQ(e2.error(), UtilError::InvalidArgument);

  // repeatability
  const auto sr_in = 44100u;
  auto x = testdsp::sine(4096, float(sr_in), 1200.0f, 0.5f);
  auto y1 = r->resample(testdsp::span_of(testdsp::make_buffer(sr_in, x)),
                        {48000u, 4});
  auto y2 = r->resample(testdsp::span_of(testdsp::make_buffer(sr_in, x)),
                        {48000u, 4});
  ASSERT_TRUE(y1.has_value() && y2.has_value());
  ASSERT_EQ(y1->samples.size(), y2->samples.size());
  for (size_t i = 0; i < y1->samples.size(); ++i)
    ASSERT_NEAR(y1->samples[i], y2->samples[i], 1e-7f);
}