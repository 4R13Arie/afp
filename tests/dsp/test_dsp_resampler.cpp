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

  auto buf = testdsp::make_buffer(sr_in, x); // keep owner alive
  auto in = testdsp::span_of(buf);

  auto r = F()->create_resampler();
  auto yE = r->resample(in, {sr_out, 8});
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
  const size_t N = 48000;
  const float f = 1000.0f;
  auto x = testdsp::sine(N, float(sr_in), f, 0.9f);

  auto buf = testdsp::make_buffer(sr_in, x); // keep owner alive
  auto in = testdsp::span_of(buf);

  auto r = F()->create_resampler();
  auto yE = r->resample(in, {sr_out, 8});
  ASSERT_TRUE(yE.has_value());

  std::vector<float> yvec(yE->samples.begin(), yE->samples.end());
  const float f_dom = testdsp::dominant_bin_freq(
      std::span<const float>(yvec.data(), yvec.size()), float(sr_out));

  // Bin resolution ~ sr_out / N_out; allow ±30 Hz slack
  EXPECT_NEAR(f_dom, f, 30.0f);
}

TEST(Resampler, QualityMonotoneAliasRejectionDownsample) {
  // 48k -> 16k; verify higher quality rejects OUT-OF-BAND energy better.
  const auto sr_in = 48000u;
  const auto sr_out = 16000u;
  const size_t N = 48000;

  // One in-band tone (should pass), two OUT-of-band tones (> 8kHz) that must be suppressed.
  auto x = testdsp::sine(N, float(sr_in), 500.0f, 0.5f); // in-band
  auto x12 = testdsp::sine(N, float(sr_in), 12000.0f, 0.3f); // out-of-band
  auto x14 = testdsp::sine(N, float(sr_in), 14000.0f, 0.3f); // out-of-band
  for (size_t i = 0; i < N; ++i) x[i] = x[i] + x12[i] + x14[i];

  auto buf = testdsp::make_buffer(sr_in, x); // keep owner alive
  auto in = testdsp::span_of(buf);

  auto r = F()->create_resampler();

  // KFR valid levels in-range are 4 (draft), 6 (low), 8 (normal).
  auto run = [&](int q) -> std::vector<float> {
    auto yE = r->resample(in, {sr_out, q});
    EXPECT_TRUE(yE.has_value());
    return std::vector<float>(yE->samples.begin(), yE->samples.end());
  };

  auto y4 = run(4); // draft
  auto y6 = run(6); // low
  auto y8 = run(8); // normal

  auto sp4 =
      testdsp::mag_spectrum(std::span<const float>(y4.data(), y4.size()));
  auto sp6 =
      testdsp::mag_spectrum(std::span<const float>(y6.data(), y6.size()));
  auto sp8 =
      testdsp::mag_spectrum(std::span<const float>(y8.data(), y8.size()));

  // Sum energy near the top of the band: mostly stopband leakage if filters are doing their job.
  const float nyq_out = 0.5f * float(sr_out);
  const float cutoff = 0.80f * nyq_out; // focus on the stopband region
  auto sum_above = [&](const std::vector<float>& sp) -> double {
    const size_t N2 = sp.size();
    double s = 0.0;
    for (size_t k = 0; k < N2; ++k) {
      const float f = float(k) * float(sr_out) / float(2 * (N2 - 1));
      // r2c approx
      if (f >= cutoff) s += sp[k];
    }
    return s;
  };

  const double a4 = sum_above(sp4);
  const double a6 = sum_above(sp6);
  const double a8 = sum_above(sp8);

  // Higher quality => better stopband rejection => less high-end energy.
  // Allow a little slack for window/FFT variance.
  EXPECT_GT(a4, a6 * 0.95);
  EXPECT_GT(a6, a8 * 0.95);
}
