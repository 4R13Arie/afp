#include <gtest/gtest.h>
#include "afp/dsp/dsp.hpp"
#include "test_utils_signal.hpp"

using namespace afp::dsp;
using afp::util::UtilError;

static std::unique_ptr<IDspFactory> F() { return make_default_dsp_factory(); }

TEST(HighPass, HPF_suppresses_DC) {
  const auto sr = 48000u;
  const size_t N = 8192;
  auto x = testdsp::dc(N, 0.3f);
  auto in = testdsp::make_buffer(sr, x);

  auto hpf = F()->create_hpf();
  BiquadParams p{200.0f, 0.707f};
  auto yE = hpf->process(testdsp::span_of(in), p);
  ASSERT_TRUE(yE.has_value());
  const auto& y = yE->samples;

  // Mean should be ~0
  EXPECT_NEAR(testdsp::mean({y.data(), y.size()}), 0.0f, 1e-3f);

  // Ignore startup transient — check RMS on tail only
  const size_t skip = 1024; // ~21 ms at 48 kHz; safe settling
  ASSERT_GE(y.size(), skip + 1);
  auto tail = std::span<const float>(y.data() + skip, y.size() - skip);
  EXPECT_NEAR(testdsp::rms(tail), 0.0f, 5e-3f);

  // Also ensure strong attenuation vs input (DC rms = 0.3)
  const float in_rms = 0.3f;
  const float tail_rms = testdsp::rms(tail);
  const float attn_db = 20.0f * std::log10(
                            (in_rms + 1e-12f) / (tail_rms + 1e-12f));
  EXPECT_GE(attn_db, 30.0f); // ≥30 dB DC rejection on the tail
}

TEST(HighPass, HPF_attenuates_low_more_than_high) {
  const auto sr = 48000u;
  const size_t N = 8192;
  auto low = testdsp::sine(N, float(sr), 50.0f, 0.9f);
  auto high = testdsp::sine(N, float(sr), 2000.0f, 0.9f);

  auto hpf = F()->create_hpf();
  BiquadParams p{200.0f, 0.707f};

  auto ylE = hpf->process(testdsp::span_of(testdsp::make_buffer(sr, low)), p);
  auto yhE = hpf->process(testdsp::span_of(testdsp::make_buffer(sr, high)), p);
  ASSERT_TRUE(ylE.has_value());
  ASSERT_TRUE(yhE.has_value());

  const float rl = testdsp::rms({ylE->samples.data(), ylE->samples.size()});
  const float rh = testdsp::rms({yhE->samples.data(), yhE->samples.size()});
  const float diff_db = 20.0f * std::log10((rh + 1e-12f) / (rl + 1e-12f));
  EXPECT_GE(diff_db, 15.0f);
}

TEST(HighPass, HPF_impulse_response_stable) {
  const auto sr = 44100u;
  const size_t N = 4096;
  auto x = testdsp::impulse(N);
  auto hpf = F()->create_hpf();
  BiquadParams p{300.0f, 0.707f};
  auto yE = hpf->process(testdsp::span_of(testdsp::make_buffer(sr, x)), p);
  ASSERT_TRUE(yE.has_value());
  const auto& y = yE->samples;
  for (float v : y) { ASSERT_TRUE(std::isfinite(v)); }
  const float max_0_10 = testdsp::max_abs({y.data(),
                                           std::min<size_t>(10, y.size())});
  const float max_256_N = testdsp::max_abs(
  {y.data() + std::min<size_t>(256, y.size()),
   y.size() - std::min<size_t>(256, y.size())});
  EXPECT_LE(max_256_N, max_0_10 + 1e-6f); // decay or at least not explode
}

TEST(HighPass, HPF_invalid_args) {
  auto hpf = F()->create_hpf();
  const auto sr = 48000u;
  auto x = testdsp::sine(1024, float(sr), 1000.0f);
  auto in = testdsp::span_of(testdsp::make_buffer(sr, x));

  // cutoff <= 0
  auto e1 = hpf->process(in, {0.0f, 0.707f});
  ASSERT_FALSE(e1.has_value());
  EXPECT_EQ(e1.error(), UtilError::InvalidArgument);
  // cutoff >= Nyquist
  auto e2 = hpf->process(in, {float(sr / 2), 0.707f});
  ASSERT_FALSE(e2.has_value());
  EXPECT_EQ(e2.error(), UtilError::InvalidArgument);
  // Q <= 0
  auto e3 = hpf->process(in, {200.0f, 0.0f});
  ASSERT_FALSE(e3.has_value());
  EXPECT_EQ(e3.error(), UtilError::InvalidArgument);
}

TEST(HighPass, HPF_cutoff_clamping_behavior) {
  const auto sr = 48000u;
  const size_t N = 4096;
  auto x = testdsp::sine(N, float(sr), 1000.0f, 0.8f);
  auto in = testdsp::span_of(testdsp::make_buffer(sr, x));
  auto hpf = F()->create_hpf();

  // reference near‑Nyquist
  const float nyq = 0.5f * float(sr);
  auto yA = hpf->process(in, {nyq * 0.99f, 0.707f});
  ASSERT_TRUE(yA.has_value());

  // Above Nyquist should be rejected per current preconditions
  auto yB = hpf->process(in, {nyq * 1.05f, 0.707f});
  ASSERT_FALSE(yB.has_value());
  EXPECT_EQ(yB.error(), afp::util::UtilError::InvalidArgument);

  // Optional: compare two close valid cutoffs to see tiny diff
  auto yC = hpf->process(in, {nyq * 0.989f, 0.707f});
  ASSERT_TRUE(yC.has_value());
  ASSERT_EQ(yA->samples.size(), yC->samples.size());
  float diff = 0.0f;
  for (size_t i = 0; i < yA->samples.size(); ++i)
    diff = std::max(diff, std::abs(yA->samples[i] - yC->samples[i]));
  EXPECT_LT(diff, 1e-3f);
}

TEST(HighPass, HPF_stateless_repeatability) {
  const auto sr = 32000u;
  const size_t N = 2048;
  auto x = testdsp::sine(N, float(sr), 800.0f, 0.7f);
  auto in = testdsp::span_of(testdsp::make_buffer(sr, x));
  auto hpf = F()->create_hpf();
  auto y1 = hpf->process(in, {300.0f, 0.707f});
  auto y2 = hpf->process(in, {300.0f, 0.707f});
  ASSERT_TRUE(y1.has_value() && y2.has_value());
  ASSERT_EQ(y1->samples.size(), y2->samples.size());
  for (size_t i = 0; i < y1->samples.size(); ++i)
    ASSERT_NEAR(y1->samples[i], y2->samples[i], 1e-7f);
}