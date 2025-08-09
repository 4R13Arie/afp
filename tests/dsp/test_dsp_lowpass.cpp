#include <gtest/gtest.h>
#include "afp/dsp/dsp.hpp"
#include "test_utils_signal.hpp"

using namespace afp::dsp;
using afp::util::UtilError;

static std::unique_ptr<IDspFactory> F() { return make_default_dsp_factory(); }

TEST(LowPass, LPF_passes_low_blocks_high) {
  const auto sr = 48000u;
  const size_t N = 8192;
  auto low = testdsp::sine(N, float(sr), 200.0f, 0.9f);
  auto high = testdsp::sine(N, float(sr), 6000.0f, 0.9f);

  auto lpf = F()->create_lpf();
  BiquadParams p{1000.0f, 0.707f};

  auto yl = lpf->process(testdsp::span_of(testdsp::make_buffer(sr, low)), p);
  auto yh = lpf->process(testdsp::span_of(testdsp::make_buffer(sr, high)), p);
  ASSERT_TRUE(yl.has_value() && yh.has_value());

  const float rl = testdsp::rms({yl->samples.data(), yl->samples.size()});
  const float rh = testdsp::rms({yh->samples.data(), yh->samples.size()});
  const float diff_db = 20.0f * std::log10((rl + 1e-12f) / (rh + 1e-12f));
  EXPECT_GE(diff_db, 20.0f);
}

TEST(LowPass, LPF_DC_preserved) {
  const auto sr = 44100u;
  const size_t N = 4096;
  auto x = testdsp::dc(N, 0.25f);
  auto lpf = F()->create_lpf();
  auto y = lpf->process(testdsp::span_of(testdsp::make_buffer(sr, x)),
                        {1000.0f, 0.707f});
  ASSERT_TRUE(y.has_value());
  EXPECT_NEAR(testdsp::mean({y->samples.data(), y->samples.size()}), 0.25f,
              1e-3f);
}

TEST(LowPass, LPF_step_response_monotone_rise_loose) {
  const auto sr = 48000u;
  const size_t N = 4096;
  auto x = testdsp::step(N, 1.0f);
  auto lpf = F()->create_lpf();
  auto y = lpf->process(testdsp::span_of(testdsp::make_buffer(sr, x)),
                        {1000.0f, 0.707f});
  ASSERT_TRUE(y.has_value());
  // No large overshoot: max should be <= 1.05x final mean
  const float m = testdsp::mean({y->samples.data() + N / 2,
                                 y->samples.size() - N / 2});
  const float mx = testdsp::max_abs({y->samples.data(), y->samples.size()});
  EXPECT_LE(mx, 1.05f * std::max(1e-6f, m));
}

TEST(LowPass, LPF_invalid_args) {
  auto lpf = F()->create_lpf();
  const auto sr = 48000u;
  auto x = testdsp::sine(1024, float(sr), 1000.0f);
  auto in = testdsp::span_of(testdsp::make_buffer(sr, x));

  auto e1 = lpf->process(in, {0.0f, 0.707f});
  ASSERT_FALSE(e1.has_value());
  EXPECT_EQ(e1.error(), UtilError::InvalidArgument);
  auto e2 = lpf->process(in, {float(sr / 2), 0.707f});
  ASSERT_FALSE(e2.has_value());
  EXPECT_EQ(e2.error(), UtilError::InvalidArgument);
  auto e3 = lpf->process(in, {1000.0f, 0.0f});
  ASSERT_FALSE(e3.has_value());
  EXPECT_EQ(e3.error(), UtilError::InvalidArgument);
}

TEST(LowPass, LPF_cutoff_clamping_behavior) {
  const auto sr = 48000u;
  const size_t N = 4096;
  auto x = testdsp::sine(N, float(sr), 200.0f, 0.8f);
  auto in = testdsp::span_of(testdsp::make_buffer(sr, x));
  auto lpf = F()->create_lpf();

  const float nyq = 0.5f * float(sr);
  auto yA = lpf->process(in, {nyq * 0.999f, 0.707f});
  ASSERT_TRUE(yA.has_value());
  auto yB = lpf->process(in, {nyq * 1.05f, 0.707f});
  ASSERT_TRUE(yB.has_value());

  float diff = 0.0f;
  for (size_t i = 0; i < yA->samples.size(); ++i) diff = std::max(
                                                      diff, std::abs(
                                                          yA->samples[i] - yB->
                                                          samples[i]));
  EXPECT_LT(diff, 1e-3f);
}

TEST(LowPass, LPF_stateless_repeatability) {
  const auto sr = 44100u;
  const size_t N = 2048;
  auto x = testdsp::sine(N, float(sr), 400.0f, 0.7f);
  auto in = testdsp::span_of(testdsp::make_buffer(sr, x));
  auto lpf = F()->create_lpf();
  auto y1 = lpf->process(in, {1000.0f, 0.707f});
  auto y2 = lpf->process(in, {1000.0f, 0.707f});
  ASSERT_TRUE(y1.has_value() && y2.has_value());
  for (size_t i = 0; i < y1->samples.size(); ++i)
    ASSERT_NEAR(y1->samples[i], y2->samples[i], 1e-7f);
}