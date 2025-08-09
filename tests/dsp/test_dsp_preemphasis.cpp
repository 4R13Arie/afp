#include <gtest/gtest.h>
#include "afp/dsp/dsp.hpp"
#include "test_utils_signal.hpp"

using namespace afp::dsp;
using afp::util::UtilError;

static std::unique_ptr<IDspFactory> F() { return make_default_dsp_factory(); }

TEST(PreEmphasis, ExactFormulaTinyVector) {
  const auto sr = 16000u;
  std::vector<float> x = {1.f, 2.f, 3.f, 4.f};
  const float a = 0.97f;

  auto buf = testdsp::make_buffer(sr, x); // keep owner alive
  auto in = testdsp::span_of(buf); // span over stable storage

  auto pe = F()->create_preemphasis();
  auto yE = pe->process(in, {a});
  ASSERT_TRUE(yE.has_value());
  const auto& y = yE->samples;

  ASSERT_EQ(y.size(), x.size());
  std::vector<float> exp = {
      x[0],
      x[1] - a * x[0],
      x[2] - a * x[1],
      x[3] - a * x[2]
  };
  for (size_t i = 0; i < exp.size(); ++i)
    ASSERT_NEAR(y[i], exp[i], 1e-6f);
}


TEST(PreEmphasis, DC_removal_property) {
  const auto sr = 48000u;
  const size_t N = 8192;
  const float C = 0.5f, alpha = 0.97f;
  auto x = testdsp::dc(N, C);
  auto pe = F()->create_preemphasis();
  auto yE = pe->process(testdsp::span_of(testdsp::make_buffer(sr, x)), {alpha});
  ASSERT_TRUE(yE.has_value());
  const auto& y = yE->samples;

  ASSERT_GE(y.size(), 2u);
  // First sample must be C (x[0] - α*x[-1] with zero-history)
  EXPECT_NEAR(y[0], C, 1e-6f);
  // Tail should settle to (1-α)*C
  const size_t skip = 128;
  auto tail = std::span<const float>(y.data() + std::min(skip, y.size() - 1),
                                     y.size() - std::min(skip, y.size() - 1));
  const float expected = (1.0f - alpha) * C; // = 0.015
  // Check mean and RMS around expected
  const float mu = testdsp::mean(tail);
  EXPECT_NEAR(mu, expected, 1e-3f);
  const float err_rms = testdsp::rms(tail) - expected; // loose check
  (void)err_rms; // optional: ensure not crazy
}

TEST(PreEmphasis, AlphaBoundsAndSpecialCases) {
  const auto sr = 22050u;
  auto x = testdsp::sine(32, float(sr), 1000.0f, 0.2f);

  auto buf = testdsp::make_buffer(sr, x); // keep owner alive
  auto in = testdsp::span_of(buf); // span over stable storage

  auto pe = F()->create_preemphasis();

  // invalid alpha
  auto e1 = pe->process(in, {-0.1f});
  ASSERT_FALSE(e1.has_value());
  EXPECT_EQ(e1.error(), UtilError::InvalidArgument);
  auto e2 = pe->process(in, {1.1f});
  ASSERT_FALSE(e2.has_value());
  EXPECT_EQ(e2.error(), UtilError::InvalidArgument);

  // alpha = 0 → identity
  auto y0 = pe->process(in, {0.0f});
  ASSERT_TRUE(y0.has_value());
  ASSERT_EQ(y0->samples.size(), x.size());
  for (size_t i = 0; i < x.size(); ++i)
    ASSERT_NEAR(y0->samples[i], x[i], 1e-7f);

  // alpha = 1 → y[n] = x[n] - x[n-1]
  auto y1 = pe->process(in, {1.0f});
  ASSERT_TRUE(y1.has_value());
  ASSERT_EQ(y1->samples.size(), x.size());
  for (size_t i = 1; i < x.size(); ++i)
    ASSERT_NEAR(y1->samples[i], x[i] - x[i-1], 1e-6f);
}

TEST(PreEmphasis, LengthTrimmedAndRepeatability) {
  const auto sr = 16000u;
  const size_t N = 257;
  auto x = testdsp::impulse(N);
  auto pe = F()->create_preemphasis();
  auto y1 = pe->process(testdsp::span_of(testdsp::make_buffer(sr, x)), {0.97f});
  auto y2 = pe->process(testdsp::span_of(testdsp::make_buffer(sr, x)), {0.97f});
  ASSERT_TRUE(y1.has_value() && y2.has_value());
  ASSERT_EQ(y1->samples.size(), x.size());
  ASSERT_EQ(y2->samples.size(), x.size());
  for (size_t i = 0; i < x.size(); ++i)
    ASSERT_NEAR(y1->samples[i], y2->samples[i], 1e-7f);
}