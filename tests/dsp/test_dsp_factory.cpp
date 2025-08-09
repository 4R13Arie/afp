#include <gtest/gtest.h>
#include "afp/dsp/dsp.hpp"
#include "test_utils_signal.hpp"

using namespace afp::dsp;

TEST(Factory, CreatesObjects) {
  auto f = make_default_dsp_factory();
  ASSERT_TRUE(f->create_hpf() != nullptr);
  ASSERT_TRUE(f->create_lpf() != nullptr);
  ASSERT_TRUE(f->create_preemphasis() != nullptr);
  ASSERT_TRUE(f->create_resampler() != nullptr);
}

TEST(Factory, ObjectsAreIndependent) {
  auto f = make_default_dsp_factory();
  auto h1 = f->create_hpf();
  auto h2 = f->create_hpf();

  const auto sr = 44100u;
  auto x = testdsp::sine(1024, float(sr), 1000.0f, 0.5f);
  auto in = testdsp::span_of(testdsp::make_buffer(sr, x));

  auto y1 = h1->process(in, {300.0f, 0.707f});
  auto y2 = h2->process(in, {300.0f, 0.707f});
  ASSERT_TRUE(y1.has_value() && y2.has_value());
  for (size_t i = 0; i < y1->samples.size(); ++i)
    ASSERT_NEAR(y1->samples[i], y2->samples[i], 1e-7f);
}