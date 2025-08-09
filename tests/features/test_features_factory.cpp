#include <gtest/gtest.h>
#include "afp/features/features.hpp"
#include "test_utils_signal.hpp"

using afp::features::IFeaturesFactory;
using afp::features::ISpectrogramBuilder;
using afp::features::make_default_features_factory;

TEST(Factory, Creates_Builder_And_Build_Works) {
  auto fac = make_default_features_factory();
  ASSERT_TRUE(!!fac);

  auto builder = fac->create_spectrogram_builder();
  ASSERT_TRUE(!!builder);

  // Minimal valid inputs
  const uint32_t sr = 16000, fft = 256, hop = 80;
  const uint32_t bins = fft / 2 + 1;
  auto spec = testfeat::make_spectra(/*frames*/1, /*bins*/bins);

  auto params = testfeat::make_params(sr, hop, fft, 0.f, sr / 2.f, 1e-8f, 60.f,
                                      false, 0);

  auto outE = builder->build(spec, params);
  ASSERT_TRUE(outE.has_value());
  EXPECT_EQ(outE->num_frames, 1u);
  EXPECT_EQ(outE->num_bins, bins);
  EXPECT_EQ(outE->sample_rate_hz, sr);
  EXPECT_EQ(outE->hop_size, hop);
  EXPECT_EQ(outE->fft_size, fft);
  ASSERT_EQ(outE->log_mag.size(), static_cast<size_t>(bins));
}
