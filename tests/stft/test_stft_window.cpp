#include <gtest/gtest.h>
#include "afp/stft/stft.hpp"
#include "test_utils_signal.hpp"

using namespace afp::stft;
using namespace teststft;
using afp::util::FrameBlock;
using afp::util::UtilError;

static FrameBlock make_frames_from(const std::vector<float>& x,
                                   std::uint32_t frame, std::uint32_t hop) {
  FrameBlock fb;
  fb.frame_size = frame;
  fb.hop_size = hop;
  fb.num_frames = 1;
  copy_to_univector(x, fb.data);
  return fb;
}

TEST(Window, Hann_OnOnes_EqualsCoeffs) {
  auto fac = make_default_stft_factory();
  auto win = fac->create_window();

  const std::uint32_t N = 1024;
  std::vector<float> ones(N, 1.0f);
  auto fb = make_frames_from(ones, N, N);

  auto outE = win->apply(fb, WindowType::kHann);
  ASSERT_TRUE(outE.has_value());
  const auto& out = *outE;

  ASSERT_EQ(out.data.size(), ones.size());
  // Validate Hann as implemented: 0.5 - 0.5*cos(2π n / N)
  const float invN = 1.0f / float(N);
  for (std::size_t n = 0; n < N; ++n) {
    const float w = 0.5f - 0.5f * std::cos(
                        2.0f * float(M_PI) * (float(n) * invN));
    ASSERT_NEAR(out.data[n], w, 1e-6f);
  }
  EXPECT_NEAR(out.data.front(), 0.0f, 1e-6f);
  EXPECT_NEAR(out.data.back(), 0.0f, 2e-5f);
}

TEST(Window, TwoFrames_Independence) {
  auto fac = make_default_stft_factory();
  auto win = fac->create_window();

  const std::uint32_t N = 8; // tiny to check exact values
  // two frames back-to-back:
  FrameBlock fb;
  fb.frame_size = N;
  fb.hop_size = N;
  fb.num_frames = 2;
  fb.data.resize(N * 2, 1.0f);

  auto outE = win->apply(fb, WindowType::kHann);
  ASSERT_TRUE(outE.has_value());
  const auto& out = *outE;

  ASSERT_EQ(out.data.size(), fb.data.size());
  // both frames should be multiplied by same window coefficients
  const float invN = 1.0f / float(N);
  for (std::size_t i = 0; i < N; ++i) {
    const float w = 0.5f - 0.5f * std::cos(
                        2.0f * float(M_PI) * (float(i) * invN));
    ASSERT_NEAR(out.data[i], w, 1e-6f);
    ASSERT_NEAR(out.data[N + i], w, 1e-6f);
  }
}

TEST(Window, Layout_Preserved) {
  auto fac = make_default_stft_factory();
  auto win = fac->create_window();

  FrameBlock fb;
  fb.frame_size = 256;
  fb.hop_size = 128;
  fb.num_frames = 7;
  fb.data.resize(std::size_t(fb.frame_size) * fb.num_frames, 2.0f);

  auto outE = win->apply(fb, WindowType::kHann);
  ASSERT_TRUE(outE.has_value());
  const auto& out = *outE;

  EXPECT_EQ(out.frame_size, fb.frame_size);
  EXPECT_EQ(out.hop_size, fb.hop_size);
  EXPECT_EQ(out.num_frames, fb.num_frames);
  EXPECT_EQ(out.data.size(), fb.data.size());
}

TEST(Window, InvalidArgs) {
  auto fac = make_default_stft_factory();
  auto win = fac->create_window();

  FrameBlock bad;
  bad.frame_size = 0;
  bad.hop_size = 64;
  bad.num_frames = 1;
  bad.data.resize(1, 1.0f);

  auto outE = win->apply(bad, WindowType::kHann);
  ASSERT_FALSE(outE.has_value());
  EXPECT_EQ(outE.error(), UtilError::InvalidArgument);
}

TEST(Window, UnsupportedWindowType) {
  auto fac = make_default_stft_factory();
  auto win = fac->create_window();

  FrameBlock fb;
  fb.frame_size = 32;
  fb.hop_size = 16;
  fb.num_frames = 1;
  fb.data.resize(32, 1.0f);

  auto outE = win->apply(fb, static_cast<WindowType>(999));
  ASSERT_FALSE(outE.has_value());
  EXPECT_EQ(outE.error(), UtilError::UnsupportedFormat);
}
