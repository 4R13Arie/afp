#include <gtest/gtest.h>
#include "afp/stft/stft.hpp"
#include "test_utils_signal.hpp"

using namespace afp::stft;
using namespace teststft;

TEST(Factory, CreatesAll) {
  auto fac = make_default_stft_factory();
  ASSERT_TRUE(fac != nullptr);

  auto fr = fac->create_framer();
  auto win = fac->create_window();
  auto fft = fac->create_fft();
  auto drv = fac->create_driver();

  EXPECT_TRUE(fr != nullptr);
  EXPECT_TRUE(win != nullptr);
  EXPECT_TRUE(fft != nullptr);
  EXPECT_TRUE(drv != nullptr);
}

TEST(Factory, FFT_Instances_Independent) {
  auto fac = make_default_stft_factory();
  auto fft1 = fac->create_fft();
  auto fft2 = fac->create_fft();

  const std::uint32_t N = 1024;
  auto fb = afp::util::FrameBlock{};
  fb.frame_size = N;
  fb.hop_size = N;
  fb.num_frames = 1u;
  auto step = teststft::step(N);
  copy_to_univector(step, fb.data);
  FftConfig cfg{N};

  // warm them differently; results must still match for same input
  fft1->warmup(cfg, 1);
  fft2->warmup(cfg, 8);

  auto aE = fft1->forward_r2c(fb, cfg);
  auto bE = fft2->forward_r2c(fb, cfg);
  ASSERT_TRUE(aE.has_value());
  ASSERT_TRUE(bE.has_value());

  ASSERT_EQ(aE->bins.size(), bE->bins.size());
  for (std::size_t i = 0; i < aE->bins.size(); ++i) {
    EXPECT_NEAR(aE->bins[i].real(), bE->bins[i].real(), 1e-7);
    EXPECT_NEAR(aE->bins[i].imag(), bE->bins[i].imag(), 1e-7);
  }
}
