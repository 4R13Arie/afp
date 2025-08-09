#include <gtest/gtest.h>
#include "afp/stft/stft.hpp"
#include "test_utils_signal.hpp"

using namespace afp::stft;
using namespace teststft;
using afp::util::FrameBlock;
using afp::util::ComplexSpectra;
using afp::util::UtilError;

static FrameBlock one_frame(const std::vector<float>& x) {
  FrameBlock fb;
  fb.frame_size = static_cast<std::uint32_t>(x.size());
  fb.hop_size   = fb.frame_size;
  fb.num_frames = 1;
  copy_to_univector(x, fb.data);
  return fb;
}

TEST(FFT, Shape_And_Size) {
  auto fac = make_default_stft_factory();
  auto fft = fac->create_fft();

  const std::uint32_t N = 1024;
  FftConfig cfg{N};

  auto fb = one_frame(step(N));
  auto outE = fft->forward_r2c(fb, cfg);
  ASSERT_TRUE(outE.has_value());
  const auto& spec = *outE;

  EXPECT_EQ(spec.num_bins, static_cast<std::uint16_t>(N/2 + 1));
  EXPECT_EQ(spec.num_frames, 1u);
  EXPECT_EQ(spec.bins.size(), std::size_t(spec.num_bins) * spec.num_frames);
}

TEST(FFT, Reject_Mismatch) {
  auto fac = make_default_stft_factory();
  auto fft = fac->create_fft();

  const std::uint32_t N = 1024;
  FftConfig cfg{N};
  auto fb = one_frame(step(512)); // mismatch

  auto outE = fft->forward_r2c(fb, cfg);
  ASSERT_FALSE(outE.has_value());
  EXPECT_EQ(outE.error(), UtilError::InvalidArgument);
}

TEST(FFT, BinDominance_Sine) {
  auto fac = make_default_stft_factory();
  auto fft = fac->create_fft();

  const std::uint32_t sr = 48000;
  const std::uint32_t N  = 1024;
  const std::uint32_t k  = 5;
  const float f = (float(k) * float(sr)) / float(N);

  auto x = sine(N, sr, f);
  auto fb = one_frame(x);

  FftConfig cfg{N};
  auto outE = fft->forward_r2c(fb, cfg);
  ASSERT_TRUE(outE.has_value());

  const auto& spec = *outE;
  ASSERT_EQ(spec.num_frames, 1u);

  // Magnitudes for first frame
  std::vector<float> mag(spec.num_bins, 0.0f);
  for (std::size_t i = 0; i < spec.num_bins; ++i) {
    auto c = spec.bins[i];
    mag[i] = std::sqrt(c.real() * c.real() + c.imag() * c.imag());
  }
  const auto dom = dominant_bin(mag);
  EXPECT_EQ(dom, k);

  // ≥ 20 dB dominance
  float max_other = 0.0f;
  for (std::size_t i = 0; i < mag.size(); ++i)
    if (i != dom) max_other = std::max(max_other, mag[i]);

  const float db = 20.0f * std::log10((mag[dom] + 1e-12f) / (max_other + 1e-12f));
  EXPECT_GE(db, 20.0f);
}

TEST(FFT, DC_Signal) {
  auto fac = make_default_stft_factory();
  auto fft = fac->create_fft();

  const std::uint32_t N = 1024;
  auto x = constant(N, 0.75f);
  auto fb = one_frame(x);

  FftConfig cfg{N};
  auto outE = fft->forward_r2c(fb, cfg);
  ASSERT_TRUE(outE.has_value());
  const auto& spec = *outE;

  std::vector<float> mag(spec.num_bins, 0.0f);
  for (std::size_t i = 0; i < spec.num_bins; ++i) {
    auto c = spec.bins[i];
    mag[i] = std::sqrt(c.real() * c.real() + c.imag() * c.imag());
  }
  const auto dom = dominant_bin(mag);
  EXPECT_EQ(dom, 0u);

  float max_other = 0.0f;
  for (std::size_t i = 1; i < mag.size(); ++i)
    max_other = std::max(max_other, mag[i]);

  const float db = 20.0f * std::log10((mag[0] + 1e-12f) / (max_other + 1e-12f));
  EXPECT_GE(db, 20.0f);
}

TEST(FFT, Impulse_FlatSpectrum_NoNaN) {
  auto fac = make_default_stft_factory();
  auto fft = fac->create_fft();

  const std::uint32_t N = 1024;
  auto x = impulse(N);
  auto fb = one_frame(x);

  FftConfig cfg{N};
  auto outE = fft->forward_r2c(fb, cfg);
  ASSERT_TRUE(outE.has_value());
  const auto& spec = *outE;

  // Magnitude variance should be small-ish and all finite
  double mean_mag = 0.0;
  std::vector<float> mags(spec.num_bins, 0.0f);
  for (std::size_t i = 0; i < spec.num_bins; ++i) {
    auto c = spec.bins[i];
    float m = std::sqrt(c.real()*c.real() + c.imag()*c.imag());
    ASSERT_TRUE(std::isfinite(m));
    mags[i] = m;
    mean_mag += m;
  }
  mean_mag /= double(spec.num_bins);

  double var = 0.0;
  for (float m : mags) var += (m - mean_mag) * (m - mean_mag);
  var /= double(spec.num_bins);

  // Not strictly flat because of implementation details, but should be low variance.
  EXPECT_LT(var, 1e-3);
}

TEST(FFT, Warmup_Idempotent) {
  auto fac = make_default_stft_factory();
  auto fft = fac->create_fft();

  const std::uint32_t N = 1024;
  FftConfig cfg{N};

  fft->warmup(cfg, /*max_frames*/ 4);
  fft->warmup(cfg, /*max_frames*/ 4);

  auto fb = one_frame(step(N));
  auto aE = fft->forward_r2c(fb, cfg);
  auto bE = fft->forward_r2c(fb, cfg);

  ASSERT_TRUE(aE.has_value());
  ASSERT_TRUE(bE.has_value());

  ASSERT_EQ(aE->bins.size(), bE->bins.size());
  for (std::size_t i = 0; i < aE->bins.size(); ++i) {
    ASSERT_NEAR(aE->bins[i].real(), bE->bins[i].real(), 1e-7);
    ASSERT_NEAR(aE->bins[i].imag(), bE->bins[i].imag(), 1e-7);
  }
}
