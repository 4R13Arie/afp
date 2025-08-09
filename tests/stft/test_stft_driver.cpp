#include <gtest/gtest.h>
#include "afp/stft/stft.hpp"
#include "test_utils_signal.hpp"

using namespace afp::stft;
using afp::util::UtilError;

static std::uint32_t expect_num_frames(std::size_t N, const FramerConfig& cfg) {
  if (N == 0) return 0;
  if (N < cfg.frame_size) return cfg.pad_end ? 1u : 0u;
  const std::size_t usable = N - cfg.frame_size;
  const std::size_t steps  = usable / cfg.hop_size;
  const std::size_t exact  = steps + 1;
  if (!cfg.pad_end) return static_cast<std::uint32_t>(exact);
  const bool needs_tail = (usable % cfg.hop_size) != 0;
  return static_cast<std::uint32_t>(exact + (needs_tail ? 1 : 0));
}

TEST(Driver, EndToEnd_Sine_BinStable) {
  auto fac = make_default_stft_factory();
  auto drv = fac->create_driver();

  const std::uint32_t sr = 48000;
  const std::uint32_t N  = 4096;
  const std::uint32_t frame = 1024;
  const std::uint32_t hop   = 256;
  const std::uint32_t k     = 5;
  const float f = (float(k) * float(sr)) / float(frame); // bin-aligned per frame size

  auto x = teststft::sine(N, sr, f);
  auto buf = teststft::make_buffer(sr, x);

  FramerConfig fr{frame, hop, true};
  FftConfig    fft{frame};

  auto specE = drv->run(teststft::span_of(buf), fr, WindowType::kHann, fft);
  ASSERT_TRUE(specE.has_value());
  const auto& spec = *specE;

  ASSERT_GT(spec.num_frames, 0u);
  ASSERT_EQ(spec.num_bins, frame/2 + 1);

  // Check dominant bin ≈ k for each frame (allow tiny windowing spread but keep dominance)
  for (std::uint32_t fidx = 0; fidx < spec.num_frames; ++fidx) {
    const std::size_t off = std::size_t(fidx) * spec.num_bins;
    float dom_val = 0.0f; std::size_t dom_idx = 0;
    float max_other = 0.0f;
    for (std::size_t i = 0; i < spec.num_bins; ++i) {
      auto c = spec.bins[off + i];
      float m = std::sqrt(c.real()*c.real() + c.imag()*c.imag());
      if (m > dom_val) { dom_val = m; dom_idx = i; }
    }
    for (std::size_t i = 0; i < spec.num_bins; ++i) {
      if (i == dom_idx) continue;
      auto c = spec.bins[off + i];
      float m = std::sqrt(c.real()*c.real() + c.imag()*c.imag());
      if (m > max_other) max_other = m;
    }
    // Dominant bin should equal k exactly in most frames; if not, keep 20 dB dominance.
    EXPECT_TRUE(dom_idx == k || (20.0f * std::log10((dom_val+1e-12f)/(max_other+1e-12f)) >= 20.0f));
  }
}

TEST(Driver, AllZeros_ToAllZeroSpectra) {
  auto fac = make_default_stft_factory();
  auto drv = fac->create_driver();

  const std::uint32_t sr = 48000;
  const std::uint32_t frame = 1024, hop = 256;
  auto buf = teststft::make_buffer(sr, std::vector<float>(4096, 0.0f));

  auto specE = drv->run(teststft::span_of(buf),
                        FramerConfig{frame, hop, true},
                        WindowType::kHann,
                        FftConfig{frame});
  ASSERT_TRUE(specE.has_value());
  const auto& spec = *specE;

  for (const auto& c : spec.bins) {
    ASSERT_NEAR(c.real(), 0.0f, 1e-7);
    ASSERT_NEAR(c.imag(), 0.0f, 1e-7);
  }
}

TEST(Driver, Mismatch_FrameVsFFT) {
  auto fac = make_default_stft_factory();
  auto drv = fac->create_driver();

  auto buf = teststft::make_buffer(48000, teststft::step(1024));
  auto specE = drv->run(teststft::span_of(buf),
                        FramerConfig{512, 256, true},  // frame != fft
                        WindowType::kHann,
                        FftConfig{1024});
  ASSERT_FALSE(specE.has_value());
  EXPECT_EQ(specE.error(), UtilError::InvalidArgument);
}

TEST(Driver, ZeroSampleRate) {
  auto fac = make_default_stft_factory();
  auto drv = fac->create_driver();

  auto buf = teststft::make_buffer(0, teststft::step(1024));
  auto specE = drv->run(teststft::span_of(buf),
                        FramerConfig{1024, 256, true},
                        WindowType::kHann,
                        FftConfig{1024});
  ASSERT_FALSE(specE.has_value());
  EXPECT_EQ(specE.error(), UtilError::InvalidArgument);
}

TEST(Driver, Short_WithPad_YieldsOneFFTFrame) {
  auto fac = make_default_stft_factory();
  auto drv = fac->create_driver();

  const std::uint32_t sr = 48000;
  const std::uint32_t frame = 1024, hop = 256;
  auto buf = teststft::make_buffer(sr, teststft::step(800)); // shorter than frame

  auto specE = drv->run(teststft::span_of(buf),
                        FramerConfig{frame, hop, true},
                        WindowType::kHann,
                        FftConfig{frame});
  ASSERT_TRUE(specE.has_value());
  EXPECT_EQ(specE->num_frames, 1u);
  EXPECT_EQ(specE->num_bins, frame/2 + 1);
}

TEST(Driver, Hop_Affects_FrameCount) {
  auto fac = make_default_stft_factory();
  auto drv = fac->create_driver();

  const std::uint32_t sr = 48000;
  const std::size_t   N  = 100000; // arbitrary
  const std::uint32_t frame = 1024;

  auto buf = teststft::make_buffer(sr, teststft::step(N));

  FramerConfig a{frame, frame, true};
  FramerConfig b{frame, frame/4, true};
  FftConfig    fft{frame};

  auto A = drv->run(teststft::span_of(buf), a, WindowType::kHann, fft);
  auto B = drv->run(teststft::span_of(buf), b, WindowType::kHann, fft);

  ASSERT_TRUE(A.has_value());
  ASSERT_TRUE(B.has_value());

  const auto nfA = expect_num_frames(N, a);
  const auto nfB = expect_num_frames(N, b);
  EXPECT_EQ(A->num_frames, nfA);
  EXPECT_EQ(B->num_frames, nfB);

  // With same input, smaller hop should yield approximately 4x frames
  // (exactness verified via the formula above).
  EXPECT_GT(nfB, nfA);
  EXPECT_NEAR(double(nfB) / double(nfA), 4.0, 0.2); // allow a slight boundary effect
}
