#include <gtest/gtest.h>
#include "afp/stft/stft.hpp"
#include "test_utils_signal.hpp"

using namespace afp::stft;
using afp::util::UtilError;

// Local mirror of compute_num_frames for exactness checks in driver tests.
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

TEST(Framer, ExactFit_NoPad) {
  auto fac = make_default_stft_factory();
  auto fr  = fac->create_framer();

  const std::uint32_t sr = 48000;
  const std::size_t   N  = 4096;
  const FramerConfig cfg{1024, 256, false};

  auto buf = teststft::make_buffer(sr, teststft::constant(N, 1.0f));
  auto outE = fr->segment(teststft::span_of(buf), cfg);

  ASSERT_TRUE(outE.has_value());
  const auto& fb = *outE;
  EXPECT_EQ(fb.frame_size, 1024u);
  EXPECT_EQ(fb.hop_size,   256u);

  const std::uint32_t expected_frames = 1u + (4096u - 1024u) / 256u; // 13
  EXPECT_EQ(fb.num_frames, expected_frames);
  EXPECT_EQ(fb.data.size(), std::size_t(expected_frames) * 1024u);
}

TEST(Framer, TailPad_Enabled) {
  auto fac = make_default_stft_factory();
  auto fr  = fac->create_framer();

  const std::uint32_t sr = 48000;
  const std::size_t   N  = 3000; // leaves a remainder
  const FramerConfig cfg{1024, 256, true};

  auto buf = teststft::make_buffer(sr, teststft::constant(N, 1.0f));
  auto outE = fr->segment(teststft::span_of(buf), cfg);
  ASSERT_TRUE(outE.has_value());
  const auto& fb = *outE;

  const auto expected = expect_num_frames(N, cfg);
  EXPECT_EQ(fb.num_frames, expected);
  EXPECT_EQ(fb.data.size(), std::size_t(expected) * cfg.frame_size);

  // Last frame zero padding correctness
  if (expected > 0) {
    const std::size_t last_off = (std::size_t(expected) - 1) * cfg.frame_size;
    // Compute how many real samples should be in the last frame
    const std::size_t consumed_before_last = (std::size_t(expected) - 1) * cfg.hop_size;
    const std::size_t remaining = (N > consumed_before_last) ? (N - consumed_before_last) : 0;
    const std::size_t copy_n = std::min<std::size_t>(cfg.frame_size, remaining);
    for (std::size_t i = copy_n; i < cfg.frame_size; ++i) {
      ASSERT_NEAR(fb.data[last_off + i], 0.0f, 1e-7f);
    }
  }
}

TEST(Framer, ShortSignal_Cases) {
  auto fac = make_default_stft_factory();
  auto fr  = fac->create_framer();

  const std::uint32_t sr = 48000;
  const std::size_t   N  = 800; // < frame_size
  auto x = teststft::step(N);

  // pad_end=true -> exactly one frame, zero-padded
  {
    FramerConfig cfg{1024, 256, true};
    auto outE = fr->segment(teststft::span_of(teststft::make_buffer(sr, x)), cfg);
    ASSERT_TRUE(outE.has_value());
    EXPECT_EQ(outE->num_frames, 1u);
    EXPECT_EQ(outE->data.size(), std::size_t(1024));
    // zeros at tail
    for (std::size_t i = N; i < 1024; ++i)
      ASSERT_NEAR(outE->data[i], 0.0f, 1e-7f);
  }

  // pad_end=false -> zero frames
  {
    FramerConfig cfg{1024, 256, false};
    auto outE = fr->segment(teststft::span_of(teststft::make_buffer(sr, x)), cfg);
    ASSERT_TRUE(outE.has_value());
    EXPECT_EQ(outE->num_frames, 0u);
    EXPECT_TRUE(outE->data.empty());
  }
}

TEST(Framer, InvalidArgs) {
  auto fac = make_default_stft_factory();
  auto fr  = fac->create_framer();

  const auto buf0 = teststft::make_buffer(0, teststft::step(1024)); // sr=0
  {
    auto outE = fr->segment(teststft::span_of(buf0), FramerConfig{1024,256,true});
    ASSERT_FALSE(outE.has_value());
    EXPECT_EQ(outE.error(), UtilError::InvalidArgument);
  }
  const auto buf1 = teststft::make_buffer(48000, teststft::step(1024));
  {
    auto outE = fr->segment(teststft::span_of(buf1), FramerConfig{0,256,true});
    ASSERT_FALSE(outE.has_value());
    EXPECT_EQ(outE.error(), UtilError::InvalidArgument);
  }
  {
    auto outE = fr->segment(teststft::span_of(buf1), FramerConfig{1024,0,true});
    ASSERT_FALSE(outE.has_value());
    EXPECT_EQ(outE.error(), UtilError::InvalidArgument);
  }
  {
    auto outE = fr->segment(teststft::span_of(buf1), FramerConfig{1024,2048,true});
    ASSERT_FALSE(outE.has_value());
    EXPECT_EQ(outE.error(), UtilError::InvalidArgument);
  }
}
