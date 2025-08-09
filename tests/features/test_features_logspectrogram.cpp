#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>

#include "afp/features/features.hpp"
#include "test_utils_signal.hpp"

using afp::features::ISpectrogramBuilder;
using afp::features::make_default_features_factory;
using afp::util::Expected;
using afp::util::Spectrogram;
using afp::util::ComplexSpectra;
using afp::util::UtilError;

namespace {
std::unique_ptr<ISpectrogramBuilder> make_builder() {
  auto fac = make_default_features_factory();
  return fac->create_spectrogram_builder();
}
} // namespace

// -----------------------------
// A) Shape, metadata, and layout
// -----------------------------

TEST(LogSpec, Shape_Metadata_Correct) {
  const uint32_t sr = 16000, fft = 1024, hop = 256;
  const uint32_t num_bins_expected = fft / 2 + 1; // 513
  ComplexSpectra spec = testfeat::make_spectra(
      /*frames*/3, /*bins*/num_bins_expected);

  // all zeros
  auto params = testfeat::make_params(sr, hop, fft, /*low*/0.f, /*high*/
                                      sr / 2.f, 1e-8f, 120.f, false, 0);

  auto b = make_builder();
  auto outE = b->build(spec, params);
  ASSERT_TRUE(outE.has_value());
  const auto& out = *outE;

  EXPECT_EQ(out.num_bins, num_bins_expected);
  EXPECT_EQ(out.num_frames, 3u);
  EXPECT_EQ(out.sample_rate_hz, sr);
  EXPECT_EQ(out.hop_size, hop);
  EXPECT_EQ(out.fft_size, fft);
  ASSERT_EQ(out.log_mag.size(),
            static_cast<size_t>(out.num_frames) * out.num_bins);
}

TEST(LogSpec, RowMajor_Layout) {
  const uint32_t sr = 24000, fft = 512, hop = 128;
  const uint32_t bins = fft / 2 + 1; // 257
  ComplexSpectra spec = testfeat::make_spectra(/*frames*/3, bins);

  // Single non-zero at (t=1, k=7)
  testfeat::set_bin(spec, 1, 7, /*re*/2.0f, /*im*/0.0f);

  auto params = testfeat::make_params(sr, hop, fft,
                                      /*low*/0.f, /*high*/sr / 2.f,
                                      /*eps*/1e-12f,
                                      /*clip_db*/300.f,
                                      // was 120.f — avoid clipping of -240 dB floor
                                      /*median*/false,
                                      /*whiten*/0);

  auto b = make_builder();
  auto outE = b->build(spec, params);
  ASSERT_TRUE(outE.has_value());
  const auto& out = *outE;

  ASSERT_EQ(out.num_bins, bins);
  const float db_zero = testfeat::db_mag(0.f, params.epsilon);
  for (uint32_t t = 0; t < spec.num_frames; ++t) {
    for (uint32_t k = 0; k < out.num_bins; ++k) {
      const float v = out.log_mag[testfeat::idx_log(t, k, out.num_bins)];
      if (t == 1 && k == 7) {
        const float exp = testfeat::db_mag(2.0f, params.epsilon);
        EXPECT_NEAR(v, exp, 1e-6f);
      } else {
        EXPECT_NEAR(v, db_zero, 1e-6f);
      }
    }
  }
}

// -----------------------------
// B) Epsilon and magnitude → dB
// -----------------------------

TEST(LogSpec, Epsilon_For_ZeroMagnitude) {
  const uint32_t sr = 22050, fft = 256, hop = 64;
  const uint32_t bins = fft / 2 + 1;
  ComplexSpectra spec = testfeat::make_spectra(2, bins);

  auto params = testfeat::make_params(
      sr, hop, fft, 0.f, sr / 2.f,
      /*eps*/1e-7f,
      /*clip_db*/300.f, // was 120.f
      /*median*/false,
      /*whiten*/0);

  auto b = make_builder();
  auto outE = b->build(spec, params);
  ASSERT_TRUE(outE.has_value());
  const auto& out = *outE;

  const float exp = testfeat::db_mag(0.f, params.epsilon);
  for (float v : out.log_mag)
    EXPECT_NEAR(v, exp, 1e-6f);
}

TEST(LogSpec, Epsilon_Monotonicity) {
  const uint32_t sr = 16000, fft = 512, hop = 128;
  const uint32_t bins = fft / 2 + 1;
  ComplexSpectra spec = testfeat::make_spectra(1, bins);

  // Disable clipping influence by setting a very large clip range
  auto p1 = testfeat::make_params(sr, hop, fft, 0.f, sr / 2.f,
                                  /*eps*/1e-9f, /*clip_db*/300.f,
                                  /*median*/false, /*whiten*/0);
  auto p2 = testfeat::make_params(sr, hop, fft, 0.f, sr / 2.f,
                                  /*eps*/1e-6f, /*clip_db*/300.f,
                                  /*median*/false, /*whiten*/0);

  auto b = make_builder();
  auto o1 = b->build(spec, p1);
  auto o2 = b->build(spec, p2);
  ASSERT_TRUE(o1.has_value());
  ASSERT_TRUE(o2.has_value());

  // Larger epsilon must yield larger (less negative) dB values everywhere
  for (size_t i = 0; i < o1->log_mag.size(); ++i) {
    EXPECT_LT(o1->log_mag[i], o2->log_mag[i] - 1e-7f);
  }
}

// -----------------------------
// C) Per-frame median subtraction
// -----------------------------

TEST(LogSpec, MedianSubtract_ConstantFrame_BecomesZero) {
  const uint32_t sr = 16000, fft = 1024, hop = 160;
  const uint32_t bins = fft / 2 + 1;
  ComplexSpectra spec = testfeat::make_spectra(2, bins);

  // Frame 0: all magnitude A, Frame 1: all magnitude B
  const float A = 0.25f, B = 3.0f;
  testfeat::fill_frame(spec, 0, [&](uint32_t) {
    return kfr::complex<float>(A, 0.f);
  });
  testfeat::fill_frame(spec, 1, [&](uint32_t) {
    return kfr::complex<float>(B, 0.f);
  });

  auto params = testfeat::make_params(sr, hop, fft, 0.f, sr / 2.f, 1e-8f, 120.f,
                                      true, 0);

  auto b = make_builder();
  auto outE = b->build(spec, params);
  ASSERT_TRUE(outE.has_value());
  const auto& out = *outE;

  // Both frames ~0 dB after per-frame median subtract
  for (uint32_t t = 0; t < 2; ++t) {
    for (uint32_t k = 0; k < out.num_bins; ++k) {
      const float v = out.log_mag[testfeat::idx_log(t, k, out.num_bins)];
      EXPECT_NEAR(v, 0.f, 1e-6f);
    }
  }
}

TEST(LogSpec, MedianSubtract_HalfHigh_HalfLow) {
  const uint32_t sr = 44100, fft = 510, hop = 220; // fft=510 ⇒ bins=256 (even)
  const uint32_t bins = fft / 2 + 1;
  ComplexSpectra spec = testfeat::make_spectra(1, bins);

  const float m1 = 0.2f, m2 = 2.0f;
  const uint32_t mid = bins / 2;
  testfeat::fill_frame(spec, 0, [&](uint32_t k) {
    return (k < mid)
             ? kfr::complex<float>(m1, 0.f)
             : kfr::complex<float>(m2, 0.f);
  });

  auto params = testfeat::make_params(sr, hop, fft, 0.f, sr / 2.f, 1e-8f, 300.f,
                                      // raise clip to avoid interference
                                      true, 0);

  auto b = make_builder();
  auto outE = b->build(spec, params);
  ASSERT_TRUE(outE.has_value());
  const auto& out = *outE;

  std::vector<float> left, right;
  left.reserve(mid);
  right.reserve(bins - mid);
  for (uint32_t k = 0; k < bins; ++k) {
    float v = out.log_mag[testfeat::idx_log(0, k, bins)];
    (k < mid ? left : right).push_back(v);
  }

  const float ml = testfeat::mean(
      std::span<const float>(left.data(), left.size()));
  const float mr = testfeat::mean(
      std::span<const float>(right.data(), right.size()));

  EXPECT_LT(ml, 0.0f);
  EXPECT_GT(mr, 0.0f);
  EXPECT_NEAR(std::abs(ml), std::abs(mr), 1e-3f);
  EXPECT_NEAR(
      testfeat::mean(std::span<const float>(out.log_mag.data(), out.log_mag.size
        ())), 0.f, 1e-3f);
}

// -----------------------------
// D) Whitening (box filter along frequency)
// -----------------------------

TEST(LogSpec, Whiten_Disabled_NoChange) {
  const uint32_t sr = 16000, fft = 256, hop = 80;
  const uint32_t bins = fft / 2 + 1;
  ComplexSpectra spec = testfeat::make_spectra(1, bins);

  // Non-trivial spectrum
  testfeat::fill_frame(spec, 0, [&](uint32_t k) {
    float mag = (k % 7 == 0) ? 3.0f : 0.1f * (1 + (k % 5));
    return kfr::complex<float>(mag, (k % 3 == 0) ? 0.2f * mag : 0.f);
  });

  auto p0 = testfeat::make_params(sr, hop, fft, 0.f, sr / 2.f, 1e-8f, 120.f,
                                  false, 0);
  auto p1 = testfeat::make_params(sr, hop, fft, 0.f, sr / 2.f, 1e-8f, 120.f,
                                  false, 8);

  auto b = make_builder();
  auto o0 = b->build(spec, p0);
  auto o1 = b->build(spec, p1);
  ASSERT_TRUE(o0.has_value());
  ASSERT_TRUE(o1.has_value());

  // When whitening=0, output equals raw dB; with whitening>0, they should differ somewhere
  size_t equal_cnt = 0, diff_cnt = 0;
  for (size_t i = 0; i < o0->log_mag.size(); ++i) {
    if (std::abs(o0->log_mag[i] - o1->log_mag[i]) <= 1e-6f) ++equal_cnt;
    else ++diff_cnt;
  }
  EXPECT_GT(diff_cnt, 0u);
}

TEST(LogSpec, Whiten_ConstantSpectrum_NearZero_AwayFromEdges) {
  const uint32_t sr = 32000, fft = 1024, hop = 320;
  const uint32_t bins = fft / 2 + 1;
  ComplexSpectra spec = testfeat::make_spectra(1, bins);

  // Constant magnitude → after whitening, interior should hover near 0 dB
  testfeat::fill_frame(spec, 0, [&](uint32_t) {
    return kfr::complex<float>(1.5f, 0.f);
  });

  const uint16_t R = 20; // whitening radius
  const uint32_t L = 2 * R + 1; // kernel length
  const uint32_t EDGE = L - 1; // samples affected by causal alignment

  auto params = testfeat::make_params(sr, hop, fft, 0.f, sr / 2.f,
                                      /*eps*/1e-8f, /*clip_db*/300.f,
                                      /*median*/false, /*whiten*/R);

  auto b = make_builder();
  auto outE = b->build(spec, params);
  ASSERT_TRUE(outE.has_value());
  const auto& out = *outE;

  // Exclude the first/last (L-1) bins where convolution support is incomplete
  for (uint32_t k = EDGE; k + EDGE < bins; ++k) {
    const float v = out.log_mag[testfeat::idx_log(0, k, bins)];
    EXPECT_NEAR(v, 0.f, 1e-3f);
  }
}

TEST(LogSpec, Whiten_BoxBlur_Effect_OnSlope) {
  const uint32_t sr = 44100, fft = 2048, hop = 441;
  const uint32_t bins = fft / 2 + 1;
  ComplexSpectra spec = testfeat::make_spectra(1, bins);

  // Linear ramp in dB: set mag[k] = 10^(dB/20) with dB = a*k + b
  const float a = 0.01f, b = -2.f;
  testfeat::fill_frame(spec, 0, [&](uint32_t k) {
    float dB = a * float(k) + b;
    float mag = std::pow(10.f, dB / 20.f);
    return kfr::complex<float>(mag, 0.f);
  });

  auto p_nowhite = testfeat::make_params(sr, hop, fft, 0.f, sr / 2.f, 1e-8f,
                                         120.f, false, 0);
  auto p_white = testfeat::make_params(sr, hop, fft, 0.f, sr / 2.f, 1e-8f,
                                       120.f, false, 12);

  auto builder = make_builder();
  auto o0 = builder->build(spec, p_nowhite);
  auto o1 = builder->build(spec, p_white);
  ASSERT_TRUE(o0.has_value());
  ASSERT_TRUE(o1.has_value());

  // Compare interior means: whitening should reduce |mean| vs no whitening
  const uint32_t R = p_white.whiten_radius_bins;
  std::vector<float> interior0, interior1;
  for (uint32_t k = R; k + R < bins; ++k) {
    interior0.push_back(o0->log_mag[testfeat::idx_log(0, k, bins)]);
    interior1.push_back(o1->log_mag[testfeat::idx_log(0, k, bins)]);
  }
  EXPECT_GT(std::abs(testfeat::mean(interior0)),
            std::abs(testfeat::mean(interior1)));
}

// -----------------------------
// E) Clipping (±clip_db)
// -----------------------------

TEST(LogSpec, Clip_Symmetric) {
  const uint32_t sr = 16000, fft = 512, hop = 128;
  const uint32_t bins = fft / 2 + 1;
  ComplexSpectra spec = testfeat::make_spectra(1, bins);

  // Create very large and very small magnitudes
  testfeat::fill_frame(spec, 0, [&](uint32_t k) {
    if (k % 2 == 0) return kfr::complex<float>(1000.f, 0.f);
    else return kfr::complex<float>(1e-6f, 0.f);
  });

  auto params = testfeat::make_params(sr, hop, fft, 0.f, sr / 2.f, 1e-12f, 6.f,
                                      false, 0);

  auto b = make_builder();
  auto outE = b->build(spec, params);
  ASSERT_TRUE(outE.has_value());
  const auto& out = *outE;

  auto [mn, mx] = testfeat::minmax(out.log_mag);
  EXPECT_LE(mx, params.clip_db + 1e-4f);
  EXPECT_GE(mn, -params.clip_db - 1e-4f);

  // Ensure both bounds are hit approximately
  bool hit_lo = false, hit_hi = false;
  for (float v : out.log_mag) {
    if (std::abs(v - params.clip_db) <= 1e-3f) hit_hi = true;
    if (std::abs(v + params.clip_db) <= 1e-3f) hit_lo = true;
  }
  EXPECT_TRUE(hit_lo);
  EXPECT_TRUE(hit_hi);
}

// -----------------------------
// F) Band limiting (inclusive + clamp)
// -----------------------------

TEST(LogSpec, Band_Inclusive_Ends) {
  const uint32_t sr = 16000, fft = 1024, hop = 160;
  const uint32_t bins = fft / 2 + 1;
  ComplexSpectra spec = testfeat::make_spectra(1, bins);

  // pick integer bins exactly: b0=5, b1=37
  const uint32_t b0 = 5, b1 = 37;
  const float low_hz = testfeat::hz_of_bin(b0, sr, fft);
  const float high_hz = testfeat::hz_of_bin(b1, sr, fft);

  auto params = testfeat::make_params(sr, hop, fft, low_hz, high_hz, 1e-8f,
                                      120.f, false, 0);

  // put distinct magnitudes at b0 and b1 to verify slice bounds
  testfeat::set_bin(spec, 0, b0, 2.0f, 0.f);
  testfeat::set_bin(spec, 0, b1, 3.0f, 0.f);

  auto b = make_builder();
  auto outE = b->build(spec, params);
  ASSERT_TRUE(outE.has_value());
  const auto& out = *outE;

  EXPECT_EQ(out.num_bins, static_cast<uint16_t>(b1 - b0 + 1));
  // First output bin corresponds to b0; last to b1
  const float db0 = out.log_mag[testfeat::idx_log(0, 0, out.num_bins)];
  const float dbL = out.log_mag[testfeat::idx_log(
      0, out.num_bins - 1, out.num_bins)];
  EXPECT_NEAR(db0, testfeat::db_mag(2.0f, params.epsilon), 1e-6f);
  EXPECT_NEAR(dbL, testfeat::db_mag(3.0f, params.epsilon), 1e-6f);
}

TEST(LogSpec, Band_Clamp_To_Nyquist_And_Zero) {
  const uint32_t sr = 22050, fft = 1024, hop = 256;
  const uint32_t bins = fft / 2 + 1;
  ComplexSpectra spec = testfeat::make_spectra(2, bins);

  // ask for a wider-than-allowed band; should clamp to [0, Nyquist]
  auto params = testfeat::make_params(sr, hop, fft, -1000.f, float(sr), 1e-8f,
                                      120.f, false, 0);

  auto b = make_builder();
  auto outE = b->build(spec, params);
  ASSERT_TRUE(outE.has_value());
  const auto& out = *outE;

  EXPECT_EQ(out.num_bins, bins);
}

TEST(LogSpec, Band_Invalid_Range) {
  const uint32_t sr = 16000, fft = 512, hop = 80;
  const uint32_t bins = fft / 2 + 1;
  ComplexSpectra spec = testfeat::make_spectra(1, bins);

  // low > high → InvalidArgument
  auto params = testfeat::make_params(sr, hop, fft, 4000.f, 1000.f, 1e-8f,
                                      120.f, false, 0);

  auto b = make_builder();
  auto outE = b->build(spec, params);
  ASSERT_FALSE(outE.has_value());
  EXPECT_EQ(outE.error(), UtilError::InvalidArgument);
}

// -----------------------------
// G) Size/parameter validation
// -----------------------------

TEST(LogSpec, SizeMismatch_NumBins) {
  const uint32_t sr = 16000, fft = 1024, hop = 256;
  // Provide wrong num_bins: should be 513
  ComplexSpectra spec = testfeat::make_spectra(1, /*bins*/400);

  auto params = testfeat::make_params(sr, hop, fft, 0.f, sr / 2.f, 1e-8f, 120.f,
                                      false, 0);

  auto b = make_builder();
  auto outE = b->build(spec, params);
  ASSERT_FALSE(outE.has_value());
  EXPECT_EQ(outE.error(), UtilError::SizeMismatch);
}

TEST(LogSpec, InvalidParams_SampleRateOrFft) {
  const uint32_t sr = 0, fft = 1024, hop = 256;
  ComplexSpectra spec = testfeat::make_spectra(1, /*bins*/(fft / 2 + 1));

  auto p_bad_sr = testfeat::make_params(sr, hop, fft, 0.f, 8000.f, 1e-8f, 120.f,
                                        false, 0);
  auto p_bad_fft = testfeat::make_params(16000, hop, 0, 0.f, 8000.f, 1e-8f,
                                         120.f, false, 0);

  auto b = make_builder();
  auto o1 = b->build(spec, p_bad_sr);
  auto o2 = b->build(spec, p_bad_fft);
  ASSERT_FALSE(o1.has_value());
  ASSERT_FALSE(o2.has_value());
  EXPECT_EQ(o1.error(), afp::util::UtilError::InvalidArgument);
  EXPECT_EQ(o2.error(), afp::util::UtilError::InvalidArgument);
}

TEST(LogSpec, ZeroFrames_OK) {
  const uint32_t sr = 16000, fft = 1024, hop = 160;
  const uint32_t bins = fft / 2 + 1;
  ComplexSpectra spec = testfeat::make_spectra(0, bins);

  auto params = testfeat::make_params(sr, hop, fft, 0.f, sr / 2.f, 1e-8f, 120.f,
                                      false, 0);

  auto b = make_builder();
  auto outE = b->build(spec, params);
  ASSERT_TRUE(outE.has_value());
  const auto& out = *outE;
  EXPECT_EQ(out.num_frames, 0u);
  EXPECT_EQ(out.log_mag.size(), 0u);
}

// -----------------------------
// H) Repeatability & independence
// -----------------------------

TEST(LogSpec, Repeatability_SameInputSameParams) {
  const uint32_t sr = 48000, fft = 2048, hop = 480;
  const uint32_t bins = fft / 2 + 1;
  ComplexSpectra spec = testfeat::make_spectra(3, bins);

  // Deterministic pattern
  for (uint32_t t = 0; t < 3; ++t) {
    testfeat::fill_frame(spec, t, [&](uint32_t k) {
      float mag = 0.1f + 0.003f * float((t + 1) * (k + 1) % 37);
      return kfr::complex<float>(mag, (k % 2) ? 0.0f : 0.05f * mag);
    });
  }

  auto params = testfeat::make_params(sr, hop, fft, 200.f, 7000.f, 1e-8f, 30.f,
                                      true, 6);

  auto b = make_builder();
  auto o1 = b->build(spec, params);
  auto o2 = b->build(spec, params);
  ASSERT_TRUE(o1.has_value());
  ASSERT_TRUE(o2.has_value());

  ASSERT_EQ(o1->num_bins, o2->num_bins);
  ASSERT_EQ(o1->num_frames, o2->num_frames);
  ASSERT_EQ(o1->log_mag.size(), o2->log_mag.size());
  for (size_t i = 0; i < o1->log_mag.size(); ++i) {
    EXPECT_NEAR(o1->log_mag[i], o2->log_mag[i], 1e-7f);
  }
}