#include <gtest/gtest.h>
#include <vector>
#include <numeric>

#include "afp/peaks/peaks.hpp"
#include "test_utils_spec.hpp"

using afp::peaks::IPeakFinder;
using afp::peaks::PeakParams;
using afp::peaks::make_default_peaks_factory;
using afp::util::Peak;
using afp::util::Spectrogram;
using afp::util::UtilError;

// Factory helper
static std::unique_ptr<IPeakFinder> make_finder() {
  auto fac = make_default_peaks_factory();
  return fac->create_peak_finder();
}

// -----------------------------
// A) Basic shape, bounds, ordering
// -----------------------------

TEST(Peaks, EmptyFramesOrBins_ReturnsEmpty_NoError) {
  auto f = make_finder();

  // Empty frames
  {
    auto S = testspec::make_spec(/*frames*/0, /*bins*/32, 0.f, /*sr*/1000,
                                           /*hop*/1000);
    auto r = f->find(S, testspec::pp_default());
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->empty());
  }
  // Empty bins
  {
    auto S = testspec::make_spec(/*frames*/4, /*bins*/0, 0.f, /*sr*/1000,
                                           /*hop*/1000);
    auto r = f->find(S, testspec::pp_default());
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->empty());
  }
}

TEST(Peaks, SizeMismatch_Raises) {
  auto f = make_finder();
  auto S = testspec::make_spec(3, 16, 0.f, 1000, 1000);
  S.log_mag.pop_back(); // break size == frames*bins
  auto r = f->find(S, testspec::pp_default());
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error(), UtilError::SizeMismatch);
}

TEST(Peaks, InvalidParams_Raises) {
  auto f = make_finder();

  // sr=0
  {
    auto S = testspec::make_spec(2, 8, 0.f, 0, 1000);
    auto r = f->find(S, testspec::pp_default());
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error(), UtilError::InvalidArgument);
  }
  // hop=0
  {
    auto S = testspec::make_spec(2, 8, 0.f, 1000, 0);
    auto r = f->find(S, testspec::pp_default());
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error(), UtilError::InvalidArgument);
  }
}

TEST(Peaks, OutputSorted_ByTimeThenFreq) {
  auto f = make_finder();
  const uint32_t T = 3, F = 12;
  auto S = testspec::make_spec(T, F, 0.f, 1000, 1000);

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 0;
  pp.neighborhood_freq_bins = 1;
  pp.threshold_db = -100.f; // allow everything above median

  // Make candidates in scrambled order of creation; sorted check is on output
  testspec::set_bin(S, 2, 3, 9.f);
  testspec::set_bin(S, 0, 11, 9.f);
  testspec::set_bin(S, 1, 0, 9.f);
  testspec::set_bin(S, 1, 7, 9.f);

  auto r = f->find(S, pp);
  ASSERT_TRUE(r.has_value());
  EXPECT_TRUE(testspec::sorted_tf(*r));
  EXPECT_TRUE(testspec::all_within_bounds(*r, T, F));
}

// -----------------------------
// B) Local-maximum detection across time×freq
// -----------------------------
TEST(Peaks, LocalMax_CenterBeatsNeighbors) {
  auto f = make_finder();

  const uint32_t T = 3, F = 11;
  auto S = testspec::make_spec(T, F, 0.f, 1000, 1000);

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 1;
  pp.neighborhood_freq_bins = 2;
  pp.threshold_db = 1.f; // median ≈ 0, so only 9/10 dB bins qualify
  pp.target_peak_density_per_sec_min = 0; // disable backfill
  pp.target_peak_density_per_sec_max = 0; // disable capping

  // Dome around (1,5): neighbors 9 dB, center 10 dB
  for (int dt = -1; dt <= 1; ++dt) {
    const int t = 1 + dt;
    if (t < 0 || t >= (int)T) continue;
    for (int df = -2; df <= 2; ++df) {
      const int k = 5 + df;
      if (k < 0 || k >= (int)F) continue;
      testspec::set_bin(S, t, k, 9.f);
    }
  }
  testspec::set_bin(S, 1, 5, 10.f);

  auto r = f->find(S, pp);
  ASSERT_TRUE(r.has_value());
  ASSERT_EQ(r->size(), 1u);
  EXPECT_EQ((*r)[0].t, 1u);
  EXPECT_EQ((*r)[0].f, 5u);
  EXPECT_NEAR((*r)[0].strength_db, 10.f, 1e-6f);
}


TEST(Peaks, Plateau_TiesAllowedThenPrunedByNMS) {
  auto f = make_finder();
  const uint32_t T = 1, F = 16;
  auto S = testspec::make_spec(T, F, 0.f, 1000, 1000);

  // Plateau 5,6,7 at +10 dB; others 0
  testspec::set_bin(S, 0, 5, 10.f);
  testspec::set_bin(S, 0, 6, 10.f);
  testspec::set_bin(S, 0, 7, 10.f);

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 0;
  pp.neighborhood_freq_bins = 3;
  pp.threshold_db = 1.f; // exclude 0 dB background from "hi"
  pp.target_peak_density_per_sec_min = 0; // disable backfill
  pp.target_peak_density_per_sec_max = 0; // disable capping

  // With min_sep=2 -> {5,7}
  {
    auto pp2 = pp;
    pp2.min_freq_separation_bins = 2;
    auto r = f->find(S, pp2);
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(r->size(), 2u);
    EXPECT_EQ((*r)[0].f, 5u);
    EXPECT_EQ((*r)[1].f, 7u);
  }
  // With min_sep=3 -> only one (any of {5,6,7}, ties broken by greedy order)
  {
    auto pp3 = pp;
    pp3.min_freq_separation_bins = 3;
    auto r = f->find(S, pp3);
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(r->size(), 1u);
    EXPECT_TRUE((*r)[0].f == 5u || (*r)[0].f == 6u || (*r)[0].f == 7u);
  }
}


TEST(Peaks, BoundaryFramesAndBins_AreHandled) {
  auto f = make_finder();
  const uint32_t T = 3, F = 8;
  auto S = testspec::make_spec(T, F, 0.f, 1000, 1000);

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 1;
  pp.neighborhood_freq_bins = 1;
  pp.threshold_db = 1.f; // exclude 0 dB background
  pp.target_peak_density_per_sec_min = 0; // no backfill
  pp.target_peak_density_per_sec_max = 0;

  testspec::set_bin(S, 0, 0, 10.f);
  testspec::set_bin(S, 0, F - 1, 10.f);
  testspec::set_bin(S, T - 1, 0, 10.f);
  testspec::set_bin(S, T - 1, F - 1, 10.f);

  auto r = f->find(S, pp);
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->size(), 4u);
  EXPECT_TRUE(testspec::all_within_bounds(*r, T, F));
}


// -----------------------------
// C) Adaptive threshold (median + threshold_db)
// -----------------------------

TEST(Peaks, Thresholding_AboveMedian_Passes_BelowFails) {
  auto f = make_finder();
  auto S = testspec::make_spec(/*T*/1, /*F*/16, 0.f, 1000, 1000);

  testspec::set_bin(S, 0, 7, 9.f); // single candidate

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 0;
  pp.neighborhood_freq_bins = 1;
  pp.min_freq_separation_bins = 1;
  // Turn off per-frame density behavior so background isn't backfilled
  pp.target_peak_density_per_sec_min = 0;
  pp.target_peak_density_per_sec_max = 0;

  // thr=6 -> keep the 9 dB bin
  {
    auto pp6 = pp;
    pp6.threshold_db = 6.f;
    auto r = f->find(S, pp6);
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(r->size(), 1u);
    EXPECT_EQ((*r)[0].f, 7u);
  }
  // thr=10 -> discard (9 < 10 above median)
  {
    auto pp10 = pp;
    pp10.threshold_db = 10.f;
    auto r = f->find(S, pp10);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->empty());
  }
}


TEST(Peaks, DifferentFrameMedians_AdjustThresholds) {
  auto f = make_finder();
  auto S = testspec::make_spec(/*T*/2, /*F*/16, 0.f, 1000, 1000);

  // Frame 0: baseline 0, one +9 => median ~0, thr=6 picks it
  testspec::set_bin(S, 0, 3, 9.f);

  // Frame 1: +5 offset everywhere, one +13 (still +8 over median ~5)
  for (uint32_t k = 0; k < 16; ++k) testspec::set_bin(S, 1, k, 5.f);
  testspec::set_bin(S, 1, 9, 13.f);

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 0;
  pp.neighborhood_freq_bins = 1;
  pp.threshold_db = 6.f;
  // Disable min/max density so no backfill/capping happens
  pp.target_peak_density_per_sec_min = 0;
  pp.target_peak_density_per_sec_max = 0;

  auto r = f->find(S, pp);
  ASSERT_TRUE(r.has_value());
  ASSERT_EQ(r->size(), 2u);
  EXPECT_EQ((*r)[0].t, 0u);
  EXPECT_EQ((*r)[1].t, 1u);
}

// -----------------------------
// D) Per-frame NMS (min frequency separation)
// -----------------------------

TEST(Peaks, NMS_MinSeparation_Enforced) {
  auto f = make_finder();
  auto S = testspec::make_spec(/*T*/1, /*F*/12, 0.f, 1000, 1000);

  // Peaks: 0,3,6,9 all strong
  testspec::set_bin(S, 0, 0, 12.f);
  testspec::set_bin(S, 0, 3, 11.f);
  testspec::set_bin(S, 0, 6, 10.f);
  testspec::set_bin(S, 0, 9, 9.f);

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 0;
  pp.neighborhood_freq_bins = 1;
  pp.threshold_db = 1.f; // exclude 0 dB background from "hi"
  pp.target_peak_density_per_sec_min = 0; // disable backfill
  pp.target_peak_density_per_sec_max = 0;

  // min_sep=5 -> keep {0,6}
  {
    auto p = pp;
    p.min_freq_separation_bins = 5;
    auto r = f->find(S, p);
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(r->size(), 2u);
    EXPECT_EQ((*r)[0].f, 0u);
    EXPECT_EQ((*r)[1].f, 6u);
  }
  // min_sep=3 -> keep all (df=3 is allowed; prune only if df<3)
  {
    auto p = pp;
    p.min_freq_separation_bins = 3;
    auto r = f->find(S, p);
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(r->size(), 4u);
    EXPECT_EQ((*r)[0].f, 0u);
    EXPECT_EQ((*r)[1].f, 3u);
    EXPECT_EQ((*r)[2].f, 6u);
    EXPECT_EQ((*r)[3].f, 9u);
  }
}


TEST(Peaks, NMS_Stable_OrderingAfterKeep) {
  auto f = make_finder();
  auto S = testspec::make_spec(/*T*/1, /*F*/10, 0.f, 1000, 1000);

  // Make 2 and 4 local maxima; 3 is lower than both
  testspec::set_bin(S, 0, 2, 10.f);
  testspec::set_bin(S, 0, 3, 8.5f); // was 9.5f
  testspec::set_bin(S, 0, 4, 9.0f);

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 0;
  pp.neighborhood_freq_bins = 1; // keep neighbors in local-max test
  pp.threshold_db = 1.f; // exclude 0 dB background
  pp.min_freq_separation_bins = 2; // NMS: df < 2 pruned; df == 2 allowed
  pp.target_peak_density_per_sec_min = 0;
  pp.target_peak_density_per_sec_max = 0;

  auto r = f->find(S, pp);
  ASSERT_TRUE(r.has_value());
  ASSERT_EQ(r->size(), 2u); // expect {2,4}
  EXPECT_EQ((*r)[0].f, 2u);
  EXPECT_EQ((*r)[1].f, 4u);
  EXPECT_TRUE(testspec::sorted_tf(*r));
}


// -----------------------------
// E) Per-frame density: min & max (and backfill)
// -----------------------------

TEST(Peaks, MaxPerFrame_CapsStrongSet) {
  auto f = make_finder();

  // Use fps=1 so min/max per sec == per frame
  auto S = testspec::make_spec(/*T*/1, /*F*/64, 0.f, /*sr*/1000, /*hop*/1000);

  // Many separated strong peaks
  for (uint32_t k = 2; k < 64; k += 5)
    testspec::set_bin(S, 0, k, 20.f + float(k % 7)); // distinct strengths

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 0;
  pp.neighborhood_freq_bins = 1;
  pp.threshold_db = 0.f;
  pp.min_freq_separation_bins = 4; // ensure separation
  testspec::set_density(pp, /*min*/0, /*max*/3);

  auto r = f->find(S, pp);
  ASSERT_TRUE(r.has_value());
  ASSERT_EQ(r->size(), 3u);

  // Verify they are the top-3 by strength among the separated set
  std::vector<Peak> all;
  for (uint32_t k = 2; k < 64; k += 5) all.push_back(
      Peak{0, (afp::util::BinIndex)k, 20.f + float(k % 7)});
  std::sort(all.begin(), all.end(), [](const Peak& a, const Peak& b) {
    return a.strength_db > b.strength_db;
  });
  std::vector<afp::util::BinIndex> top3 = {all[0].f, all[1].f, all[2].f};
  std::vector<afp::util::BinIndex> got = {(*r)[0].f, (*r)[1].f, (*r)[2].f};
  std::sort(top3.begin(), top3.end());
  std::sort(got.begin(), got.end());
  EXPECT_EQ(top3, got);
}

TEST(Peaks, Backfill_FromLowSet_MeetsMinPerFrame) {
  auto f = make_finder();
  auto S = testspec::make_spec(/*T*/1, /*F*/64, 0.f, 1000, 1000);

  // One above thr (10), several just below (9.5), all well-separated
  testspec::set_bin(S, 0, 5, 10.f);
  for (uint32_t k = 10; k < 64; k += 10) testspec::set_bin(S, 0, k, 9.5f);

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 0;
  pp.neighborhood_freq_bins = 1;
  pp.threshold_db = 10.f; // med ~0 → only the 10 dB is above threshold
  pp.min_freq_separation_bins = 4;
  testspec::set_density(pp, /*min*/4, /*max*/5);

  auto r = f->find(S, pp);
  ASSERT_TRUE(r.has_value());
  EXPECT_GE(r->size(), 4u);
  EXPECT_LE(r->size(), 5u);
}

TEST(Peaks, Backfill_Respects_MinSeparation) {
  auto f = make_finder();

  // Make the frame small enough that every other bin is within min_sep of the kept peak.
  auto S = testspec::make_spec(/*T*/1, /*F*/15, 0.f, 1000, 1000);

  // One above-threshold local max at f=8
  testspec::set_bin(S, 0, 8, 10.f);
  // Some nearby below-threshold bumps (not strictly necessary, but realistic)
  testspec::set_bin(S, 0, 9, 9.5f);
  testspec::set_bin(S, 0, 10, 9.4f);
  testspec::set_bin(S, 0, 11, 9.3f);

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 0;
  pp.neighborhood_freq_bins = 1;
  pp.threshold_db = 10.f; // only 10 dB bin is "hi"
  pp.min_freq_separation_bins = 20; // larger than any |f - 8| in F=15
  testspec::set_density(pp, /*min*/3, /*max*/6); // backfill will try...

  auto r = f->find(S, pp);
  ASSERT_TRUE(r.has_value());
  // ...but can't add anything due to min_sep -> stays below requested min.
  EXPECT_LT(r->size(), 3u);
  // And the single kept peak is the above-threshold one at f=8.
  ASSERT_FALSE(r->empty());
  EXPECT_EQ((*r)[0].f, 8u);
}


TEST(Peaks, MaxPerFrame_DominatesAfterBackfill) {
  auto f = make_finder();
  auto S = testspec::make_spec(/*T*/1, /*F*/64, 0.f, 1000, 1000);

  // One above thr, many below thr & well-separated
  testspec::set_bin(S, 0, 5, 10.f);
  for (uint32_t k = 12; k < 64; k += 5) testspec::set_bin(S, 0, k, 9.5f);

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 0;
  pp.neighborhood_freq_bins = 1;
  pp.threshold_db = 10.f;
  pp.min_freq_separation_bins = 4;
  testspec::set_density(pp, /*min*/4, /*max*/4); // exact cap

  auto r = f->find(S, pp);
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->size(), 4u);
  // should stop exactly at max even with more low candidates
}

// -----------------------------
// F) Strength propagation and indexing
// -----------------------------

TEST(Peaks, StrengthDb_MatchesSpectrogramValues) {
  auto f = make_finder();
  auto S = testspec::make_spec(/*T*/1, /*F*/12, 0.f, 1000, 1000);

  const std::vector<std::pair<uint32_t, float>> pts = {
      {2, 7.f}, {5, 9.3f}, {9, 2.2f}};
  for (auto [k,db] : pts) testspec::set_bin(S, 0, k, db);

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 0;
  pp.neighborhood_freq_bins = 1;
  pp.threshold_db = -100.f;

  auto r = f->find(S, pp);
  ASSERT_TRUE(r.has_value());
  for (const auto& p : *r) {
    EXPECT_NEAR(p.strength_db, S.log_mag[p.t * S.num_bins + p.f], 1e-7f);
  }
}

TEST(Peaks, IndicesWithinBounds) {
  auto f = make_finder();
  const uint32_t T = 4, F = 20;
  auto S = testspec::make_spec(T, F, 0.f, 1000, 1000);

  // sprinkle peaks
  testspec::set_bin(S, 0, 0, 9.f);
  testspec::set_bin(S, 3, 19, 9.f);
  testspec::set_bin(S, 2, 5, 9.f);

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 1;
  pp.neighborhood_freq_bins = 2;
  pp.threshold_db = 0.f;

  auto r = f->find(S, pp);
  ASSERT_TRUE(r.has_value());
  EXPECT_TRUE(testspec::all_within_bounds(*r, T, F));
}

// -----------------------------
// G) Multi-frame temporal effect
// -----------------------------

TEST(Peaks, TemporalNeighborhood_SuppressesNearTimePeaks) {
  auto f = make_finder();
  auto S = testspec::make_spec(/*T*/2, /*F*/16, 0.f, 1000, 1000);

  // Same bin strong in both frames; frame 1 slightly higher
  testspec::set_bin(S, 0, 8, 9.f);
  testspec::set_bin(S, 1, 8, 10.f);

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 1; // look across adjacent frames
  pp.neighborhood_freq_bins = 0; // only same bin in freq
  pp.threshold_db = 1.f; // exclude 0 dB background
  pp.target_peak_density_per_sec_min = 0; // no backfill
  pp.target_peak_density_per_sec_max = 0;

  auto r = f->find(S, pp);
  ASSERT_TRUE(r.has_value());
  ASSERT_EQ(r->size(), 1u); // only the higher one survives
  EXPECT_EQ((*r)[0].t, 1u);
  EXPECT_EQ((*r)[0].f, 8u);
}


// -----------------------------
// H) Repeatability
// -----------------------------

TEST(Peaks, Repeatability_SameInputSameParams) {
  auto f = make_finder();
  auto S = testspec::make_spec(/*T*/3, /*F*/32, 0.f, 1000, 1000);

  // Deterministic pattern with spaced peaks
  for (uint32_t t = 0; t < 3; ++t)
    for (uint32_t k = t; k < 32; k += 7)
      testspec::set_bin(S, t, k, 8.f + float((t + k) % 5));

  PeakParams pp = testspec::pp_default();
  pp.neighborhood_time_frames = 1;
  pp.neighborhood_freq_bins = 2;
  pp.threshold_db = -1.f;
  pp.min_freq_separation_bins = 2;
  testspec::set_density(pp, 0, 10);

  auto r1 = f->find(S, pp);
  auto r2 = f->find(S, pp);
  ASSERT_TRUE(r1.has_value());
  ASSERT_TRUE(r2.has_value());
  ASSERT_EQ(r1->size(), r2->size());
  for (size_t i = 0; i < r1->size(); ++i) {
    EXPECT_EQ((*r1)[i].t, (*r2)[i].t);
    EXPECT_EQ((*r1)[i].f, (*r2)[i].f);
    EXPECT_NEAR((*r1)[i].strength_db, (*r2)[i].strength_db, 1e-7f);
  }
}
