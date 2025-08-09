#include <gtest/gtest.h>
#include "afp/peaks/peaks.hpp"
#include "test_utils_spec.hpp"

using afp::peaks::IPeaksFactory;
using afp::peaks::make_default_peaks_factory;

TEST(PeaksFactory, Creates_Finder_And_Find_Works) {
  auto fac = make_default_peaks_factory();
  ASSERT_TRUE(!!fac);

  auto finder = fac->create_peak_finder();
  ASSERT_TRUE(!!finder);

  auto S = testspec::make_spec(/*T*/1, /*F*/8, 0.f, 1000, 1000);
  testspec::set_bin(S, 0, 3, 9.0f);

  auto pp = testspec::pp_default();
  pp.neighborhood_time_frames = 0;
  pp.neighborhood_freq_bins = 1;
  pp.threshold_db = 0.f;

  auto r = finder->find(S, pp);
  ASSERT_TRUE(r.has_value());
  ASSERT_EQ(r->size(), 1u);
  EXPECT_EQ((*r)[0].t, 0u);
  EXPECT_EQ((*r)[0].f, 3u);
  EXPECT_NEAR((*r)[0].strength_db, 9.0f, 1e-7f);
}
