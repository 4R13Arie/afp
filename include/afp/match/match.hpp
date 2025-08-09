#pragma once
#include <cstdint>
#include <memory>
#include <span>

#include "afp/util/util.hpp"
#include "afp/index/index.hpp"

namespace afp::match_ {
/**
 * Matching configuration.
 * Units:
 *  - top_k: candidates returned
 *  - offset_bin_ms: histogram bin width for offsets (ms)
 *  - delta_t_tolerance_ms: tolerance for Δt during tolerant key expansion (ms)
 *  - freq_tolerance_bins: tolerance for fa/ft (bins)
 *  - min_inliers: minimum votes in best offset bin to accept a candidate
 *  - min_margin_ratio: require top1.score >= ratio * top2.score (if top2 exists)
 *  - min_span_ms: minimum temporal span (ms) of inliers in the best cluster
 */
struct MatchParams {
  std::uint32_t top_k{5};
  std::uint32_t offset_bin_ms{10}; // ms per offset histogram bin
  std::uint16_t delta_t_tolerance_ms{25}; // ms tolerance for Δt
  std::uint16_t freq_tolerance_bins{1}; // bins for fa/ft
  std::uint32_t min_inliers{20};
  float min_margin_ratio{1.8f}; // top1 >= ratio * top2
  std::uint32_t min_span_ms{1500};
};

/**
 * Shazam-style matcher: tolerant key expansion, inverted index lookups,
 * per-track offset voting, top-K selection.
 *
 * Thread-safety: YES (stateless; all temporaries are local).
 * Invariants: assumes the same key packing/quantization as the indexed corpus.
 */
class IMatcher {
public:
  virtual ~IMatcher() = default;

  /**
   * Purpose:
   *   Identify the best-matching track(s) by:
   *     (1) expanding each query pair into tolerant keys (±freq, ±Δt),
   *     (2) batch lookups in the inverted index,
   *     (3) voting offsets (posting.t_anchor_ms − query.t_anchor_ms) into
   *         per-track histograms (bin width = offset_bin_ms),
   *     (4) extracting top-K by inlier count (and span as tiebreaker).
   *
   * Preconditions:
   *   - query_pairs were built with the SAME pipeline & key quantization as the index.
   *   - index.open() has been called successfully.
   *
   * Postconditions:
   *   - Returns MatchResult.topk containing up to top_k candidates,
   *     each with { track, offset_ms (bin center), inliers, score, span_ms }.
   *   - Sorted by score descending. Candidates that do not meet
   *     {min_inliers, min_span_ms} are filtered out.
   *   - If a second-best exists and top1.score < min_margin_ratio * top2.score,
   *     the result MAY be empty (no clear winner).
   *
   * Units:
   *   - Offsets & spans in ms. Frequencies in bins. Δt in ms (quantized in key).
   *
   * Complexity:
   *   - Let Q = |query_pairs|.
   *   - Key expansion per pair ~ O((2Bf+1)^2 * (2Bt+1)) where
   *       Bf = freq_tolerance_bins, Bt ≈ ceil(delta_t_tolerance_ms / Δt_step_ms).
   *   - Lookups are batched: O(U) gets where U is #unique expanded keys.
   *   - Voting: O(total_postings_returned).
   *
   * Thread-safety: YES (const; no shared mutable state).
   */
  virtual afp::util::Expected<afp::util::MatchResult>
  match(std::span<const afp::util::FingerprintPair> query_pairs,
        const afp::index::IIndexReader& index,
        const MatchParams& mp) const = 0;
};

/** Factory for matchers (stateless). */
class IMatchFactory {
public:
  virtual ~IMatchFactory() = default;

  virtual std::unique_ptr<IMatcher> create_matcher() = 0;
};

/** Default matcher factory (11/11/10 key packing with Δt step = 2.5 ms). */
std::unique_ptr<IMatchFactory> make_default_match_factory();
} // namespace afp::match_
