#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include <kfr/all.hpp>
#include "afp/util/util.hpp"

namespace afp::peaks {
  /**
   * Peak detection parameters for a Shazam-style constellation map.
   * Units:
   *  - neighborhood_*: frames/bins (integer radii)
   *  - threshold_db: dB above the per-frame median
   *  - min_freq_separation_bins: bins (enforced via per-frame NMS)
   *  - target_peak_density_per_sec_*: peaks/second (controls per-frame caps)
   */
  struct PeakParams {
    std::uint16_t neighborhood_time_frames{3}; // ±3 frames
    std::uint16_t neighborhood_freq_bins{15}; // ±15 bins
    float threshold_db{9.0f}; // above per-frame median (dB)
    std::uint16_t min_freq_separation_bins{5}; // NMS separation in bins
    std::uint32_t target_peak_density_per_sec_min{30};
    std::uint32_t target_peak_density_per_sec_max{60};
  };

  /**
   * Detect local maxima in a band-limited log spectrogram with adaptive thresholds,
   * non-maximum suppression (NMS), and target peak density constraints.
   *
   * Thread-safety: YES (stateless; all temporaries are local).
   */
  class IPeakFinder {
  public:
    virtual ~IPeakFinder() = default;

    /**
     * Purpose:
     *   Produce a sorted (by time then frequency) constellation map:
     *   local maxima within a rectangular neighborhood, filtered by an
     *   adaptive per-frame threshold (median + threshold_db), then pruned
     *   via per-frame NMS with a minimum frequency separation and density capping.
     *
     * Preconditions:
     *   - S.num_frames >= 0, S.num_bins > 0
     *   - S.log_mag.size() == S.num_frames * S.num_bins
     *   - S.sample_rate_hz > 0, S.hop_size > 0 (for density per second calc)
     *
     * Postconditions:
     *   - Returns peaks sorted by (t, f)
     *   - Each peak.t ∈ [0, S.num_frames-1], peak.f ∈ [0, S.num_bins-1]
     *   - peak.strength_db is the windowed/normalized dB from S
     *
     * Units:
     *   - Spectrogram in dB; strengths reported in dB
     *   - Time index is frames; frequency is bins
     *
     * Complexity:
     *   - O(T * F * W) with W ≈ (2·neighborhood_time+1)·(2·neighborhood_freq+1)
     *   - Plus O(K log K) per frame for NMS sorting, K = candidates in frame
     *
     * Thread-safety:
     *   - YES (no shared state).
     *
     * Decoder formats:
     *   - N/A (no audio container/codec in this module).
     */
    virtual afp::util::Expected<std::vector<afp::util::Peak> >
    find(const afp::util::Spectrogram &S, const PeakParams &pp) = 0;
  };

  /** Factory for peak-finder instances. */
  class IPeaksFactory {
  public:
    virtual ~IPeaksFactory() = default;

    virtual std::unique_ptr<IPeakFinder> create_peak_finder() = 0;
  };

  /** Default factory (stateless KFR-based implementation). */
  std::unique_ptr<IPeaksFactory> make_default_peaks_factory();
} // namespace afp::peaks
