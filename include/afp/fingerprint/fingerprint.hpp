#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <kfr/all.hpp>
#include "afp/util/util.hpp"

namespace afp::fingerprint {
/**
 * Pairing configuration.
 * Units:
 *  - delta_min_ms / delta_max_ms: milliseconds between anchor and target
 *  - max_targets_per_anchor: count (0 = disable pairing)
 *  - freq_tolerance_bins / time_tolerance_frames: bins/frames (used later at match-time)
 *  - sample_rate_hz: Hz; hop_size: samples (for frame→ms mapping)
 */
struct PairingParams {
  afp::util::TimeMs delta_min_ms{250};
  afp::util::TimeMs delta_max_ms{2000};
  std::uint8_t max_targets_per_anchor{4};
  std::uint16_t freq_tolerance_bins{1};
  std::uint16_t time_tolerance_frames{1};
  afp::util::SampleRateHz sample_rate_hz{};
  std::uint32_t hop_size{}; // samples
};

/**
 * Build anchor→target pairs from peaks.
 * Thread-safety: YES (stateless; no shared state).
 */
class IFingerprintBuilder {
public:
  virtual ~IFingerprintBuilder() = default;

  /**
   * Purpose:
   *   For each peak (anchor), choose up to `max_targets_per_anchor` *later* peaks
   *   whose time difference Δt is in [delta_min_ms, delta_max_ms]. Produce
   *   (f_anchor, f_target, Δt_ms, t_anchor_ms) pairs sorted by t_anchor_ms.
   *
   * Preconditions:
   *   - `peaks` refers to a band-limited constellation, **sorted by (t, f)**.
   *   - `pp.sample_rate_hz > 0`, `pp.hop_size > 0`, `pp.max_targets_per_anchor >= 0`.
   *   - `pp.delta_min_ms <= pp.delta_max_ms`.
   *
   * Postconditions:
   *   - Returns pairs with non-decreasing `t_anchor_ms`.
   *   - For each pair: `delta_ms ∈ [delta_min_ms, delta_max_ms]` and `target.t > anchor.t`.
   *
   * Units:
   *   - `t_anchor_ms`, `delta_ms` in milliseconds.
   *   - `f_anchor`, `f_target` are frequency bins (0-based).
   *
   * Complexity:
   *   - Average **O(P + E)** with a sliding window over peaks, where
   *     P = number of peaks, E = emitted pairs (capped by max_targets_per_anchor).
   *
   * Thread-safety:
   *   - YES (pure function; all temporaries local).
   */
  virtual afp::util::Expected<std::vector<afp::util::FingerprintPair>>
  make_pairs(std::span<const afp::util::Peak> peaks,
             const PairingParams& pp) = 0;
};

/**
 * Pack pairs into 32-bit keys: (f_a<<23) | (f_t<<14) | (delta_ms).
 * Thread-safety: YES (stateless).
 */
class IKeyPacker {
public:
  virtual ~IKeyPacker() = default;

  /**
   * Purpose:
   *   Convert (f_anchor, f_target, delta_ms) to 32-bit keys for the inverted index.
   *
   * Preconditions:
   *   - `delta_ms <= 16383` (14 bits).
   *   - `f_anchor <= 511` and `f_target <= 511` (9 bits each).
   *
   * Postconditions:
   *   - Returns a vector of `FingerprintKey` with `.value` packed as above.
   *   - Pairs that do not fit the ranges are **skipped** (not an error).
   *
   * Units: milliseconds for `delta_ms`; bins for `f_*`.
   *
   * Complexity: O(N) over pair count.
   *
   * Thread-safety: YES (pure function; no shared state).
   */
  virtual afp::util::Expected<std::vector<afp::util::FingerprintKey>>
  pack(std::span<const afp::util::FingerprintPair> pairs) = 0;
};

/** Factory for fingerprint components. */
class IFingerprintFactory {
public:
  virtual ~IFingerprintFactory() = default;

  virtual std::unique_ptr<IFingerprintBuilder> create_builder() = 0;

  virtual std::unique_ptr<IKeyPacker> create_packer() = 0;
};

/** Default KFR-based factory. */
std::unique_ptr<IFingerprintFactory> make_default_fingerprint_factory();
} // namespace afp::fingerprint
