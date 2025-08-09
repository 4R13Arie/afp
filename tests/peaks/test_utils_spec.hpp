#pragma once
#include <cstdint>
#include <vector>
#include <span>
#include <algorithm>
#include <cmath>

#include <kfr/all.hpp>
#include "afp/util/util.hpp"
#include "afp/peaks/peaks.hpp"

namespace testspec {
// ---------- Builders ----------

inline afp::util::Spectrogram make_spec(uint32_t frames, uint32_t bins,
                                        float fill_db = 0.f,
                                        uint32_t sr = 48000,
                                        uint32_t hop = 480) {
  afp::util::Spectrogram S{};
  S.num_frames = frames;
  S.num_bins = static_cast<std::uint16_t>(bins);
  S.sample_rate_hz = sr;
  S.hop_size = hop;
  S.fft_size = 0; // not used by the peak finder
  S.log_mag.resize(static_cast<size_t>(frames) * bins, fill_db);
  return S;
}

inline void set_bin(afp::util::Spectrogram& S, uint32_t t, uint32_t f,
                    float db) {
  const size_t idx = static_cast<size_t>(t) * S.num_bins + f;
  S.log_mag[idx] = db;
}

inline std::span<const float> frame_row(const afp::util::Spectrogram& S,
                                        uint32_t t) {
  const size_t off = static_cast<size_t>(t) * S.num_bins;
  return std::span<const float>(S.log_mag.data() + off, S.num_bins);
}

// ---------- Analysis helpers ----------

inline bool sorted_tf(const std::vector<afp::util::Peak>& v) {
  for (size_t i = 1; i < v.size(); ++i) {
    const auto& a = v[i - 1];
    const auto& b = v[i];
    if (a.t > b.t) return false;
    if (a.t == b.t && a.f >= b.f) return false;
  }
  return true;
}

inline bool all_within_bounds(const std::vector<afp::util::Peak>& v,
                              uint32_t T, uint32_t F) {
  for (const auto& p : v) {
    if (!(p.t < T && p.f < F)) return false;
  }
  return true;
}

// ---------- Params ----------

inline afp::peaks::PeakParams pp_default() {
  return {};
}

// Choose sr=1000, hop=1000 so fps=1 and per-second densities == per-frame.
inline void set_density(afp::peaks::PeakParams& pp,
                        uint32_t min_per_sec, uint32_t max_per_sec) {
  pp.target_peak_density_per_sec_min = min_per_sec;
  pp.target_peak_density_per_sec_max = max_per_sec;
}
} // namespace testspec
