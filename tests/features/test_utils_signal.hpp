#pragma once
#include <cstdint>
#include <vector>
#include <functional>
#include <span>
#include <cmath>
#include <kfr/all.hpp>

#include "afp/util/util.hpp"
#include "afp/features/features.hpp"

namespace testfeat {
// ---------- Builders ----------

inline afp::util::ComplexSpectra make_spectra(uint32_t num_frames,
                                              uint32_t num_bins) {
  afp::util::ComplexSpectra s{};
  s.num_frames = num_frames;
  s.num_bins = num_bins;
  s.bins.resize(static_cast<size_t>(num_frames) * num_bins,
                kfr::complex<float>(0.f, 0.f));
  return s;
}

inline void set_bin(afp::util::ComplexSpectra& s, uint32_t frame, uint32_t bin,
                    float re, float im = 0.f) {
  const size_t idx = static_cast<size_t>(frame) * s.num_bins + bin;
  s.bins[idx] = kfr::complex<float>(re, im);
}

inline void fill_frame(afp::util::ComplexSpectra& s,
                       uint32_t frame,
                       const std::function<kfr::complex<float>(uint32_t bin)>&
                       gen) {
  for (uint32_t b = 0; b < s.num_bins; ++b) {
    set_bin(s, frame, b, gen(b).real(), gen(b).imag());
  }
}

inline afp::features::LogSpecParams make_params(uint32_t sr,
                                                uint32_t hop,
                                                uint32_t fft,
                                                float low_hz,
                                                float high_hz,
                                                float eps = 1e-8f,
                                                float clip_db = 60.f,
                                                bool median = true,
                                                uint16_t whiten = 0) {
  afp::features::LogSpecParams p{};
  p.sample_rate_hz = sr;
  p.hop_size = hop;
  p.fft_size = fft;
  p.band_low_hz = low_hz;
  p.band_high_hz = high_hz;
  p.epsilon = eps;
  p.clip_db = clip_db;
  p.per_frame_median_subtract = median;
  p.whiten_radius_bins = whiten;
  return p;
}

// ---------- Analysis ----------

inline float mean(std::span<const float> v) {
  if (v.empty()) return 0.f;
  double s = 0.0;
  for (float x : v) s += x;
  return static_cast<float>(s / double(v.size()));
}

inline float max_abs(const std::vector<float>& v) {
  float m = 0.f;
  for (float x : v) m = std::max(m, std::abs(x));
  return m;
}

inline std::pair<float, float> minmax(std::span<const float> v) {
  if (v.empty()) return {0.f, 0.f};
  float lo = v[0], hi = v[0];
  for (float x : v) {
    lo = std::min(lo, x);
    hi = std::max(hi, x);
  }
  return {lo, hi};
}

// ---------- Layout helper ----------

inline size_t idx_log(size_t frame, size_t bin, size_t out_bins) {
  return frame * out_bins + bin;
}

// ---------- Math helpers for tests ----------

inline float db_mag(float mag, float eps) {
  const float v = std::max(mag, 0.0f) + eps;
  return 20.f * std::log10(v);
}

inline float hz_of_bin(uint32_t bin, uint32_t sr, uint32_t fft) {
  // Inverse of internal rounding; for integer bin selection in tests.
  return (float(bin) * float(sr)) / float(fft);
}
} // namespace testfeat
