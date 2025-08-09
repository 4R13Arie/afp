#pragma once
#include <vector>
#include <span>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <kfr/all.hpp>
#include <kfr/dft.hpp>

#include "afp/util/util.hpp"

namespace testdsp {
// -----------------------------
// Basic signal builders
// -----------------------------
inline std::vector<float> impulse(size_t N) {
  std::vector<float> x(N, 0.0f);
  if (N) x[0] = 1.0f;
  return x;
}

inline std::vector<float> step(size_t N, float v = 1.0f) {
  return std::vector<float>(N, v);
}

inline std::vector<float> dc(size_t N, float v) {
  return std::vector<float>(N, v);
}

inline std::vector<float> sine(size_t N, float sr, float f, float amp = 1.0f,
                               float phase = 0.0f) {
  std::vector<float> x(N);
  const float w = 2.0f * float(M_PI) * (f / sr);
  for (size_t n = 0; n < N; ++n) x[n] = amp * std::sin(phase + w * float(n));
  return x;
}

// -----------------------------
// Pcm helpers
// -----------------------------
inline afp::util::PcmBuffer make_buffer(afp::util::SampleRateHz sr,
                                        const std::vector<float>& x) {
  afp::util::PcmBuffer b;
  b.sample_rate_hz = sr;
  b.samples = kfr::univector<float>(x.begin(), x.end());
  return b;
}

inline afp::util::PcmSpan span_of(const afp::util::PcmBuffer& b) {
  return {b.sample_rate_hz,
          std::span<const float>(b.samples.data(), b.samples.size())};
}

// -----------------------------
// Stats helpers
// -----------------------------
inline float mean(std::span<const float> x) {
  if (x.empty()) return 0.0f;
  double s = 0.0;
  for (float v : x) s += v;
  return float(s / double(x.size()));
}

inline float rms(std::span<const float> x) {
  if (x.empty()) return 0.0f;
  double s = 0.0;
  for (float v : x) s += double(v) * double(v);
  return float(std::sqrt(s / double(x.size())));
}

inline float max_abs(std::span<const float> x) {
  float m = 0.0f;
  for (float v : x) m = std::max(m, std::abs(v));
  return m;
}

inline float snr_db(std::span<const float> ref, std::span<const float> test) {
  const size_t N = std::min(ref.size(), test.size());
  if (N == 0) return 0.0f;
  double se = 0.0, ss = 0.0;
  for (size_t i = 0; i < N; ++i) {
    const double e = double(test[i]) - double(ref[i]);
    se += e * e;
    ss += double(ref[i]) * double(ref[i]);
  }
  if (se <= 1e-30) return 120.0f;
  if (ss <= 1e-30) return -120.0f;
  return float(10.0 * std::log10(ss / se));
}

// -----------------------------
// Spectrum helpers (real DFT using KFR)
// Returns magnitude spectrum of length N/2+1 for real x
// -----------------------------
inline std::vector<float> mag_spectrum(std::span<const float> x) {
  const size_t N = x.size();
  if (N == 0) return {};

  // Real DFT plan
  kfr::dft_plan_real<float> plan(N);

  // Real→complex output has N/2 + 1 bins
  const size_t M = N / 2 + 1;
  kfr::univector<kfr::complex<float>> cpx(M);

  // Provide input to plan: use a non-owning ref (or allocate if your KFR needs it)
  // Preferred (zero-copy):
  kfr::univector_ref<const float> in_ref(x.data(), x.size());
  kfr::univector<kfr::u8> temp(plan.temp_size);
  plan.execute(cpx.data(), in_ref.data(), temp.data());

  // If your KFR build doesn’t accept univector_ref for input, fall back to:
  // kfr::univector<float> in_uv(x.begin(), x.end());
  // plan.execute(cpx, in_uv);

  std::vector<float> mag(M);
  for (size_t i = 0; i < M; ++i) {
    const auto re = cpx[i].real();
    const auto im = cpx[i].imag();
    mag[i] = std::sqrt(re * re + im * im);
  }
  return mag;
}

inline float dominant_bin_freq(std::span<const float> x, float sr) {
  const size_t N = x.size();
  if (N == 0) return 0.0f;
  auto mag = mag_spectrum(x);
  auto it = std::max_element(mag.begin(), mag.end());
  const size_t k = size_t(std::distance(mag.begin(), it));
  const float bin_hz = sr * float(k) / float(N);
  return bin_hz;
}
} // namespace testdsp