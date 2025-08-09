#pragma once

#include "test_utils_signal.hpp"
#include <complex>

#include "afp/util/util.hpp"

using afp::util::PcmBuffer;

namespace teststft {
inline std::vector<float> impulse(std::size_t N) {
  std::vector<float> x(N, 0.0f);
  if (N > 0) x[0] = 1.0f;
  return x;
}

inline std::vector<float> step(std::size_t N) {
  return std::vector<float>(N, 1.0f);
}

inline std::vector<float> constant(std::size_t N, float v) {
  return std::vector<float>(N, v);
}

inline std::vector<float> sine(std::size_t N, std::uint32_t sr, float f) {
  std::vector<float> x(N, 0.0f);
  const float two_pi_over_sr = 2.0f * float(M_PI) / float(sr);
  for (std::size_t n = 0; n < N; ++n) {
    x[n] = std::sin(two_pi_over_sr * float(n) * f);
  }
  return x;
}

inline PcmBuffer make_buffer(std::uint32_t sr, const std::vector<float>& x) {
  PcmBuffer b;
  b.sample_rate_hz = sr;
  b.samples.resize(x.size());                  // works for kfr::univector
  std::copy_n(x.data(), x.size(), b.samples.data());
  // or: std::memcpy(b.samples.data(), x.data(), x.size() * sizeof(float));
  return b;
}

inline float mean_(const std::vector<float>& x) {
  if (x.empty()) return 0.0f;
  double s = 0.0;
  for (float v : x) s += v;
  return static_cast<float>(s / double(x.size()));
}

inline float rms(const std::vector<float>& x) {
  if (x.empty()) return 0.0f;
  double s = 0.0;
  for (float v : x) s += double(v) * double(v);
  return static_cast<float>(std::sqrt(s / double(x.size())));
}

inline float max_abs(const std::vector<float>& x) {
  float m = 0.0f;
  for (float v : x) m = std::max(m, std::fabs(v));
  return m;
}

inline std::vector<float> mag_spectrum_real(const std::vector<float>& x) {
  // Naive DFT (N^2) is fine for small N in unit tests.
  const std::size_t N = x.size();
  const std::size_t K = N / 2 + 1;
  std::vector<float> mag(K, 0.0f);

  for (std::size_t k = 0; k < K; ++k) {
    std::complex<double> acc(0.0, 0.0);
    const double ang = -2.0 * M_PI * double(k) / double(N);
    for (std::size_t n = 0; n < N; ++n) {
      double phase = ang * double(n);
      acc += std::complex<double>(std::cos(phase), std::sin(phase)) * double(x[n]);
    }
    mag[k] = static_cast<float>(std::abs(acc));
  }
  return mag;
}

inline std::size_t dominant_bin(const std::vector<float>& mag) {
  if (mag.empty()) return 0;
  return static_cast<std::size_t>(std::distance(
      mag.begin(), std::max_element(mag.begin(), mag.end())));
}

inline afp::util::PcmSpan span_of(const afp::util::PcmBuffer& b) {
  return afp::util::PcmSpan{b.sample_rate_hz, b.samples};
}

inline void copy_to_univector(const std::vector<float>& src, kfr::univector<float>& dst) {
  dst.resize(src.size());
  if (!src.empty()) std::copy_n(src.data(), src.size(), dst.data());
}

} // namespace testutils
