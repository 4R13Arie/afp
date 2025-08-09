#include "afp/stft/stft.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace afp::stft {
  using afp::util::Expected;
  using afp::util::PcmBuffer;
  using afp::util::PcmSpan;
  using afp::util::FrameBlock;
  using afp::util::ComplexSpectra;
  using afp::util::UtilError;

  //==============================
  // Helpers
  //==============================

  namespace {
    inline bool valid_framer(const FramerConfig &c) noexcept {
      return c.frame_size > 0 && c.hop_size > 0 && c.hop_size <= c.frame_size;
    }

    inline std::uint32_t compute_num_frames(std::size_t N, const FramerConfig &cfg) noexcept {
      if (N == 0) return 0;
      if (N < cfg.frame_size) return cfg.pad_end ? 1u : 0u;
      const std::size_t usable = N - cfg.frame_size;
      const std::size_t steps = usable / cfg.hop_size;
      const std::size_t exact = steps + 1;
      if (!cfg.pad_end) return static_cast<std::uint32_t>(exact);
      const bool needs_tail = (usable % cfg.hop_size) != 0;
      return static_cast<std::uint32_t>(exact + (needs_tail ? 1 : 0));
    }

    inline void make_hann_window(kfr::univector<float> &w) {
      const std::size_t N = w.size();
      if (N == 0) return;
      // periodic Hann (analysis): 0.5 - 0.5*cos(2πn/N)
      const float invN = 1.0f / static_cast<float>(N);
      for (std::size_t n = 0; n < N; ++n) {
        w[n] = 0.5f - 0.5f * std::cos(2.0f * float(M_PI) * (static_cast<float>(n) * invN));
      }
    }
  } // namespace

  //==============================
  // Framer (KFR buffers, span input)
  //==============================

  class FramerKfr final : public IFramer {
  public:
    Expected<FrameBlock> segment(const PcmSpan &in, const FramerConfig &cfg) override {
      if (in.sample_rate_hz == 0 || !valid_framer(cfg))
        return tl::unexpected(UtilError::InvalidArgument);

      const std::size_t N = in.samples.size();
      const std::uint32_t frames = compute_num_frames(N, cfg);
      FrameBlock fb;
      fb.frame_size = cfg.frame_size;
      fb.hop_size = cfg.hop_size;
      fb.num_frames = frames;
      fb.data.resize(static_cast<std::size_t>(frames) * cfg.frame_size);

      std::size_t src = 0;
      for (std::uint32_t f = 0; f < frames; ++f) {
        const std::size_t dst_off = static_cast<std::size_t>(f) * cfg.frame_size;
        const std::size_t copy_n = std::min<std::size_t>(cfg.frame_size, (N > src ? N - src : 0));
        if (copy_n > 0) {
          std::memcpy(fb.data.data() + dst_off, in.samples.data() + src, copy_n * sizeof(float));
        }
        if (copy_n < cfg.frame_size) {
          std::memset(fb.data.data() + dst_off + copy_n, 0, (cfg.frame_size - copy_n) * sizeof(float));
        }
        src += cfg.hop_size;
      }
      return fb;
    }
  };

  //==============================
  // Window (Hann, KFR vectors)
  //==============================

  class WindowKfr final : public IWindow {
  public:
    Expected<FrameBlock> apply(const FrameBlock &frames, WindowType wt) override {
      if (frames.frame_size == 0)
        return tl::unexpected(UtilError::InvalidArgument);

      FrameBlock out;
      out.frame_size = frames.frame_size;
      out.hop_size = frames.hop_size;
      out.num_frames = frames.num_frames;
      out.data.resize(frames.data.size());

      // Precompute window
      kfr::univector<float> w(frames.frame_size);
      switch (wt) {
        case WindowType::kHann: make_hann_window(w);
          break;
        default: return tl::unexpected(UtilError::UnsupportedFormat);
      }

      // Apply per-frame (vectorized multiply)
      for (std::uint32_t f = 0; f < frames.num_frames; ++f) {
        const std::size_t off = static_cast<std::size_t>(f) * frames.frame_size;
        // KFR multiply: y = x * w
        for (std::size_t i = 0; i < frames.frame_size; ++i) {
          out.data[off + i] = frames.data[off + i] * w[i];
        }
      }
      return out;
    }
  };

  //==============================
  // FFT (KFR dft_plan_real<float>)
  //==============================

  class FftKfr final : public IFFT {
  public:
    afp::util::Expected<ComplexSpectra>
    forward_r2c(const FrameBlock &frames, const FftConfig &cfg) override {
      if (cfg.fft_size == 0 || cfg.fft_size != frames.frame_size)
        return tl::unexpected(UtilError::InvalidArgument);

      ensure_plan(cfg.fft_size);

      ComplexSpectra out;
      out.num_frames = frames.num_frames;
      out.num_bins = static_cast<std::uint16_t>(cfg.fft_size / 2 + 1);
      out.bins.resize(static_cast<std::size_t>(out.num_bins) * out.num_frames);

      // Temporary input/output per frame
      kfr::univector<float> in_v(cfg.fft_size);
      kfr::univector<kfr::complex<float> > out_v(out.num_bins);
      kfr::univector<kfr::u8> temp(plan_->temp_size);

      for (std::uint32_t f = 0; f < frames.num_frames; ++f) {
        const std::size_t off = static_cast<std::size_t>(f) * frames.frame_size;
        std::memcpy(in_v.data(), frames.data.data() + off, sizeof(float) * cfg.fft_size);

        // KFR real→complex FFT
        plan_->execute(out_v.data(), in_v.data(), temp.data());

        // Store row-major [frame][bin]
        const std::size_t dst_off = static_cast<std::size_t>(f) * out.num_bins;
        std::memcpy(out.bins.data() + dst_off, out_v.data(), sizeof(kfr::complex<float>) * out.num_bins);
      }

      return out;
    }

    void warmup(const FftConfig &cfg, std::uint32_t /*max_frames*/) override { ensure_plan(cfg.fft_size); }

  private:
    void ensure_plan(std::uint32_t n) {
      if (fft_size_ == n && plan_) return;
      plan_.reset(); // destroy old plan first
      // kfr::dft_plan_real<float> requires size at construction
      plan_ = std::make_unique<kfr::dft_plan_real<float> >(static_cast<int>(n));
      fft_size_ = n;
    }

    std::unique_ptr<kfr::dft_plan_real<float> > plan_;
    std::uint32_t fft_size_{0};
  };

  //==============================
  // STFT driver (compose)
  //==============================

  class StftDriverKfr final : public IStftDriver {
  public:
    StftDriverKfr()
      : framer_(std::make_unique<FramerKfr>()),
        window_(std::make_unique<WindowKfr>()),
        fft_(std::make_unique<FftKfr>()) {
    }

    afp::util::Expected<ComplexSpectra>
    run(const PcmSpan &in, const FramerConfig &fr, WindowType wt, const FftConfig &fftc) override {
      if (in.sample_rate_hz == 0 || fr.frame_size != fftc.fft_size)
        return tl::unexpected(UtilError::InvalidArgument);

      auto framesE = framer_->segment(in, fr);
      if (!framesE) return tl::unexpected(framesE.error());

      auto winE = window_->apply(*framesE, wt);
      if (!winE) return tl::unexpected(winE.error());

      fft_->warmup(fftc, winE->num_frames);
      auto specE = fft_->forward_r2c(*winE, fftc);
      if (!specE) return tl::unexpected(specE.error());

      return specE;
    }

  private:
    std::unique_ptr<IFramer> framer_;
    std::unique_ptr<IWindow> window_;
    std::unique_ptr<IFFT> fft_;
  };

  //==============================
  // Factory
  //==============================

  class DefaultStftFactory final : public IStftFactory {
  public:
    std::unique_ptr<IFramer> create_framer() override { return std::make_unique<FramerKfr>(); }
    std::unique_ptr<IWindow> create_window() override { return std::make_unique<WindowKfr>(); }
    std::unique_ptr<IFFT> create_fft() override { return std::make_unique<FftKfr>(); }
    std::unique_ptr<IStftDriver> create_driver() override { return std::make_unique<StftDriverKfr>(); }
  };

  std::unique_ptr<IStftFactory> make_default_stft_factory() {
    return std::make_unique<DefaultStftFactory>();
  }
} // namespace afp::stft
