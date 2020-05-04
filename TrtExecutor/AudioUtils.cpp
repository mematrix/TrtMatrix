#include "AudioUtils.h"

#include <algorithm>
#include <limits>

#include "AudioFFT.h"


nc::NdArray<double> utils::stft(const nc::NdArray<double> &frame, audiofft::AudioFFT &fft, uint32_t dft_size, bool zero_phase)
{
    auto size = audiofft::AudioFFT::ComplexSize(dft_size);
    nc::NdArray<double> ret(1, 2 * size);

    std::vector<float> input(dft_size);
    std::vector<float> out_re(size);
    std::vector<float> out_im(size);

    if (!zero_phase || frame.size() >= dft_size) {
        auto cp_size = std::min(dft_size, frame.size());
        std::copy_n(frame.data(), cp_size, input.data());
    } else {
        auto f_size = frame.size();
        auto offset = f_size / 2;
        std::copy_n(frame.data() + offset, frame.size() - offset, input.data());
        std::copy_n(frame.data(), offset, input.data() + dft_size - offset);
    }

    fft.fft(input.data(), out_re.data(), out_im.data());
    std::copy_n(out_re.data(), size, ret.data());
    std::copy_n(out_im.data(), size, ret.data() + size);

    return ret;
}

nc::NdArray<float> utils::istft(const nc::NdArray<float> &complex, audiofft::AudioFFT &fft)
{
    auto count = complex.size() / 2;
    auto dft_size = (count - 1) * 2;
    nc::NdArray<float> output(1, dft_size);

    fft.ifft(output.data(), complex.data(), complex.data() + count);

    return output;
}

void utils::mag_phasor(const nc::NdArray<double> &complex, nc::NdArray<double> &mag, nc::NdArray<double> &unit)
{
    auto count = complex.size() / 2;
    mag = nc::NdArray<double>(1, count);
    unit = nc::NdArray<double>(1, 2 * count);

    for (unsigned i = 0; i < count; ++i) {
        auto hypot = std::hypot(complex[i], complex[i + count]);
        mag[i] = hypot;
        if (hypot != 0.) {
            unit[i] = complex[i] / hypot;
            unit[i + count] = complex[i + count] / hypot;
        } else {
            unit[i] = 1;
            unit[i + count] = 0;
        }
    }
}

nc::NdArray<double> utils::log_pow(const nc::NdArray<double> &sig, float floor)
{
    constexpr double e = 2.718281828459045;
    auto log10e = nc::log10(e);
    auto pspec = nc::square(sig);
    auto non_zero_min = std::numeric_limits<double>::max();
    for (auto p : pspec) {
        if (p > 0 && p < non_zero_min) {
            non_zero_min = p;
        }
    }
    if (non_zero_min != std::numeric_limits<double>::max()) {
        auto zero_floor = std::log(non_zero_min) + floor / 10. / log10e;
        std::transform(pspec.begin(), pspec.end(), pspec.begin(), [zero_floor](double v) {
            return v == 0 ? zero_floor : std::log(v);
            });
    } else {
        pspec.fill(-80. / 10. / log10e);
    }

    return pspec;
}

void utils::onlineMVN_per_frame(nc::NdArray<double> &feat, int frame_count, nc::NdArray<double> &mu, nc::NdArray<double> &sigma_square,
                                double frame_shift, double tau_feat, double tau_feat_init, double t_init)
{
    constexpr double sigma_eps = 1e-12;

    auto n_init_frames = static_cast<int>(std::ceil(t_init / frame_shift));
    auto alpha_feat_init = std::exp(-frame_shift / tau_feat_init);
    auto alpha_feat = std::exp(-frame_shift / tau_feat);

    auto alpha = frame_count < n_init_frames ? alpha_feat_init : alpha_feat;

    mu *= alpha;
    mu += (1 - alpha) * feat;
    sigma_square *= alpha;
    sigma_square += (1 - alpha) * nc::square(feat);

    auto sigma_raw = sigma_square - nc::square(mu);
    for (auto &s : sigma_raw) {
        if (s < sigma_eps) {
            s = sigma_eps;
        }
    }
    auto sigma = nc::sqrt(sigma_raw);

    feat -= mu;
    feat /= sigma;
}

static nc::NdArray<double> hamming_window(int size, int wind_size = -1)
{
    constexpr double pi = 3.141592653589793;

    if (wind_size < size) {
        wind_size = size;
    }

    if (wind_size < 1 || size < 1) {
        return nc::NdArray<double>();
    }
    if (wind_size == 1) {
        return nc::ones<double>(1, 1);
    }

    nc::NdArray<double> wind(1, size);
    for (int i = 0; i < size; ++i) {
        wind[i] = 0.54 - 0.46 * std::cos(2.0 * pi * i / (wind_size - 1));
    }
    return wind;
}

bool tcola(const nc::NdArray<double> &wind, double &amp)
{
    auto w_size = static_cast<int>(wind.size());
    auto h_size = 160;
    auto buff = wind;
    for (int wi = h_size; wi < w_size; wi += h_size) {
        for (int i = wi; i < w_size; ++i) {
            buff[i] += wind[i - wi];
        }
    }
    for (int wj = w_size - h_size; wj > 0; wj -= h_size) {
        for (int j = 0; j < wj; ++j) {
            buff[j] += wind[w_size - wj + j];
        }
    }

    auto val = buff[0];
    if (std::all_of(buff.begin(), buff.end(), [val](double v) {return std::abs(val - v) < 1e-5; })) {
        amp = val;
        return true;
    }

    return false;
}

bool tnorm(nc::NdArray<double> &wind)
{
    double amp;
    if (!tcola(wind, amp)) {
        return false;
    }

    wind /= amp;
    return true;
}

nc::NdArray<double> utils::hamming(int window_size, float hop)
{
    if (hop == 0) {
        return hamming_window(window_size);
    }

    auto wind = hamming_window(window_size, window_size + 1 - window_size % 2);
    if (window_size % 2) {
        wind[0] /= 2;
        wind[wind.size() - 1] /= 2;
    }

    auto norm = tnorm(wind);
    assert(norm && "wind_size violates COLA in time");

    return wind;
}
