#pragma once

#include "NumCpp.hpp"
#include "AudioFFT.h"


namespace utils {

nc::NdArray<double> stft(const nc::NdArray<double> &frame, audiofft::AudioFFT &fft, uint32_t dft_size, bool zero_phase = false);

nc::NdArray<float> istft(const nc::NdArray<float> &complex, audiofft::AudioFFT &fft);

void mag_phasor(const nc::NdArray<double> &complex, nc::NdArray<double> &mag, nc::NdArray<double> &unit);

nc::NdArray<double> log_pow(const nc::NdArray<double> &sig, float floor = -30.0f);

void onlineMVN_per_frame(nc::NdArray<double> &feat, int frame_count, nc::NdArray<double> &mu,
                         nc::NdArray<double> &sigma_square, double frame_shift = 0.01, double tau_feat = 3,
                         double tau_feat_init = 0.1, double t_init = 0.1);

nc::NdArray<double> hamming(int window_size, float hop = 0.0f);

}
