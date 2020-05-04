#include "TrtExecutor.h"

#include <iostream>
#include <cmath>

#include <sndfile.hh>

#include "AudioUtils.h"


struct VoiceFileInputConfig
{
    float window_len = 0.02f;
    float hot_fraction = 0.5f;
    int dft_size = 512;
    float spectral_floor = -120.0f;
    float time_signal_floor = 1e-12f;
};

class LocalFileInputStream : public TrtInputStream
{
public:
    LocalFileInputStream(const VoiceFileInputConfig &config, const std::string &path, TrtExecutor *executor)
        : config_(config),
          executor_(executor),
          snd_file_(path)
    {
        if (!snd_file_) {
            return;
        }

        frames_ = snd_file_.frames();
        sampling_rate_ = snd_file_.samplerate();
        channels_ = snd_file_.channels();
        format_ = snd_file_.format();
        std::cout << "[SndFile]: frames: " << frames_ << ", sample_rate: " << sampling_rate_;
        std::cout << ", channels: " << channels_ << ", format: " << format_ << std::endl;

        fft_.init(config_.dft_size);

        int frame_size = static_cast<int>(config_.window_len * sampling_rate_);
        wind_ = utils::hamming(frame_size, config_.hot_fraction);

        int s_size = channels_ * frames_;
        int f_size = wind_.size();
        int h_size = static_cast<int>(config_.hot_fraction * frame_size);
        hot_fraction_size_ = h_size;

        int s_start = h_size - f_size;
        int s_end = s_size;
        int n_frame = static_cast<int>(std::ceil(1.0 * (s_end - s_start) / h_size));
        frame_count_ = n_frame;

        int zp_left = -s_start;
        int zp_right = (n_frame - 1) * h_size + f_size - zp_left - s_size;

        sig_pad_ = nc::zeros<double>(1, s_size + zp_left + zp_right);
        double *sig_ptr = sig_pad_.data() + zp_left;
        const auto read_cnt = snd_file_.read(sig_ptr, s_size);
        assert(static_cast<int>(read_cnt) == s_size);
    }

    Dims GetDynamicDim(const char *input_name) override
    {
        std::cout << "Info: Get dim for " << input_name << std::endl;
        return Dims3{1, 1, 257};
    }

    std::vector<std::string> GetInputTensorNames(const nvinfer1::ICudaEngine &engine) override
    {
        std::vector<std::string> ret;
        ret.emplace_back("input");
        for (int i = 0; i < engine.getNbBindings(); ++i) {
            if (engine.bindingIsInput(i) && strcmp(engine.getBindingName(i), "input") != 0) {
                ret.emplace_back(engine.getBindingName(i));
            }
        }
        return ret;
    }

    bool TryTake(const std::vector<void *> &host_buffer, const std::vector<size_t> &sizes) override
    {
        ++cur_frame_;
        if (cur_frame_ >= frame_count_) {
            executor_->Terminate();
            return false;
        }
        assert(host_buffer.size() == sizes.size());
        if (cur_frame_ == 0) {
            for (unsigned i = 1; i < host_buffer.size(); ++i) {
                memset(host_buffer[i], 0, sizes[i]);
            }
        }

        input_buffer_ = host_buffer;
        sizes_ = sizes;

        // cur frame data
        int frame_start = cur_frame_ * hot_fraction_size_;
        int frame_end = frame_start + wind_.size();
        auto frame_sig_pad = sig_pad_(sig_pad_.rSlice(), nc::Slice(frame_start, frame_end)) * wind_;

        utils::mag_phasor(utils::stft(frame_sig_pad, fft_, config_.dft_size, false), x_mag_, x_phs_);

        auto feat = utils::log_pow(x_mag_, config_.spectral_floor);
        if (cur_frame_ == 0) {
            mu_ = feat;
            sigma_square_ = nc::square(feat);
        }
        utils::onlineMVN_per_frame(feat, cur_frame_, mu_, sigma_square_);

        assert(feat.size() * sizeof(float) == sizes[0]);
        auto *input = static_cast<float *>(host_buffer[0]);
        std::copy_n(feat.data(), feat.size(), input);

        return true;
    }

    void MergeOutput(const std::vector<void *> &output, const std::vector<size_t> &sizes)
    {
        assert(input_buffer_.size() == output.size());
        assert(sizes_ == sizes);

        for (int i = 1; i < input_buffer_.size(); ++i) {
            memcpy(input_buffer_[i], output[i], sizes[i]);
        }
    }

    const VoiceFileInputConfig &VoiceConfig() const
    {
        return config_;
    }

    const SndfileHandle &SndFile() const
    {
        return snd_file_;
    }

    audiofft::AudioFFT &AudioFFT()
    {
        return fft_;
    }

    const nc::NdArray<double> &Wind() const
    {
        return wind_;
    }

    int HotFractionSize() const
    {
        return hot_fraction_size_;
    }

    int FrameCount() const
    {
        return frame_count_;
    }

    int CurFrame() const
    {
        return cur_frame_;
    }

    const nc::NdArray<double> &XMag() const
    {
        return x_mag_;
    }

    const nc::NdArray<double> &XPhs() const
    {
        return x_phs_;
    }

private:
    VoiceFileInputConfig config_;
    TrtExecutor *executor_;

    SndfileHandle snd_file_;
    sf_count_t frames_ = 0;
    int sampling_rate_ = 0;
    int channels_ = 0;
    int format_ = 0;

    audiofft::AudioFFT fft_;

    int hot_fraction_size_ = 0;
    int frame_count_ = 0;
    nc::NdArray<double> wind_;
    nc::NdArray<double> sig_pad_;

    int cur_frame_ = -1;
    std::vector<void *> input_buffer_;
    std::vector<size_t> sizes_;

    nc::NdArray<double> x_mag_;
    nc::NdArray<double> x_phs_;
    nc::NdArray<double> mu_;
    nc::NdArray<double> sigma_square_;
};

class LocalFileOutputHandler : public TrtOutputHandler
{
public:
    LocalFileOutputHandler(std::shared_ptr<LocalFileInputStream> input, const std::string &path)
        : input_(std::move(input)),
          save_path_(path),
          out_(nc::zeros<float>(1, input_->FrameCount() * input_->HotFractionSize())),
          old_(nc::zeros<float>(1, input_->HotFractionSize()))
    {
        //
    }

    ~LocalFileOutputHandler() noexcept
    {
        auto &in_handle = input_->SndFile();
        SndfileHandle snd_file(save_path_, SFM_WRITE, in_handle.format(), in_handle.channels(), in_handle.samplerate());
        if (!snd_file) {
            return;
        }

        snd_file.write(out_.data(), out_.size());
    }

    std::vector<std::string> GetOutputTensorNames(const nvinfer1::ICudaEngine &engine) override
    {
        std::vector<std::string> ret;
        ret.emplace_back("output");
        for (int i = 0; i < engine.getNbBindings(); ++i) {
            if (!engine.bindingIsInput(i) && strcmp(engine.getBindingName(i), "output") != 0) {
                ret.emplace_back(engine.getBindingName(i));
            }
        }
        return ret;
    }

    void SetTensorDim(const char *output_name, const Dims &dims) override
    {
        //
    }

    void Consume(const std::vector<void *> &host_buffer, const std::vector<size_t> &sizes) override
    {
        assert(host_buffer.size() == sizes.size());
        input_->MergeOutput(host_buffer, sizes);

        auto &x_mag = input_->XMag();
        auto &x_phs = input_->XPhs();
        auto *output = static_cast<float *>(host_buffer[0]);
        auto len = sizes[0] / sizeof(float);
        nc::NdArray<float> mask(1, len * 2);
        for (unsigned i = 0; i < len; ++i) {
            auto t = x_mag[i] * output[i];
            mask[i] = t * x_phs[i];
            mask[i + len] = t * x_phs[i + len];
        }

        auto x_enh = utils::istft(mask, input_->AudioFFT());
        int cur_frame = input_->CurFrame();
        int h_size = input_->HotFractionSize();
        int frame_start = cur_frame * h_size;
        std::transform(x_enh.data(), x_enh.data() + h_size, old_.data(), x_enh.data(), std::plus<float>());
        memcpy(out_.data() + frame_start, x_enh.data(), sizeof(float) * h_size);
        memcpy(old_.data(), x_enh.data() + h_size, sizeof(float) * h_size);
    }

private:
    std::shared_ptr<LocalFileInputStream> input_;
    std::string save_path_;

    nc::NdArray<float> out_;
    nc::NdArray<float> old_;
};

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " TensorRT-model-file Src-voice-file Enhanced-save-file" << std::endl;
        return -1;
    }

    TrtExecuteConfig config;
    config.model_path = argv[1];
    TrtExecutor executor(config);

    VoiceFileInputConfig voice_config;
    auto input_stream = std::make_shared<LocalFileInputStream>(voice_config, argv[2], &executor);
    auto output_handler = std::make_shared<LocalFileOutputHandler>(input_stream, argv[3]);
    executor.SetInputStream(input_stream);
    executor.SetOutputHandler(output_handler);
    executor.Process();

    return 0;
}
