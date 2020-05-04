// TrtExecutor.h: Inference TensorRT model
//

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <atomic>

#include <NvInfer.h>

#include "common/common.h"


class TrtInputStream
{
public:
    virtual ~TrtInputStream() = default;

    virtual Dims GetDynamicDim(const char *input_name) = 0;

    virtual std::vector<std::string> GetInputTensorNames(const nvinfer1::ICudaEngine &engine) = 0;

    virtual bool TryTake(const std::vector<void *> &host_buffer, const std::vector<size_t> &sizes) = 0;
};

class TrtOutputHandler
{
public:
    virtual ~TrtOutputHandler() = default;

    virtual std::vector<std::string> GetOutputTensorNames(const nvinfer1::ICudaEngine &engine) = 0;

    virtual void SetTensorDim(const char *output_name, const Dims &dims) = 0;

    virtual void Consume(const std::vector<void *> &host_buffer, const std::vector<size_t> &sizes) = 0;
};

struct TrtExecuteConfig
{
    std::string model_path;
};

class TrtExecutor
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    explicit TrtExecutor(const TrtExecuteConfig &config);

    void SetInputStream(const std::shared_ptr<TrtInputStream> &input);

    void SetOutputHandler(const std::shared_ptr<TrtOutputHandler> &output);

    void Process();

    void Terminate();

private:
    TrtExecuteConfig config_;

    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<TrtInputStream> input_;
    std::shared_ptr<TrtOutputHandler> output_;

    std::atomic<bool> terminate_;
};
