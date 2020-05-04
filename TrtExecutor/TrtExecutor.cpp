// TrtExecutor.cpp: Impl
//

#include "TrtExecutor.h"

#include <fstream>
#include <thread>

#include "common/buffers.h"
#include "common/logger.h"


TrtExecutor::TrtExecutor(const TrtExecuteConfig &config) : config_(config), terminate_(false)
{
    std::ifstream model_file(config_.model_path, std::ios_base::binary);
    if (!model_file) {
        std::cerr << "Error: Unable to open model file: " << config_.model_path << std::endl;
        assert(false);
    }

    model_file.seekg(0, std::ios_base::end);
    const auto file_len = model_file.tellg();
    model_file.seekg(0, std::ios_base::beg);
    std::cout << "Info: model file length: " << file_len << "bytes." << std::endl;
    std::vector<char> file_buffer(file_len);
    model_file.read(file_buffer.data(), file_len);

    auto runtime = SampleUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(file_buffer.data(), file_len), samplesCommon::InferDeleter());
    if (!engine_) {
        std::cerr << "Error: Unable to deserialize model." << std::endl;
        assert(false);
    }
}

void TrtExecutor::SetInputStream(const std::shared_ptr<TrtInputStream> &input)
{
    input_ = input;
}

void TrtExecutor::SetOutputHandler(const std::shared_ptr<TrtOutputHandler> &output)
{
    output_ = output;
}

static bool IsDynamicDim(const Dims &dims)
{
    return std::any_of(dims.d, dims.d + dims.nbDims, [](int dim) { return dim == -1; });
}

static bool ValidateBuffer(const std::vector<void *> &buffer)
{
    return std::all_of(std::begin(buffer), std::end(buffer), [](void *p) { return p != nullptr; });
}

void TrtExecutor::Process()
{
    if (!input_) {
        std::cout << "Warning: no input stream for trt executor." << std::endl;
        return;
    }
    if (!output_) {
        std::cout << "Warning: no output handler for trt executor." << std::endl;
        return;
    }

    SampleUniquePtr<nvinfer1::IExecutionContext> context(engine_->createExecutionContext());
    context->setOptimizationProfile(0);
    const auto bind_num = engine_->getNbBindings();
    for (int i = 0; i < bind_num; ++i) {
        if (engine_->bindingIsInput(i) && IsDynamicDim(engine_->getBindingDimensions(i))) {
            const auto dims = input_->GetDynamicDim(engine_->getBindingName(i));
            context->setBindingDimensions(i, dims);
        }
    }
    for (int i = 0; i < bind_num; ++i) {
        if (!engine_->bindingIsInput(i)) {
            output_->SetTensorDim(engine_->getBindingName(i), context->getBindingDimensions(i));
        }
    }

    std::cout << "***** Context Info *****" << std::endl;
    for (int i = 0; i < bind_num; ++i) {
        auto dims = context->getBindingDimensions(i);
        std::cout << "  [" << i << "] " << (engine_->bindingIsInput(i) ? "Input" : "Output");
        std::cout << ", Name: " << engine_->getBindingName(i) << ", Dim: " << dims << std::endl;
    }

    samplesCommon::BufferManager buffer_(engine_, 1, context.get());

    const auto input_tensor_names = input_->GetInputTensorNames(*engine_);
    std::vector<void *> input_host_buffers(input_tensor_names.size());
    std::vector<size_t> input_sizes(input_tensor_names.size());
    for (int i = 0; i < input_tensor_names.size(); ++i) {
        input_host_buffers[i] = buffer_.getHostBuffer(input_tensor_names[i]);
        input_sizes[i] = buffer_.size(input_tensor_names[i]);
    }
    assert(ValidateBuffer(input_host_buffers));

    const auto output_tensor_names = output_->GetOutputTensorNames(*engine_);
    std::vector<void *> output_host_buffers(output_tensor_names.size());
    std::vector<size_t> output_sizes(output_tensor_names.size());
    for (int i = 0; i < output_tensor_names.size(); ++i) {
        output_host_buffers[i] = buffer_.getHostBuffer(output_tensor_names[i]);
        output_sizes[i] = buffer_.size(output_tensor_names[i]);
    }
    assert(ValidateBuffer(output_host_buffers));

    while (!terminate_.load(std::memory_order::memory_order_relaxed)) {
        const auto take_res = input_->TryTake(input_host_buffers, input_sizes);
        if (!take_res) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        buffer_.copyInputToDevice();
        if (!context->executeV2(buffer_.getDeviceBindings().data())) {
            std::cout << "Warning: Unable to execute context." << std::endl;
            continue;
        }

        buffer_.copyOutputToHost();
        output_->Consume(output_host_buffers, output_sizes);
    }
}

void TrtExecutor::Terminate()
{
    terminate_.store(true, std::memory_order::memory_order_relaxed);
}
