// TrtTransformer.h: Transform ONNX model to TensorRT model
//

#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "common/buffers.h"


//!
//! \brief The SampleParams structure groups the basic parameters required by
//!        all sample networks.
//!
struct SampleParams
{
    int batchSize{ 1 };                     //!< Number of inputs in a batch
    int dlaCore{ -1 };                   //!< Specify the DLA core to run network on.
    bool int8{ false };                  //!< Allow runnning the network in Int8 mode.
    bool fp16{ false };                  //!< Allow running the network in FP16 mode.
    // std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    std::string outputTrtFile;
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
};

//!
//! \brief The CaffeSampleParams structure groups the additional parameters required by
//!         networks that use caffe
//!
struct CaffeSampleParams : public SampleParams
{
    std::string prototxtFileName; //!< Filename of prototxt design file of a network
    std::string weightsFileName;  //!< Filename of trained weights file of a network
    std::string meanFileName;     //!< Filename of mean file of a network
};


//!
//! \brief The OnnxSampleParams structure groups the additional parameters required by
//!         networks that use ONNX
//!
struct OnnxSampleParams : public SampleParams
{
    std::string onnxFileName; //!< Filename of ONNX file of a network
};


//! \brief  The SampleOnnxBuilder class implements the ONNX sample
//!
//! \details It creates the network using an ONNX model
//!
class SampleOnnxBuilder
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleOnnxBuilder(const OnnxSampleParams& params)
        : mParams(params), mEngine(nullptr) { }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    bool save() const;

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    // bool infer();

private:
    OnnxSampleParams mParams; //!< The parameters for the sample.
    
    // nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    // nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    // int mNumber{ 0 };             //!< The number to classify

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser) const;
};
