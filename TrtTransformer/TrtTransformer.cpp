// TrtTransformer.cpp: Impl
//

#include "TrtTransformer.h"

#include "common/buffers.h"


static const char *data_type_to_str(nvinfer1::DataType type)
{
    switch (type) {
        case DataType::kFLOAT: return "[fp32]";
        case DataType::kHALF: return "[fp16]";
        case DataType::kINT8: return "[int8]";
        case DataType::kINT32: return "[int32]";
        default: return "[unknown]";
    }
}

bool SampleOnnxBuilder::build()
{
    std::cout << "Construct Builder" << std::endl;
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        return false;
    }

    std::cout << "Construct NetworkDef" << std::endl;
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    std::cout << "Construct BuilderConfig" << std::endl;
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    std::cout << "Construct Parser" << std::endl;
    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
        return false;
    }

    std::cout << "Construct Network" << std::endl;
    const auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed) {
        return false;
    }

    std::cout << "Construct CudaEngine" << std::endl;
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine) {
        return false;
    }

    std::cout << "***** Network Info *****" << std::endl;
    std::cout << "Input num: " << network->getNbInputs() << std::endl;
    for (int i = 0; i < network->getNbInputs(); ++i) {
        auto *tensor = network->getInput(i);
        std::cout << "  [" << i << "] Dim: " << tensor->getDimensions();
        std::cout << ", Name: " << tensor->getName() << ", Type: " << data_type_to_str(tensor->getType()) << std::endl;
    }

    std::cout << "Output num: " << network->getNbOutputs() << std::endl;
    for (int i = 0; i < network->getNbOutputs(); ++i) {
        auto *tensor = network->getOutput(i);
        std::cout << "  [" << i << "] Dim: " << tensor->getDimensions();
        std::cout << ", Name: " << tensor->getName() << ", Type: " << data_type_to_str(tensor->getType()) << std::endl;
    }

    std::cout << std::endl;
    std::cout << std::boolalpha;
    std::cout << "***** Cuda Engine Info *****" << std::endl;
    std::cout << "Name: " << mEngine->getName() << std::endl;
    std::cout << "Refit: " << mEngine->isRefittable() << std::endl;
    std::cout << "Layer num: " << mEngine->getNbLayers() << std::endl;
    std::cout << "Bindings num: " << mEngine->getNbBindings() << std::endl;
    for (int i = 0; i < mEngine->getNbBindings(); ++i) {
        std::cout << "  [" << i << "] " << (mEngine->bindingIsInput(i) ? "Input" : "Output") << std::endl;
        std::cout << "      Name: " << mEngine->getBindingName(i) << std::endl;
        std::cout << "      Dim: " << mEngine->getBindingDimensions(i) << std::endl;
        std::cout << "      Vec-ed idx: " << mEngine->getBindingVectorizedDim(i) << std::endl;
        if (-1 != mEngine->getBindingVectorizedDim(i)) {
            std::cout << "      ComponentsPerElem: " << mEngine->getBindingComponentsPerElement(i) << std::endl;
            std::cout << "      BytesPerComponent: " << mEngine->getBindingBytesPerComponent(i) << std::endl;
        }
        std::cout << "      DataType: " << data_type_to_str(mEngine->getBindingDataType(i)) << std::endl;
        std::cout << "      Format: " << mEngine->getBindingFormatDesc(i) << std::endl;
        std::cout << "      ShapeBinding: " << mEngine->isShapeBinding(i) << std::endl;
        std::cout << "      ExecutionBinding: " << mEngine->isExecutionBinding(i) << std::endl;
    }

    return true;
}

bool SampleOnnxBuilder::save() const
{
    if (!mEngine) {
        return false;
    }

    auto mem = SampleUniquePtr<nvinfer1::IHostMemory>(mEngine->serialize());
    if (!mem) {
        std::cerr << "[Save] Unable to serialize cuda engine." << std::endl;
        return false;
    }
    std::cout << "[Save] size=" << mem->size() << "bytes, type: " << data_type_to_str(mem->type()) << std::endl;

    std::ofstream ofs(mParams.outputTrtFile, std::ios_base::binary);
    if (!ofs) {
        std::cerr << "[Save] Unable to open file to write: " << mParams.outputTrtFile << std::endl;
        return false;
    }

    ofs.write(static_cast<char*>(mem->data()), mem->size());
    ofs.flush();
    ofs.close();

    // std::ifstream ifs(mParams.outputTrtFile, std::ios_base::binary);
    // std::vector<char> buf(mem->size());
    // ifs.read(buf.data(), mem->size());
    // int n = memcmp(mem->data(), buf.data(), mem->size());
    // std::cout << "** Info: mem compare: " << n << std::endl;
    //
    // auto runtime = SampleUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    // SampleUniquePtr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(buf.data(), buf.size()));
    // if (!engine) {
    //     std::cerr << "Error: engine deserialize error." << std::endl;
    // }

    return true;
}

bool SampleOnnxBuilder::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                                         SampleUniquePtr<nvinfer1::INetworkDefinition> &network,
                                         SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                                         SampleUniquePtr<nvonnxparser::IParser> &parser) const
{
    auto parsed = parser->parseFromFile(mParams.onnxFileName.c_str(), static_cast<int>(gLogger.GetSeverity()));
    if (!parsed) {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(16_MiB);
    if (mParams.fp16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8) {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    // add optimization config
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input", OptProfileSelector::kMIN, Dims3{1, 1, 257});
    profile->setDimensions("input", OptProfileSelector::kOPT, Dims3{1, 1, 257});
    profile->setDimensions("input", OptProfileSelector::kMAX, Dims3{1, 1, 257});
    config->addOptimizationProfile(profile);

    return true;
}
