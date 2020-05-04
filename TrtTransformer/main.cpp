#include "TrtTransformer.h"

#include <iostream>


int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " onnx_model_file TensorRT-save-file" << std::endl;
        return -1;
    }
    OnnxSampleParams params;
    params.onnxFileName = argv[1];
    params.outputTrtFile = argv[2];

    SampleOnnxBuilder onnx_builder(params);
    auto succeed = onnx_builder.build();
    if (!succeed) {
        std::cout << "Unable to build onnx model." << std::endl;
        return 1;
    }
    succeed = onnx_builder.save();
    if (!succeed) {
        return 2;
    }

    return 0;
}
