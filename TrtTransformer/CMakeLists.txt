# CMakeList.txt: TrtTransformer
#
cmake_minimum_required (VERSION 3.8)


add_executable (TrtTransformer "main.cpp" "TrtTransformer.cpp" ${SHARED_COMMON_FILES})

target_link_libraries(TrtTransformer ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvonnxparser)

# TODO: Add test if necessary
