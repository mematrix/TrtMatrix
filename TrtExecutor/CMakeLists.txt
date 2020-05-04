# CMakeLists.txt: TrtExecutor.
#
cmake_minimum_required (VERSION 3.8)


add_executable(TrtExecutor main.cpp TrtExecutor.cpp ${SHARED_COMMON_FILES} ${AUDIO_FFT_SRC} "AudioUtils.cpp")
target_include_directories(TrtExecutor PRIVATE ${AUDIO_FFT_INC_DIR})
target_link_libraries(TrtExecutor ${CUDA_LIBRARIES} nvinfer nvinfer_plugin sndfile)
