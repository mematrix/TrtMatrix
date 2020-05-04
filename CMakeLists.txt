# CMakeList.txt: Top CMake file. Do global config and add sub project.
#
cmake_minimum_required (VERSION 3.10)

project ("TrtTransformer")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "VER: ${CMAKE_VERSION}")
# set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA INC: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA TOOLKIT DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA LIB: ${CUDA_LIBRARIES}")


# Find TensorRT root dir from $ENV{PATH}
set(FILTER_ENV_PATH $ENV{PATH})
list(FILTER FILTER_ENV_PATH INCLUDE REGEX "TensorRT")
file(TO_CMAKE_PATH "${FILTER_ENV_PATH}" TENSORRT_SEARCH_PATH)
list(TRANSFORM TENSORRT_SEARCH_PATH APPEND "/../")
message(STATUS "TensorRT search dir: ${TENSORRT_SEARCH_PATH}")

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_SEARCH_PATH} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
if ("${TENSORRT_INCLUDE_DIR}" EQUAL "TENSORRT_INCLUDE_DIR-NOTFOUND")
    message(ERROR "Could not find TensorRT headers dir.")
endif ()

find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_SEARCH_PATH} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
message(STATUS "Found nvinfer lib at ${TENSORRT_LIBRARY_INFER}")
if ("${TENSORRT_LIBRARY_INFER}" EQUAL "TENSORRT_LIBRARY_INFER-NOTFOUND")
    message(ERROR "Could not find TensorRT nvinfer lib.")
endif ()
get_filename_component(TENSORRT_LIBRARY_DIR "${TENSORRT_LIBRARY_INFER}" DIRECTORY)
message(STATUS "Set TensorRT libraries dir: ${TENSORRT_LIBRARY_DIR}")


include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARY_DIR})


### Third party library
find_package(Boost REQUIRED)
message(STATUS "Boost INC: ${Boost_INCLUDE_DIR}")
include_directories(${Boost_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR}/ThirdParty/NumCpp/include)

set(BUILD_PROGRAMS OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_TESTING OFF)
add_subdirectory(ThirdParty/libsndfile)

# AudioFFT
set(AUDIO_FFT_SRC ${CMAKE_CURRENT_LIST_DIR}/ThirdParty/AudioFFT/AudioFFT.cpp)
set(AUDIO_FFT_INC_DIR ${CMAKE_CURRENT_LIST_DIR}/ThirdParty/AudioFFT)

### Third party end


set(SHARED_PATH ${CMAKE_CURRENT_LIST_DIR}/Shared)
set(SHARED_COMMON_INC ${SHARED_PATH}/include/common)
set(SHARED_COMMON_SRC ${SHARED_PATH}/src/common)
set(SHARED_COMMON_FILES ${SHARED_COMMON_INC}/buffers.h ${SHARED_COMMON_INC}/common.h 
  ${SHARED_COMMON_SRC}/logger.cpp ${SHARED_COMMON_INC}/logger.h)

include_directories(${SHARED_PATH}/include)

# sub project.
add_subdirectory(TrtTransformer)
add_subdirectory(TrtExecutor)
