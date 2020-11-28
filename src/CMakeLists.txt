cmake_minimum_required(VERSION 2.8)

project(Tn LANGUAGES CXX)


find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES include)
MESSAGE(STATUS "found tensorrt header at ${TENSORRT_INCLUDE_DIR}")

find_library(TENSORRT_INFER libnvinfer.so
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_ONNX libnvonnxparser.so
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_PLUGIN libnvinfer_plugin.so
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64)

MESSAGE(STATUS "found tensorrt so at ${TENSORRT_INFER}")
MESSAGE(STATUS "found tensorrt so at ${TENSORRT_ONNX}")
MESSAGE(STATUS "found tensorrt so at ${TENSORRT_PLUGIN}")

include_directories(
    ./
    ../include
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
     )
link_directories(${OpenCV_LIBRARIES_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Wall -Ofast")
set(CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")
set(SRC_LIST resize.cu trt.cpp seg_conv.cu postprocess.cpp postprocess_gpu.cu)
cuda_add_library(Tn SHARED ${SRC_LIST})
target_link_libraries(Tn ${TENSORRT_INFER} ${TENSORRT_ONNX} ${TENSORRT_PLUGIN} ${OpenCV_LIBS} )
set_target_properties(Tn PROPERTIES VERSION 0.1 SOVERSION 0.1)