cmake_minimum_required(VERSION 2.8)

project(tensorrt_solo)

set(CMAKE_BUILD_TYPE Relaese)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Wall -Ofast")

add_subdirectory(src) 

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURVE_DIR}/include)
include_directories(${OPENCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARIES_DIRS})

ADD_EXECUTABLE(cuda_engine cuda_engine.cpp )
target_link_libraries(cuda_engine ${OpenCV_LIBS} Tn)

ADD_EXECUTABLE(infer_gpupost test_gpupost.cpp )
target_link_libraries(infer_gpupost ${OpenCV_LIBS} Tn)