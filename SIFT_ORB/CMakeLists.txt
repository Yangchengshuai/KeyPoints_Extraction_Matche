cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Debug)
set(DCMAKE_BUILD_TYPE Debug)

project(KeyPointsExtractionAndMatche)

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

## CUDA(可选择，SIFT可以进行GPU加速)
# FIND_PACKAGE(CUDA)
# IF(CUDA_FOUND)
# set (EXTRA_INC_DIRS
# ${CUDA_INCLUDE_DIRS}
# ${CUDA_SDK_INCLUDE_DIR}
# )
# cuda_include_directories(${EXTRA_INC_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
# set (EXTRA_LIBRARIES
# ${CUDA_LIBS}
# ${CUDA_cublas_LIBRARY}
# )
# ENDIF(CUDA_FOUND)

add_executable(SiftExtractionAndMatche SIFT.cpp)
add_executable(OrbExtractionAndMatche ORB.cpp)

target_link_libraries(SiftExtractionAndMatche ${OpenCV_LIBS})
target_link_libraries(OrbExtractionAndMatche ${OpenCV_LIBS})