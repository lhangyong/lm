cmake_minimum_required(VERSION 3.10)
project(qc_detect CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV（板子上 apt 安装的 libopencv-dev）
find_package(OpenCV REQUIRED)

# RKNN SDK 路径：指向含 rknn_api.h 的 include 目录和 librknnrt.so 的 lib 目录。
# 板子上若 rknn_api.h 已在 /usr/include、librknnrt.so 已在 /usr/lib，可留空。
# 交叉编译时填 SDK 里的 libs/aarch64-linux-gnu。
set(RKNN_DIR "" CACHE PATH "RKNN SDK 根目录(含 include/rknn_api.h 与 libs)")

include_directories(${CMAKE_SOURCE_DIR}/include)
if(RKNN_DIR)
    include_directories(${RKNN_DIR}/include)
    link_directories(${RKNN_DIR}/libs)
endif()

add_executable(qc_detect
    src/main.cpp
    src/rknn_model.cpp
    src/postprocess.cpp
    src/memory_bank.cpp
)

target_link_libraries(qc_detect ${OpenCV_LIBS} rknnrt)
