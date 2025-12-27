cmake_minimum_required(VERSION 3.20)

# ================= 平台无关的基础设置（Linux/macOS/WSL） =================
project(liuzhou_v0 LANGUAGES CXX)

option(USE_CUDA "Build with CUDA kernels" ON)
set(BUILD_CUDA_KERNELS ${USE_CUDA} CACHE BOOL "Build CUDA kernels for v0_core")

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# GCC/Clang 常用安全开关；MSVC 仍然可以自定义
# 注意：每个选项需要单独使用生成器表达式，避免分号列表被错误解析
add_compile_options(
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wextra>
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wpedantic>
)

# ================= 依赖 =================
# Python（允许用户用 -DPython3_EXECUTABLE=... 覆盖）
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11（CONFIG 模式，推荐通过 pip/conda 安装）
find_package(pybind11 CONFIG REQUIRED)

# Torch（需提前设置 Torch_DIR 或 CMAKE_PREFIX_PATH 指向 libtorch/share/cmake/Torch）
find_package(Torch REQUIRED)

# CUDA（仅在需要时启用；没有 CUDA 或只想 CPU 时可 -DUSE_CUDA=OFF）
if(BUILD_CUDA_KERNELS)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA arch list" FORCE)
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

# ================= 子目录 =================
add_subdirectory(src)
