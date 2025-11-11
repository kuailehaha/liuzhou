

cmake_minimum_required(VERSION 3.20)

# 禁用 Windows 下的 manifest 生成，防止 rc/mt 触发 "no such file or directory" 错误
set(CMAKE_EXE_LINKER_FLAGS   "${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO"   CACHE STRING "No manifest for exes"   FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /MANIFEST:NO" CACHE STRING "No manifest for dlls"  FORCE)


# -------------------- EDIT HERE: 硬编码你的路径 --------------------
# Python 可执行文件（你的 $PY）
set(PYTHON_EXECUTABLE "C:/Users/wuqin/.conda/envs/torchenv/python.exe" CACHE FILEPATH "")
get_filename_component(_PY_ENV_DIR "${PYTHON_EXECUTABLE}" DIRECTORY)

# === 直接给出 torch 的常见路径（Windows/conda）===
# <env>/Lib/site-packages/torch/{lib,share/cmake}
set(TORCH_ROOT         "${_PY_ENV_DIR}/Lib/site-packages/torch"            CACHE PATH "" FORCE)
set(TORCH_LIB_DIR      "${TORCH_ROOT}/lib"                                  CACHE PATH "" FORCE)
set(TORCH_CMAKE_ROOT   "${TORCH_ROOT}/share/cmake"                          CACHE PATH "" FORCE)
set(TORCH_PYTHON_LIBRARY "${TORCH_LIB_DIR}/torch_python.lib"                CACHE FILEPATH "" FORCE)

# 让 find_package(Torch CONFIG) 走我们给的路径
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_ROOT}")



set(CMAKE_C_COMPILER  "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"   CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"  CACHE FILEPATH "" FORCE)
set(CMAKE_LINKER       "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/link.exe" CACHE FILEPATH "" FORCE)
set(CMAKE_RC_COMPILER  "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/rc.exe" CACHE FILEPATH "" FORCE)
set(CMAKE_MT           "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/mt.exe" CACHE FILEPATH "" FORCE)
set(CMAKE_CUDA_HOST_COMPILER
  "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"
  CACHE FILEPATH "" FORCE)

# CUDA 根目录（你的 v12.6），以及 nvcc / cudart.lib
set(CUDA_TOOLKIT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT}/bin/nvcc.exe" CACHE FILEPATH "" FORCE)
set(CUDA_CUDART_LIBRARY "${CUDA_TOOLKIT_ROOT}/lib/x64/cudart.lib" CACHE FILEPATH "")

# Torch / pybind11 的 CMake 路径（按你的 conda 环境改）
# 典型路径如下（Windows Conda）：
#   torch:   <env>/Lib/site-packages/torch/share/cmake
#   pybind11:<env>/Lib/site-packages/pybind11/share/cmake/pybind11
set(TORCH_CMAKE_ROOT "C:/Users/wuqin/.conda/envs/torchenv/Lib/site-packages/torch/share/cmake" CACHE PATH "")
set(PYBIND11_CMAKE_DIR "C:/Users/wuqin/.conda/envs/torchenv/Lib/site-packages/pybind11/share/cmake/pybind11" CACHE PATH "")
# -------------------------------------------------------------------

# 基本编译设置
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_STANDARD 17)
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_definitions(_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)
add_compile_definitions(_HAS_DEPRECATED_RESULT_OF=1)
# CUDA 相关（固定 86 架构；按需改 80/86/89/90）
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")

# 兼容不同 Find 模块对变量名的期待（都写死为同一根）
set(CUDAToolkit_ROOT         "${CUDA_TOOLKIT_ROOT}" CACHE PATH "")
set(CUDA_TOOLKIT_ROOT_DIR    "${CUDA_TOOLKIT_ROOT}" CACHE PATH "")
set(CUDAToolkit_INCLUDE_DIR  "${CUDA_TOOLKIT_ROOT}/include" CACHE PATH "")
set(CUDAToolkit_LIBRARY_DIR  "${CUDA_TOOLKIT_ROOT}/lib/x64" CACHE PATH "")

project(liuzhou_v0 LANGUAGES CXX CUDA)
# 让 CMake 能直接找到 Torch/pybind11 的 Config.cmake
# 注意：顺序无所谓，这里统一塞进 CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH
    "${TORCH_CMAKE_ROOT};${PYBIND11_CMAKE_DIR}"
    CACHE STRING "Prefix path for Torch/pybind11" FORCE)

# Python（用我们硬编码的解释器）
set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH "" FORCE)


find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11 / Torch（均走 CONFIG 模式，直接吃上面的 CMAKE_PREFIX_PATH）
find_package(pybind11 CONFIG REQUIRED)
# Torch 有时还需要显式 Torch_DIR；顺手把 Caffe2_DIR 也写死（部分旧版/脚本仍会找它）
set(Torch_DIR   "${TORCH_CMAKE_ROOT}/Torch"   CACHE PATH "" FORCE)
set(Caffe2_DIR  "${TORCH_CMAKE_ROOT}/Caffe2"  CACHE PATH "" FORCE)
find_package(Torch REQUIRED)

# 子目录（你的目标与源码组织放在 src/ 下）
add_subdirectory(src)

