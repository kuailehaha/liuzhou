

# cmake_minimum_required(VERSION 3.20)

# # 禁用 Windows 下的 manifest 生成，防止 rc/mt 触发 "no such file or directory" 错误
# set(CMAKE_EXE_LINKER_FLAGS   "${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO"   CACHE STRING "No manifest for exes"   FORCE)
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /MANIFEST:NO" CACHE STRING "No manifest for dlls"  FORCE)


# # -------------------- EDIT HERE: 硬编码你的路径 --------------------
# # Python 可执行文件（你的 $PY）
# set(PYTHON_EXECUTABLE "C:/Users/wuqin/.conda/envs/torchenv/python.exe" CACHE FILEPATH "")
# get_filename_component(_PY_ENV_DIR "${PYTHON_EXECUTABLE}" DIRECTORY)

# # === 直接给出 torch 的常见路径（Windows/conda）===
# # <env>/Lib/site-packages/torch/{lib,share/cmake}
# set(TORCH_ROOT         "${_PY_ENV_DIR}/Lib/site-packages/torch"            CACHE PATH "" FORCE)
# set(TORCH_LIB_DIR      "${TORCH_ROOT}/lib"                                  CACHE PATH "" FORCE)
# set(TORCH_CMAKE_ROOT   "${TORCH_ROOT}/share/cmake"                          CACHE PATH "" FORCE)
# set(TORCH_PYTHON_LIBRARY "${TORCH_LIB_DIR}/torch_python.lib"                CACHE FILEPATH "" FORCE)

# # 让 find_package(Torch CONFIG) 走我们给的路径
# list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_ROOT}")



# set(CMAKE_C_COMPILER  "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"   CACHE FILEPATH "" FORCE)
# set(CMAKE_CXX_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"  CACHE FILEPATH "" FORCE)
# set(CMAKE_LINKER       "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/link.exe" CACHE FILEPATH "" FORCE)
# set(CMAKE_RC_COMPILER  "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/rc.exe" CACHE FILEPATH "" FORCE)
# set(CMAKE_MT           "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/mt.exe" CACHE FILEPATH "" FORCE)
# set(CMAKE_CUDA_HOST_COMPILER
#   "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"
#   CACHE FILEPATH "" FORCE)

# # CUDA 根目录（你的 v12.6），以及 nvcc / cudart.lib
# set(CUDA_TOOLKIT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6" CACHE PATH "")
# set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT}/bin/nvcc.exe" CACHE FILEPATH "" FORCE)
# set(CUDA_CUDART_LIBRARY "${CUDA_TOOLKIT_ROOT}/lib/x64/cudart.lib" CACHE FILEPATH "")

# # Torch / pybind11 的 CMake 路径（按你的 conda 环境改）
# # 典型路径如下（Windows Conda）：
# #   torch:   <env>/Lib/site-packages/torch/share/cmake
# #   pybind11:<env>/Lib/site-packages/pybind11/share/cmake/pybind11
# set(TORCH_CMAKE_ROOT "C:/Users/wuqin/.conda/envs/torchenv/Lib/site-packages/torch/share/cmake" CACHE PATH "")
# set(PYBIND11_CMAKE_DIR "C:/Users/wuqin/.conda/envs/torchenv/Lib/site-packages/pybind11/share/cmake/pybind11" CACHE PATH "")
# # -------------------------------------------------------------------

# # 基本编译设置
# set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
# set(CMAKE_CXX_STANDARD 17)
# add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# add_compile_definitions(_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)
# add_compile_definitions(_HAS_DEPRECATED_RESULT_OF=1)
# # CUDA 相关（固定 86 架构；按需改 80/86/89/90）
# set(CMAKE_CUDA_ARCHITECTURES 86)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")

# # 兼容不同 Find 模块对变量名的期待（都写死为同一根）
# set(CUDAToolkit_ROOT         "${CUDA_TOOLKIT_ROOT}" CACHE PATH "")
# set(CUDA_TOOLKIT_ROOT_DIR    "${CUDA_TOOLKIT_ROOT}" CACHE PATH "")
# set(CUDAToolkit_INCLUDE_DIR  "${CUDA_TOOLKIT_ROOT}/include" CACHE PATH "")
# set(CUDAToolkit_LIBRARY_DIR  "${CUDA_TOOLKIT_ROOT}/lib/x64" CACHE PATH "")

# project(liuzhou_v0 LANGUAGES CXX CUDA)

# if(MSVC)
#   foreach(_cuda_rule CMAKE_CUDA_COMPILE_OBJECT CMAKE_CUDA_DEVICE_LINK_LIBRARY CMAKE_CUDA_DEVICE_LINK_EXECUTABLE)
#     if(DEFINED ${_cuda_rule})
#       string(REPLACE "-Xcompiler=-Fd<TARGET_COMPILE_PDB>,-FS" "-Xcompiler=\"-Fd<TARGET_COMPILE_PDB>,-FS\"" ${_cuda_rule} "${${_cuda_rule}}")
#     endif()
#   endforeach()
# endif()
# # 让 CMake 能直接找到 Torch/pybind11 的 Config.cmake
# # 注意：顺序无所谓，这里统一塞进 CMAKE_PREFIX_PATH
# set(CMAKE_PREFIX_PATH
#     "${TORCH_CMAKE_ROOT};${PYBIND11_CMAKE_DIR}"
#     CACHE STRING "Prefix path for Torch/pybind11" FORCE)

# # Python（用我们硬编码的解释器）
# set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH "" FORCE)


# find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# # pybind11 / Torch（均走 CONFIG 模式，直接吃上面的 CMAKE_PREFIX_PATH）
# find_package(pybind11 CONFIG REQUIRED)
# # Torch 有时还需要显式 Torch_DIR；顺手把 Caffe2_DIR 也写死（部分旧版/脚本仍会找它）
# set(Torch_DIR   "${TORCH_CMAKE_ROOT}/Torch"   CACHE PATH "" FORCE)
# set(Caffe2_DIR  "${TORCH_CMAKE_ROOT}/Caffe2"  CACHE PATH "" FORCE)
# find_package(Torch REQUIRED)

# # 子目录（你的目标与源码组织放在 src/ 下）
# add_subdirectory(src)

cmake_minimum_required(VERSION 3.20)

# ================= 基础开关 =================
# 关掉 Windows manifest，避免 rc/mt 报错
set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO"   CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /MANIFEST:NO" CACHE STRING "" FORCE)

# ================= 硬编码环境路径（按需改） =================
# Python
set(PYTHON_EXECUTABLE "C:/Users/wuqin/.conda/envs/torchenv/python.exe" CACHE FILEPATH "" FORCE)
get_filename_component(_PY_ENV_DIR "${PYTHON_EXECUTABLE}" DIRECTORY)

# Torch (Conda)
set(TORCH_ROOT       "${_PY_ENV_DIR}/Lib/site-packages/torch"            CACHE PATH "" FORCE)
set(TORCH_LIB_DIR    "${TORCH_ROOT}/lib"                                  CACHE PATH "" FORCE)
set(TORCH_CMAKE_ROOT "${TORCH_ROOT}/share/cmake"                          CACHE PATH "" FORCE)
set(TORCH_PYTHON_LIBRARY "${TORCH_LIB_DIR}/torch_python.lib"              CACHE FILEPATH "" FORCE)
set(Torch_DIR        "${TORCH_CMAKE_ROOT}/Torch"                          CACHE PATH "" FORCE)
set(Caffe2_DIR       "${TORCH_CMAKE_ROOT}/Caffe2"                         CACHE PATH "" FORCE)

# pybind11 CMake
set(PYBIND11_CMAKE_DIR "C:/Users/wuqin/.conda/envs/torchenv/Lib/site-packages/pybind11/share/cmake/pybind11" CACHE PATH "" FORCE)

# VS 工具链 (x64)
set(CMAKE_C_COMPILER   "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"   CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"   CACHE FILEPATH "" FORCE)
set(CMAKE_LINKER       "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/link.exe" CACHE FILEPATH "" FORCE)
set(CMAKE_RC_COMPILER  "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/rc.exe" CACHE FILEPATH "" FORCE)
set(CMAKE_MT           "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/mt.exe" CACHE FILEPATH "" FORCE)

# CUDA 12.6
set(CUDA_TOOLKIT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6" CACHE PATH "" FORCE)
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT}/bin/nvcc.exe" CACHE FILEPATH "" FORCE)
set(CUDAToolkit_ROOT         "${CUDA_TOOLKIT_ROOT}"         CACHE PATH "" FORCE)
set(CUDA_TOOLKIT_ROOT_DIR    "${CUDA_TOOLKIT_ROOT}"         CACHE PATH "" FORCE)
set(CUDAToolkit_INCLUDE_DIR  "${CUDA_TOOLKIT_ROOT}/include" CACHE PATH "" FORCE)
set(CUDAToolkit_LIBRARY_DIR  "${CUDA_TOOLKIT_ROOT}/lib/x64" CACHE PATH "" FORCE)
set(CUDA_CUDART_LIBRARY      "${CUDA_TOOLKIT_ROOT}/lib/x64/cudart.lib" CACHE FILEPATH "" FORCE)

# 让 find_package 能按我们的路径找
set(CMAKE_PREFIX_PATH
  "${TORCH_CMAKE_ROOT};${PYBIND11_CMAKE_DIR}"
  CACHE STRING "Prefix path for Torch/pybind11" FORCE)
# # 让 -ccbin 使用 8.3 短路径（和 nvcc 打印的一模一样）
# set(MSVC_CL_DIR_SHORT "C:/PROGRA~2/MICROS~4/2022/BUILDT~1/VC/Tools/MSVC/1444~1.352/bin/Hostx64/x64")
# set(CMAKE_CUDA_HOST_COMPILER "${MSVC_CL_DIR_SHORT}/cl.exe" CACHE FILEPATH "" FORCE)

# # 确保 PATH 里的第一个 cl.exe 也是同一个短路径，避免 nvcc 比对不一致
# # （注意：这会只影响本次 CMake 配置/生成流程的环境）
# set(ENV{PATH} "${MSVC_CL_DIR_SHORT};$ENV{PATH}")
# set(MSVC_CL_DIR_LONG "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64")
# set(CMAKE_CUDA_HOST_COMPILER "${MSVC_CL_DIR_LONG}/cl.exe" CACHE FILEPATH "" FORCE)
# set(CUDAHOSTCXX "${MSVC_CL_DIR_LONG}/cl.exe" CACHE FILEPATH "" FORCE)
# # 确保 nvcc 在 PATH 里看到的 cl.exe 也是这条长路径
# set(ENV{PATH} "${MSVC_CL_DIR_LONG};$ENV{PATH}")

# ✅ 统一使用你写死的长路径
set(MSVC_CL_DIR_LONG "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64")

# C/C++
set(CMAKE_C_COMPILER   "${MSVC_CL_DIR_LONG}/cl.exe" CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER "${MSVC_CL_DIR_LONG}/cl.exe" CACHE FILEPATH "" FORCE)

# 告诉 nvcc 的 host 编译器（有些 CMake 版本还是会自己塞 -ccbin，用长路径兜底）
set(CMAKE_CUDA_HOST_COMPILER "${MSVC_CL_DIR_LONG}/cl.exe" CACHE FILEPATH "" FORCE)
set(CUDAHOSTCXX "${MSVC_CL_DIR_LONG}/cl.exe" CACHE FILEPATH "" FORCE)

# 关键：强制 nvcc 使用这条长路径作为 -ccbin
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-bindir=\"${MSVC_CL_DIR_LONG}\"")

# 把同一条长路径放到 PATH 最前，保证 nvcc 检测 PATH 时看到的也是同一条
set(ENV{PATH} "${MSVC_CL_DIR_LONG};$ENV{PATH}")
# ================= 工程设置 =================
project(liuzhou_v0 LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 你项目里需要的宏
add_compile_definitions(_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH _HAS_DEPRECATED_RESULT_OF=1)

# 架构
set(CMAKE_CUDA_ARCHITECTURES 86)

# ================= 关键：严格分离 MSVC 与 NVCC 选项 =================
# —— 只给 C/C++ ：（不会传到 NVCC）
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} /utf-8 /bigobj /MP /wd4267")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8 /bigobj /MP /wd4267")

# —— 只给 CUDA（通过 -Xcompiler 转发到 cl.exe），且修正 CUDA 选项拼写
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler --expt-relaxed-constexpr --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8 -Xcompiler=/bigobj -Xcompiler=/MP -Xcompiler=/wd4267")

# （可选）修正 CMake 生成的 PDB 转发参数引号问题
if(MSVC)
  foreach(_cuda_rule CMAKE_CUDA_COMPILE_OBJECT CMAKE_CUDA_DEVICE_LINK_LIBRARY CMAKE_CUDA_DEVICE_LINK_EXECUTABLE)
    if(DEFINED ${_cuda_rule})
      string(REPLACE "-Xcompiler=-Fd<TARGET_COMPILE_PDB>,-FS"
                     "-Xcompiler=\"-Fd<TARGET_COMPILE_PDB>,-FS\""
             ${_cuda_rule} "${${_cuda_rule}}")
    endif()
  endforeach()
endif()

# ================= 依赖 =================
# 指定 Python
set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH "" FORCE)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11（CONFIG）
find_package(pybind11 CONFIG REQUIRED)

# Torch
find_package(Torch REQUIRED)

# ================= 子目录 =================
add_subdirectory(src)
