
# v0/src/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# 解决新版 CMake 中 pybind11 的 FindPython 警告
set(PYBIND11_FINDPYTHON ON)

# 先找 pybind11 / Torch（上层已把路径塞进 CMAKE_PREFIX_PATH）
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

set(V0_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../include)

# Torch Python 库：优先用导出的目标，其次回落到变量，最后手动查找
set(_TORCH_PY_LIB "")
if(TARGET torch_python)
  set(_TORCH_PY_LIB torch_python)
elseif(TARGET Torch::Python)
  set(_TORCH_PY_LIB Torch::Python)
elseif(DEFINED TORCH_PYTHON_LIBRARY)
  set(_TORCH_PY_LIB "${TORCH_PYTHON_LIBRARY}")
else()
  # 手动查找 libtorch_python.so
  find_library(_TORCH_PY_LIB_PATH
    NAMES torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib"
    NO_DEFAULT_PATH
  )
  if(_TORCH_PY_LIB_PATH)
    set(_TORCH_PY_LIB "${_TORCH_PY_LIB_PATH}")
  endif()
endif()

message(STATUS "Torch Python library: ${_TORCH_PY_LIB}")

# ---- 可选 CUDA 源清单（有 .cu 再填；没有就留空，完全不触发 nvcc） ----
option(BUILD_CUDA_KERNELS "Build CUDA kernels for v0_core" ${USE_CUDA})
set(CUDA_SRCS)

if(BUILD_CUDA_KERNELS)
  list(APPEND CUDA_SRCS
    game/fast_legal_mask_cuda.cu
    game/fast_apply_moves_cuda.cu
  )
endif()

# 封装一个添加扩展的函数
function(add_v0_extension target)
  pybind11_add_module(${target} MODULE ${ARGN})

  target_include_directories(${target} PRIVATE
    ${V0_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
  )

  # 链接 Torch（含 torch_python）、保持和你原先一致
  target_link_libraries(${target} PRIVATE
    ${TORCH_LIBRARIES}
    ${_TORCH_PY_LIB}
  )

  target_compile_definitions(${target} PRIVATE
    TORCH_API_INCLUDE_EXTENSION_H
    NOMINMAX
  )

  # ！！！关键修复：严格区分 CXX 与 CUDA 的编译选项 ！！！
  if(MSVC)
    # 仅 C++ 源吃 MSVC 原生开关（不会泄漏到 NVCC）
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/utf-8;/bigobj;/MP;/wd4267;/EHsc>
    )
    # CUDA 源通过 -Xcompiler 转发到 cl.exe（避免 NVCC 把 /xxx 当成“输入文件”）
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8;-Xcompiler=/bigobj;-Xcompiler=/MP;-Xcompiler=/wd4267;-Xcompiler=/EHsc>
    )
  else()
    # GCC/Clang：常规警告和 -fPIC，CUDA 分支保证与主机编译器一致
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
      $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
      $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
    )
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()
endfunction()

add_v0_extension(
  v0_core
  bindings/module.cpp
  game/game_state.cpp
  game/tensor_state_batch.cpp
  rules/rule_engine.cpp
  moves/move_generator.cpp
  net/encoding.cpp
  net/inference_engine.cpp
  net/torchscript_runner.cpp
  net/project_policy_logits_fast.cpp
  game/fast_legal_mask.cpp
  game/fast_apply_moves.cpp
  mcts/mcts_core.cpp
  ${CUDA_SRCS}  # 有 .cu 时才会起作用
)

# ---- 仅当有 CUDA 源且你想编 CUDA 时，才对 v0_core 打开 CUDA 支持 ----
if(BUILD_CUDA_KERNELS AND CUDA_SRCS)
  # 目标级别启用 CUDA 语言（而不是全局 enable_language）
  if(NOT CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  endif()

  # 明确告诉 CMake：这些文件用 CUDA 编译
  set_source_files_properties(${CUDA_SRCS} PROPERTIES LANGUAGE CUDA)

  # CUDA 目标属性与体系结构
  # 注意：关闭 CUDA_SEPARABLE_COMPILATION，因为 .cu 文件之间没有跨文件的
  # __device__ 函数调用，不需要分离编译。分离编译会导致 fatbinData 符号问题。
  set_target_properties(v0_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )

  # 仅对 CUDA 编译路径附加 nvcc 兼容选项（避免影响 C++ 源）
  target_compile_options(v0_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler;--expt-relaxed-constexpr;--expt-extended-lambda>
  )

  # MSVC STL / 工具集版本与 CUDA 头可能不完全匹配时的兜底宏（仅 CUDA 路径）
  target_compile_definitions(v0_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH>
  )
endif()

# 你原来的宏
target_compile_definitions(
  v0_core PRIVATE
  PROJECT_POLICY_NO_MODULE
  FAST_LEGAL_MASK_NO_MODULE
  FAST_APPLY_MOVES_NO_MODULE
)

if(BUILD_CUDA_KERNELS)
  target_compile_definitions(v0_core PRIVATE V0_HAS_CUDA_LEGAL_MASK V0_HAS_CUDA_APPLY_MOVES)
endif()

if(C10_CUDA_LIBRARY)
  target_compile_definitions(v0_core PRIVATE TORCH_CUDA_AVAILABLE)
endif()
