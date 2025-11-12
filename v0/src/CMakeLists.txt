# 解决新版 CMake 中 pybind11 的 FindPython 警告
set(PYBIND11_FINDPYTHON ON)

# 先找 pybind11 / Torch（上层已把路径塞进 CMAKE_PREFIX_PATH）
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

set(V0_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../include)

# ---- 可选 CUDA 源清单（有 .cu 再填；没有就留空，完全不触发 nvcc） ----
option(BUILD_CUDA_KERNELS "Build CUDA kernels for v0_core" ${USE_CUDA})
set(CUDA_SRCS
    # 例如：kernels/my_kernel.cu
)

# function(add_v0_extension target)
#   pybind11_add_module(${target} MODULE ${ARGN})
#   target_include_directories(${target} PRIVATE ${V0_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
#   target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES})
#   target_link_libraries(${target} PRIVATE
#   "C:/Users/wuqin/.conda/envs/torchenv/Lib/site-packages/torch/lib/torch_python.lib"
# )
#   target_compile_definitions(${target} PRIVATE TORCH_API_INCLUDE_EXTENSION_H)
#   if(MSVC)
#     target_compile_options(${target} PRIVATE /bigobj /MP)
#     target_compile_definitions(${target} PRIVATE NOMINMAX)
#   else()
#     target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-unused-parameter)
#   endif()
# endfunction()

function(add_v0_extension target)
  pybind11_add_module(${target} MODULE ${ARGN})
  target_include_directories(${target} PRIVATE ${V0_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
  # 关键：用上父级传下来的变量
  target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES} "${TORCH_PYTHON_LIBRARY}")
  target_compile_definitions(${target} PRIVATE TORCH_API_INCLUDE_EXTENSION_H)
  if(MSVC)
    target_compile_options(${target} PRIVATE /bigobj /MP /wd4267)
    target_compile_definitions(${target} PRIVATE NOMINMAX)
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
  net/project_policy_logits_fast.cpp
  game/fast_legal_mask.cpp
  game/fast_apply_moves.cpp
  mcts/mcts_core.cpp
  ${CUDA_SRCS}  # 有 .cu 时才会起作用
)

# ---- 仅当有 CUDA 源且你想编 CUDA 时，才对 v0_core 打开 CUDA 支持 ----
if(BUILD_CUDA_KERNELS AND CUDA_SRCS)
  # 目标级别启用 CUDA 语言（而不是全局 enable_language）
  enable_language(CUDA)

  # 明确告诉 CMake：这些文件用 CUDA 编译
  set_source_files_properties(${CUDA_SRCS} PROPERTIES LANGUAGE CUDA)

  # CUDA 目标属性与体系结构
  set_target_properties(v0_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )

  # 仅对 CUDA 编译路径附加 nvcc 兼容选项（避免影响 C++ 源）
  target_compile_options(v0_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler --expt-relaxed-constexpr --extended-lambda>
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



# set(PYBIND11_FINDPYTHON ON)
# find_package(pybind11 CONFIG REQUIRED)
# find_package(Torch REQUIRED)

# set(V0_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../include)

# # ---- 自动定位 torch_python.lib ----
# # 先拿到 Torch::torch 的具体 lib 路径
# get_target_property(_TORCH_LIB Torch::torch IMPORTED_LOCATION_RELEASE)
# if(NOT _TORCH_LIB)
#   get_target_property(_TORCH_LIB Torch::torch IMPORTED_LOCATION)
# endif()
# get_filename_component(_TORCH_LIBDIR "${_TORCH_LIB}" DIRECTORY)

# # 在同目录查找 torch_python.lib
# find_library(TORCH_PYTHON_LIBRARY
#   NAMES torch_python
#   PATHS "${_TORCH_LIBDIR}"
#   NO_DEFAULT_PATH
#   REQUIRED)

# # ---- 可选 CUDA 清单 ----
# option(BUILD_CUDA_KERNELS "Build CUDA kernels for v0_core" ${USE_CUDA})
# set(CUDA_SRCS)

# function(add_v0_extension target)
#   pybind11_add_module(${target} MODULE ${ARGN})
#   target_include_directories(${target} PRIVATE ${V0_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
#   # 关键：把 torch_python 加进来（对当前 ${target}）
#   target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES} "${TORCH_PYTHON_LIBRARY}")
#   target_compile_definitions(${target} PRIVATE TORCH_API_INCLUDE_EXTENSION_H)
#   if(MSVC)
#     target_compile_options(${target} PRIVATE /bigobj /MP /wd4267)
#     target_compile_definitions(${target} PRIVATE NOMINMAX)
#   else()
#     target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-unused-parameter)
#   endif()
# endfunction()

# add_v0_extension(
#   v0_core
#   bindings/module.cpp
#   game/game_state.cpp
#   game/tensor_state_batch.cpp
#   rules/rule_engine.cpp
#   moves/move_generator.cpp
#   net/encoding.cpp
#   mcts/mcts_core.cpp
#   ../../v1/net/project_policy_logits_fast.cpp
#   ../../v1/game/fast_legal_mask.cpp
#   ../../v1/game/fast_apply_moves.cpp
#   ${CUDA_SRCS}
# )

# if(BUILD_CUDA_KERNELS AND CUDA_SRCS)
#   enable_language(CUDA)
#   set_source_files_properties(${CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
#   set_target_properties(v0_core PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
#     CUDA_STANDARD 17
#     CUDA_STANDARD_REQUIRED ON
#   )
#   target_compile_options(v0_core PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler --expt-relaxed-constexpr --extended-lambda>
#   )
#   target_compile_definitions(v0_core PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH>
#   )
# endif()

# target_compile_definitions(
#   v0_core PRIVATE
#   PROJECT_POLICY_NO_MODULE
#   FAST_LEGAL_MASK_NO_MODULE
#   FAST_APPLY_MOVES_NO_MODULE
# )
