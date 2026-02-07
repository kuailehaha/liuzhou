# V0 C++ Core 构建指南

本文档说明如何在 Windows 和 Linux 上编译 v0 C++ 核心扩展。

---

## 环境准备

### Python 环境

```bash
conda create -n liuzhou python=3.13 -y
conda activate liuzhou
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 获取 PyTorch CMake 路径
python3 -c "import torch; print(torch.__version__); print(torch.utils.cmake_prefix_path)"
```

### 系统依赖

```bash
# Linux
conda install -c conda-forge cmake ninja pybind11 -y

# Windows: 安装 Visual Studio 2022 + MSVC + Ninja
```

---

## Windows (RTX 3060)

装MSVC，ninja

驱动和CUDA更新到和MSVC、torch匹配的版本，cmakelist写死路径。

打开 x64 Native Tools Command Prompt for VS 2022，输入

```bash
rmdir /S /Q build\v0
cmake -S v0 -B build/v0 -G "Ninja" -DUSE_CUDA=ON -DBUILD_CUDA_KERNELS=ON
cmake --build build\v0 --config Release
```

```powershell
$env:PYTHONPATH = "D:\CODES\liuzhou;D:\CODES\liuzhou\build\v0\src"
```

---

## Linux (H20)

建议使用 CMake + Make/Ninja，按需开启 CUDA（Hopper 架构 90）。

```bash
# 进入仓库根目录
cd liuzhou

export CUDACXX=/usr/local/cuda/bin/nvcc

PY=$CONDA_PREFIX/bin/python
cmake -S v0 -B v0/build \
  -G Ninja \
  -DUSE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DPython3_EXECUTABLE=$PY \
  -DPython3_INCLUDE_DIR=$CONDA_PREFIX/include/python3.13 \
  -DPython3_LIBRARY=$CONDA_PREFIX/lib/libpython3.13.so \
  -Dpybind11_DIR="$($PY -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$($PY -c 'import torch; print(torch.utils.cmake_prefix_path)')"

# 构建
cmake --build v0/build -j$(nproc)

# 运行 Python 时可选设置 PYTHONPATH 指向源码与生成的扩展
export PYTHONPATH=./liuzhou:./liuzhou/v0/build/src:$PYTHONPATH
```

注意：
- 请将 `/path/to/libtorch/share/cmake/Torch` 替换为实际 libtorch CMake 路径，或用 `-DCMAKE_PREFIX_PATH=/path/to/libtorch`。
- 如 CUDA 不在 `/usr/local/cuda`，调整 `-DCUDAToolkit_ROOT`。
- 如需清理重配：`rm -rf v0/build` 后重新运行 cmake。
