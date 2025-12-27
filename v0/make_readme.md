windows RTX 3060

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


linux H20

建议使用 CMake + Make/Ninja，按需开启 CUDA（Hopper 架构 90）。

```bash
# 进入仓库根目录
cd /home/ubuntu/.cache/liuzhou

# 纯 CPU
cmake -S v0 -B v0/build \
  -DUSE_CUDA=OFF \
  -DPython3_EXECUTABLE=$(which python3) \
  -DTorch_DIR=/path/to/libtorch/share/cmake/Torch

# 或 启用 CUDA（H20/Hopper）
cmake -S v0 -B v0/build \
  -G Ninja \
  -DUSE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DPython3_EXECUTABLE=$(which python3) \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"


# 构建
cmake --build v0/build -j$(nproc)

# 运行 Python 时可选设置 PYTHONPATH 指向源码与生成的扩展
export PYTHONPATH=/home/ubuntu/.cache/liuzhou:/home/ubuntu/.cache/liuzhou/v0/build/src:$PYTHONPATH
```

注意：
- 请将 `/path/to/libtorch/share/cmake/Torch` 替换为实际 libtorch CMake 路径，或用 `-DCMAKE_PREFIX_PATH=/path/to/libtorch`。
- 如 CUDA 不在 `/usr/local/cuda`，调整 `-DCUDAToolkit_ROOT`。
- 如需清理重配：`rm -rf v0/build` 后重新运行 cmake。
