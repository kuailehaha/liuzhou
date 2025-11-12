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