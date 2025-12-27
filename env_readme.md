conda create -n testenv python=3.13 -y
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 拿一下位置
python3 -c "import torch; print(torch.__version__); print(torch.utils.cmake_prefix_path)"
2.8.0+cu128
/home/ubuntu/miniconda3/envs/testenv/lib/python3.13/site-packages/torch/share/cmake

sudo apt install cmake
sudo apt install -y ninja-build
sudo apt install -y pybind11-dev

