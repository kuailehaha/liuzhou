# 快速开始

本页面向第一次接触仓库的读者，目标是用最短路径回答三个问题：如何构建、如何训练、如何评估，以及如何启动人机对战。

## 1. 环境前提

### Linux 训练环境

- Python 环境：`/2023533024/users/zhangmq/condaenvs/naivetorch`
- 可用 GPU：4 x H20（通常通过 `CUDA_VISIBLE_DEVICES=0,1,2,3` 使用）
- 当前训练主线：`v1/`

### Windows 本地环境

- Conda 环境：`conda activate torchenv`
- 常见用途：本地功能验证、小规模回归、人机对战

## 2. 构建入口

正式构建入口是 `scripts/instruct.sh`。

它的职责是：

- 校验 Python / CUDA 路径
- 在需要时重建 `v0_core`
- 调用 `scripts/instruction.sh` 执行后续验证流程

典型 Linux 用法：

```bash
bash scripts/instruct.sh \
  --python-bin /2023533024/users/zhangmq/condaenvs/naivetorch/bin/python \
  --cuda-home /usr/local/cuda \
  --cuda-visible-devices 0 \
  --device cuda:0 \
  --build-v0 1
```

如果你已经固定好了环境变量，也可以直接执行底层脚本：

```bash
BUILD_V0=1 bash scripts/instruction.sh
```

## 3. 训练入口

训练主入口是 `scripts/big_train_v1.sh`，它负责 v1 的 staged 流程：`selfplay -> train -> eval -> infer`。

最小示例：

```bash
bash scripts/big_train_v1.sh
```

常见做法是先用较保守配置验证流程，再用默认的大规模配置放量：

```bash
PROFILE=stable bash scripts/big_train_v1.sh
```

训练脚本中已经集中管理了以下关键参数：

- 自博弈设备与训练设备
- `MCTS_SIMULATIONS`
- `SELF_PLAY_CONCURRENT_GAMES`
- 评估参数与 gating 逻辑
- staged 输出目录与 checkpoint 目录

## 4. 评估入口

### 单点评估

主入口：`scripts/eval_checkpoint.py`

典型示例：

```bash
python scripts/eval_checkpoint.py \
  --challenger_checkpoint checkpoints_v1_big_1/latest.pt \
  --backend v1 \
  --device cuda:0 \
  --eval_games_vs_random 200 \
  --mcts_simulations 1024 \
  --v1_concurrent_games 256
```

### 锦标赛评估

主入口：`scripts/tournament_v1_eval.py`

适用场景：

- 比较一批 checkpoint 的相对强度
- 生成淘汰赛或循环赛结果
- 配合 Elo / BT 排名做模型选择

## 5. 启动人机对战

后端位于 `backend/main.py`，Web 界面位于 `web_ui/`，后端启动后会把静态资源挂载到 `/ui/`。

典型开发方式：

```bash
uvicorn backend.main:app --reload
```

启动后访问：

```text
http://localhost:8000/ui/index.html
```

常用环境变量：

- `MODEL_PATH`：默认加载的模型路径
- `MODEL_DEVICE`：模型运行设备
- `MCTS_SIMULATIONS`：默认搜索次数
- `MCTS_TEMPERATURE`：默认温度

## 6. 推荐阅读顺序

- 想理解系统结构：继续看 [架构总览](./architecture.md)
- 想理解训练方法：继续看 [方法说明](./method.md)
- 想确认规则：直接看 [规则说明](./rules.md)
- 想沿着研究问题继续深挖：参见 [高难问题清单](./faq.md)
