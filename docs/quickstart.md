# 快速开始

## 1. 环境前提

### Linux 训练环境

- Python 环境：`/2023533024/users/zhangmq/condaenvs/naivetorch`
- 可用 GPU：4 x H20（通常通过 `CUDA_VISIBLE_DEVICES=0,1,2,3` 使用）
- 当前训练主线：`v1/`

### Windows 本地环境

- Conda 环境：`conda activate torchenv`
- 常见用途：本地功能验证、小规模回归、人机对战

### macOS Apple Silicon portable 环境

无需构建 `v0_core` 或安装 CUDA。`portable` 后端把规则和完整 MCTS 树留在 CPU，并在可用时使用 MPS 做 PyTorch 批量推理与 float32 训练；现有 CUDA 默认路径不受影响。

一键小规模闭环：

```bash
python tools/smoke_v1_portable.py \
  --device auto \
  --work_dir tmp/v1_portable_smoke
```

`auto` 在 MPS 可用时选择 `mps`，否则选择 CPU 并在报告中记录原因。不要设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`；portable MPS 会拒绝该静默回退模式。

固定 checkpoint 的 portable 评估：

```bash
python scripts/eval_checkpoint.py \
  --challenger_checkpoint tmp/v1_portable_smoke/model_iter_001.pt \
  --backend portable \
  --device auto \
  --eval_workers 1 \
  --eval_games_vs_random 2 \
  --mcts_simulations 16
```

固定模型 Root-PUCT/full-MCTS 对比入口为 `tools/ab_portable_search.py`。在模型尚未对 random 饱和、高级模型互战平局率过高时，可以把固定条件的 `vs_random win-loss` 作为较长阶段的粗粒度趋势/筛选指标，但应同时报告 draw rate；接近饱和或需要最终棋力结论时，仍以固定 checkpoint 对战或 tournament/Elo 为准。

#### Apple M5 约 20 小时可恢复长训

本机冻结配置由 `scripts/long_train_portable_mps.py` 管理：self-play 为 128 games/concurrency 128，训练为 batch 256、3 epochs、replay window 4，搜索为 8 simulations；每 10 轮分别进行 500 局 RandomAgent 评估和 500 局 candidate-versus-incumbent 自评估，评估 concurrency 为 64。评估 seed、250/250 黑白分配、逐颜色 W/L/D、模型/optimizer SHA 和 replay 输入都会落盘。

8/16/32/64 simulations 的本机同条件测试显示 self-play 墙钟成本约为 `1.00x/2.07x/3.88x/6.93x`。同 seed 500 局 RandomAgent 对照中，64 simulations 相对 8 simulations 仅从 `433/500` 提高到 `449/500`，却耗时 `6.91x`，且高 simulations 自博弈明显更偏和棋。因此 20 小时默认仍为 8 simulations；不要依据 100 局小样本中偶然出现的 `98%` 改成 64。

合盖运行前必须接交流电、外接显示器和外接键盘/鼠标。`caffeinate` 只保证其子进程存活期间抑制 idle/system/disk sleep，不能代替 Apple silicon 的合盖硬件条件；命令中的 `--require-external-display` 会在没有外接显示器时直接失败。

从已验收的一小时 checkpoint 继续训练并在观测、独立复验均达到 `495/500` 后提前停止：

```bash
mkdir -p logs && nohup zsh scripts/run_long_train_mps.sh \
  --run-dir tmp/v1_portable_long_20h \
  --hours 20 --resume \
  --initial-checkpoint tmp/v1_portable_goal_20260722/formal_1h/model_iter_117.pt \
  --initial-optimizer-state tmp/v1_portable_goal_20260722/formal_1h/optimizer_state.pt \
  --require-external-display --stop-on-target \
  >> logs/portable_mps_20h.log 2>&1 & echo $!
```

`--resume` 对新目录和已有目录都可用；已有目录会核对冻结配置、外层 iteration、current checkpoint、optimizer 和 commit SHA 后续跑。运行状态位于 `tmp/v1_portable_long_20h/state.json`，最终摘要位于 `final_summary.json`，`best_model.pt` 和 `best_vs_random.pt` 位于其 `checkpoints/` 子目录。

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

## 6. 更多文档

- [架构总览](./architecture.md) — 系统结构与三代演进
- [方法说明](./method.md) — 网络设计、MCTS、训练目标
- [规则说明](./rules.md) — 完整六洲棋规则
- [高难问题清单](./faq.md) — 深层技术问题
