# 架构总览

项目由五个子系统组成：规则系统、搜索系统、训练系统、评估系统、人机对战系统。

## 1. 系统拆分

### 规则系统

职责：定义六洲棋状态、阶段流转、合法动作与动作应用。

核心位置：

- `src/game_state.py`
- `src/rule_engine.py`
- `src/move_generator.py`
- `v0/src/rules/`
- `v0/src/moves/`

### 搜索系统

职责：基于规则系统与神经网络输出执行 MCTS。

核心位置：

- `src/mcts.py`
- `v0/src/mcts/`
- `v1/python/mcts_gpu.py`

### 训练系统

职责：执行自博弈、生成训练样本、训练模型、管理 staged 流程。

核心位置：

- `v1/train.py`
- `v1/python/self_play_gpu_runner.py`
- `v1/python/self_play_worker.py`
- `v1/python/train_bridge.py`
- `v1/python/trajectory_buffer.py`
- `scripts/big_train_v1.sh`

### 评估系统

职责：对 checkpoint 做 `vs_random`、`vs_previous` 与锦标赛评估。

核心位置：

- `scripts/eval_checkpoint.py`
- `scripts/tournament_v1_eval.py`
- `src/evaluate.py`

### 人机对战系统

职责：把模型推理能力暴露给前端界面，支持玩家与 AI 对局。

核心位置：

- `backend/`
- `web_ui/`

## 2. 目录结构

### `src/`

Legacy 纯 Python 实现，当前职责：

- 规则参考实现
- 功能验证
- 部分评估与 UI 路径的基础组件

### `v0/`

C++/CUDA 高性能实现，为 v1 提供底层能力：

- `v0_core` 扩展
- 动作编码与规则核心
- 原生 MCTS 与张量状态批处理能力

### `v1/`

当前训练主线。在已有规则与 CUDA 能力上，把自博弈到训练的数据流重写为 staged pipeline。

## 3. 三代实现的演进关系

### Legacy

- 用纯 Python 搭出完整规则与训练闭环
- 主要解决“先让系统工作起来”的问题
- 适合功能验证，不适合大规模训练

### v0

- 以 C++/CUDA 重写关键能力
- 把性能瓶颈从 Python 规则/动作处理推进到更靠近推理与调度的层面
- 同时保留 Python 参考实现便于对拍与验证

### v1

- 聚焦训练主线，而不是做一套完全独立的新规则系统
- 目标是把 self-play、tensor 输出、训练桥接、评估入口连成统一流水线
- 当前主线

## 4. 一次完整训练迭代的高层数据流

1. `scripts/big_train_v1.sh` 调度 staged 流程
2. `v1/train.py` 组织 self-play / train / eval / infer 阶段
3. `self_play_gpu_runner.py` 批量生成张量化对局轨迹
4. `train_bridge.py` 把张量样本喂给模型训练
5. `eval_checkpoint.py` / `tournament_v1_eval.py` 评估新旧模型
6. 结果写回 checkpoint、日志与阶段目录

## 5. 文档与实现的分工

- 项目总说明：本 `docs/` 文档树
- 深设计归档：`v1/Design.md`
- 规则权威：`docs/rules.md`
- 入口导航：根目录 `README.md`
