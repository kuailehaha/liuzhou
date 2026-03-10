# 基于 MCTS 的强化学习六洲棋 AI 系统实现

本项目聚焦传统棋类“六洲棋”的智能对弈系统，覆盖规则建模、MCTS 搜索、强化学习训练与人机对战实现。

当前主线是 `v1/`。如果你第一次打开这个仓库，建议从 `docs/` 文档树开始，而不是直接从历史实现细节倒推。

## 文档导航

- [项目文档首页](./docs/index.md)
- [快速开始](./docs/quickstart.md)
- [架构总览](./docs/architecture.md)
- [方法说明](./docs/method.md)
- [人机对战系统](./docs/gameplay_system.md)
- [规则说明](./docs/rules.md)
- [项目结果](./docs/results.md)
- [高难问题清单](./docs/faq.md)

## 核心入口

- 构建 / 编译：`scripts/instruct.sh`
  - 该脚本会调用 `scripts/instruction.sh`
- 训练：`scripts/big_train_v1.sh`
- 统一训练入口：`scripts/train_entry.py --pipeline v1`
- 单点评估：`scripts/eval_checkpoint.py --backend v1`
- 锦标赛评估：`scripts/tournament_v1_eval.py`
- 人机对战后端：`backend/main.py`
- Web UI：`web_ui/`

## 项目概览

- 六洲棋 AI 系统：规则引擎 + MCTS + 强化学习训练 + 人机对战前后端
- 三层实现：`src/`（Legacy 参考实现）、`v0/`（C++/CUDA 核心）、`v1/`（当前训练主线）
- 当前重点：在保持规则一致性的前提下，提高训练主线的强度转化效率，而不只是继续追求吞吐

## 推荐阅读顺序

1. [docs/index.md](./docs/index.md)
2. [docs/quickstart.md](./docs/quickstart.md)
3. [docs/method.md](./docs/method.md)
4. [docs/rules.md](./docs/rules.md)
5. [docs/results.md](./docs/results.md)

## 说明

- 正式规则文档以 [docs/rules.md](./docs/rules.md) 为准
- 根目录 [rule_description.md](./rule_description.md) 保留为兼容入口
- `v1/Design.md` 作为深设计与阶段记录归档保留，不再作为项目主说明

任何建议或帮助，请联系[我](mailto:kuailepapa@gmail.com)。
