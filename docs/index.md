# 项目文档首页

本项目是一个基于 MCTS 与强化学习的六洲棋 AI 系统，覆盖规则建模、自博弈训练、模型评估，以及人机对战前后端。

当前主线是 `v1/`。如果你的目标是理解“项目现在如何工作”，应优先阅读 v1 相关文档，而不是从 legacy 或 v0 的历史实现开始。

## 你可以从这里开始

- 想快速上手运行：参见 [快速开始](./quickstart.md)
- 想了解项目由哪些系统组成：参见 [架构总览](./architecture.md)
- 想理解训练方法与模型设计：参见 [方法说明](./method.md)
- 想直接看规则：参见 [规则说明](./rules.md)
- 想了解阶段成果与对外口径：参见 [项目结果](./results.md)
- 想沿着难问题反推系统理解：参见 [高难问题清单](./faq.md)

## 当前主线

- 训练主入口：`scripts/big_train_v1.sh`
- 构建主入口：`scripts/instruct.sh`
- 单点评估：`scripts/eval_checkpoint.py`
- 锦标赛评估：`scripts/tournament_v1_eval.py`
- 人机对战后端：`backend/main.py`
- Web UI：`web_ui/`

## 推荐阅读顺序

1. [快速开始](./quickstart.md)
2. [架构总览](./architecture.md)
3. [方法说明](./method.md)
4. [规则说明](./rules.md)
5. [项目结果](./results.md)
6. [高难问题清单](./faq.md)

## 阅读提示

- `src/`、`v0/`、`v1/` 对应三代实现，但文档主叙事聚焦 `v1/`。
- `v1/Design.md` 仍保留，作为深设计与阶段记录的归档材料；它不是项目总说明的入口。
- 根目录 `rule_description.md` 已降级为兼容入口，正式规则文档以 [docs/rules.md](./rules.md) 为准。
