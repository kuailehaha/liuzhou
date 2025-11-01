# TODOS

TODO以产生时间为准。

### 2025.10.20

- [x] 重构逻辑：将形成square和line后capture等action拆分为原子化的move+capture action

### 2025.10.22

- [x] batch推理叶子节点
- [x] 调整log verbose
- [x] 策略头调整，对应各策略头语义
- [x] 试跑，时间分析
- [x] 分析强化学习原理

### 2025.10.23

- [x] 添加batch_K：自博弈和eval阶段批量推理不同的节点 

### 2025.10.25

- [x] 预留虚损

### 2025.10.26
- [x] 置换策略头 12 在 move动作上的功能 
- [x] 修复 no_legal_moves 的异常
- [x] 多进程推理，添加numworkers功能
- [x] 实现 RandomAgent 的对局，并获取胜率数据
- [x] 合并参数：self_play_games、self_play_games_per_worker
- [x] 整理test file
- [ ] 张量化实现自博弈过程
- [ ] 并行化DDP训练或实现“自对弈集群+训练服务器”模式
- [ ] 确认代码实现逻辑
- [ ] 模型结构和超参调优
- [ ] 对场上棋子实现reward并检查效果（新开分支？感觉不用，可以开个选项。【感觉实现张量化自博弈之前很有必要添加啊！】）

### 2025.10.27

- [x] 实现前端，人机对弈功能
- [x] 修补bug：同时square和line只能提掉一个

### 2025.10.28

- [x] eval log添加 胜负和 信息
- [x] 前端：添加敌方上一步信息

### 2025.10.29

- [x] 添加前端对弈value值显示（胜率信息）

### 2025.10.30

- [x] 创建 v1 张量化重写骨架目录
- [x] 实现 v1/game 张量化状态转换与规则逻辑
- [x] 完成 v1/game/move_encoder.py 动作编码与解码
- [x] 写出 VectorizedMCTS 主流程并替换自博弈调度
- [ ] 打通 v1/self_play -> v1/train 张量化训练闭环
- [x] 为 v1 分支新增测试与对照验证脚本

### 2025.10.31

- [ ] 对拍传统管线和新管线（MCTS部分）

### 2025.11.1

- [x] 梳理 legacy 流程关键函数并整理 inputs/outputs 对照表
- [x] 设计并实现 states_to_model_input 及 _board_planes/_mark_planes/_phase_planes
- [x] B=1 parity test matches legacy state encoding
- [x] 复核 ActionEncodingSpec/encode/decode 索引约定并补齐批量掩码 helper
- [x] 在 project_policy_logits 中完成 logits 拆分、掩码与兜底归一化逻辑
- [x] 如有需要新增 batched gather/reshape 等张量工具函数
- [x] 编写 cross-check 脚本对比两套管线的输入张量与动作分布
- [x] 更新自博弈/MCTS/训练脚本调用路径切换至 v1 张量化实现
- [ ] README/TODO 记录切换方式与验证状态
- [ ] Future: unify policy heads (head1=place/move-to, head2=move-from, head3=mark+all removals) and drop phase one-hot channels when legacy parity is secured.

