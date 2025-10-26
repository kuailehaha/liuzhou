# TODOS

- [x] 重构逻辑：将形成square和line后capture等action拆分为原子化的move+capture action
- [x] batch推理叶子节点
- [x] 多进程推理，添加numworkers功能
- [x] 策略头调整，完全对应各策略头语义
- [x] 修复 no_legal_moves 的异常
- [x] 实现 RandomAgent 的对局，并获取胜率数据
- [ ] 张量化实现自博弈过程
- [ ] 合并参数：self_play_games、self_play_games_per_worker
- [ ] 并行化DDP训练或实现“自对弈集群+训练服务器”模式
- [ ] 确认代码实现逻辑

