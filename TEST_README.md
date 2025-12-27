# Test Matrix (Legacy & v0)

本文档描述项目的测试组织结构和运行方式。

---

## 目录结构

```
tests/
├── legacy/           # legacy (纯 Python) 实现测试
│   ├── test_mcts.py          # MCTS 单元测试
│   └── test_self_play.py     # 自我对弈冒烟测试
│
├── v0/               # v0 (C++ core) 实现测试
│   ├── test_actions.py       # action 编码正确性
│   ├── test_state_batch.py   # state batch 往返正确性
│   ├── test_mcts.py          # v0 MCTS 正确性
│   └── cuda/                 # CUDA 相关测试
│       ├── test_fast_apply_moves_cuda.py
│       └── test_fast_legal_mask_cuda.py
│
├── integration/      # 集成测试
│   └── test_self_play.py
│
└── random_agent/     # 随机智能体测试
    └── ...

tools/
├── benchmark_mcts.py         # MCTS 性能基准
├── benchmark_self_play.py    # 自我对弈性能基准
├── benchmark_cuda.py         # CUDA 设备对比
├── run_test_matrix.py        # 批量运行测试
└── get_thread.py             # 线程信息工具
```

---

## 快速开始

### 运行所有测试

```bash
# 运行 legacy 测试
pytest tests/legacy/ -v

# 运行 v0 测试
pytest tests/v0/ -v

# 运行 v0 CUDA 测试 (需要 CUDA)
pytest tests/v0/cuda/ -v

# 运行集成测试
pytest tests/integration/ -v

# 运行所有测试
pytest tests/ -v
```

### 使用测试矩阵运行器

```bash
# 运行所有测试组
python -m tools.run_test_matrix

# 只运行特定组
python -m tools.run_test_matrix --group legacy --group v0

# 预览命令但不执行
python -m tools.run_test_matrix --dry-run

# 失败时立即停止
python -m tools.run_test_matrix --fail-fast
```

---

## 测试分组

| 组名 | 说明 | 测试文件 |
|------|------|----------|
| `legacy` | Legacy Python 实现测试 | `tests/legacy/test_*.py` |
| `v0` | v0 C++ core 实现测试 | `tests/v0/test_*.py` |
| `v0_cuda` | v0 CUDA 特定测试 | `tests/v0/cuda/test_*.py` |
| `integration` | 集成测试 | `tests/integration/test_*.py` |
| `random_agent` | 随机智能体测试 | `tests/random_agent/` |
| `benchmark` | 性能基准 (CLI) | `tools/benchmark_*.py` |

---

## 性能基准

```bash
# MCTS 性能对比 (legacy vs v0)
python -m tools.benchmark_mcts --samples 10 --sims 128 --device cpu

# 仅测试 v0 性能
python -m tools.benchmark_mcts --samples 10 --sims 128 --skip-legacy

# 自我对弈性能
python -m tools.benchmark_self_play --num-games 4 --mcts-simulations 64

# CUDA vs CPU 对比
python -m tools.benchmark_cuda --samples 10 --sims 128 --devices cpu,cuda
```

---

## 约定

| 项目 | 约定 |
|------|------|
| 随机种子 (accuracy) | `rng = random.Random(0xF00DCAFE)` + `torch.manual_seed(...)` |
| 状态采样 (accuracy) | `num_states = 400`, `max_random_moves = 160` |
| 设备 | 默认 `cpu`，CUDA 测试会自动跳过如果不可用 |
| 慢速测试 | 使用 `@pytest.mark.slow` 标记，可通过 `-m "not slow"` 排除 |

---

## 结果日志

- 测试日志: `tests/result/*.txt`
- 基准日志: `tools/result/*.txt`

---

## 添加新测试

1. **Legacy 测试**: 放入 `tests/legacy/test_<name>.py`
2. **v0 测试**: 放入 `tests/v0/test_<name>.py`
3. **v0 CUDA 测试**: 放入 `tests/v0/cuda/test_<name>.py`
4. **性能基准**: 放入 `tools/benchmark_<name>.py`
5. 更新 `tools/run_test_matrix.py` 中的 `TEST_GROUPS`
