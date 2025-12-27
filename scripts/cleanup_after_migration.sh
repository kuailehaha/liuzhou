#!/bin/bash
# 清理迁移后不再需要的文件和目录
# Usage: bash scripts/cleanup_after_migration.sh
#
# ⚠️ 请确保先运行 verify_migration.sh 验证测试通过后再执行此脚本

set -e
cd "$(dirname "$0")/.."

echo "=========================================="
echo "清理迁移后不再需要的文件"
echo "=========================================="

# 1. 删除 v1 目录
if [ -d "v1" ]; then
    echo "[1] 删除 v1/ 目录..."
    rm -rf v1/
    echo "    ✓ v1/ 已删除"
else
    echo "[1] v1/ 目录不存在，跳过"
fi

# 2. 删除 tests/v1 目录
if [ -d "tests/v1" ]; then
    echo "[2] 删除 tests/v1/ 目录..."
    rm -rf tests/v1/
    echo "    ✓ tests/v1/ 已删除"
else
    echo "[2] tests/v1/ 目录不存在，跳过"
fi

# 3. 删除重复的 tests 目录
echo "[3] 删除重复的测试目录..."
rm -rf tests/unit/
rm -rf tests/accuracy/
rm -rf tests/cuda/
rm -rf tests/cross_check/
echo "    ✓ 重复目录已删除"

# 4. 删除根目录下的旧测试文件
echo "[4] 删除根目录下的旧测试文件..."
rm -f tests/test_mcts.py
rm -f tests/test_rule_engine.py
echo "    ✓ 旧测试文件已删除"

# 5. 删除依赖 v1 的 tools 文件
echo "[5] 删除依赖 v1 的 tools 文件..."
rm -f tools/cross_check_mcts.py
rm -f tools/cross_check_policy_projection.py
rm -f tools/benchmark_apply_moves.py
rm -f tools/benchmark_policy_projection.py
rm -f tools/benchmark_legal_mask.py
rm -f tools/verify_v0_mcts.py
rm -f tools/verify_v0_state_batch.py
rm -f tools/check_v0_actions_all.py
echo "    ✓ 依赖 v1 的 tools 文件已删除"

# 6. 删除根目录下的临时文件
echo "[6] 删除根目录下的临时文件..."
rm -f tmp_encoding_compare.py
rm -f tmp_mcts_debug.py
rm -f tmp_policy_compare.py
rm -f tmp_sim_compare.py
echo "    ✓ 临时文件已删除"

# 7. 删除 v0/tests 和 v0/tools (已迁移)
echo "[7] 删除 v0/tests 和 v0/tools (已迁移到根目录)..."
rm -rf v0/tests/
rm -rf v0/tools/
echo "    ✓ v0 子目录已删除"

echo ""
echo "=========================================="
echo "清理完成！最终目录结构："
echo "=========================================="
echo ""
echo "tests/"
echo "├── legacy/           # legacy 实现测试"
echo "│   ├── test_mcts.py"
echo "│   └── test_self_play.py"
echo "├── v0/               # v0 实现测试"
echo "│   ├── test_actions.py"
echo "│   ├── test_state_batch.py"
echo "│   ├── test_mcts.py"
echo "│   └── cuda/"
echo "│       ├── test_fast_apply_moves_cuda.py"
echo "│       └── test_fast_legal_mask_cuda.py"
echo "├── integration/      # 集成测试"
echo "│   └── test_self_play.py"
echo "└── random_agent/     # 随机智能体测试"
echo ""
echo "tools/"
echo "├── benchmark_mcts.py"
echo "├── benchmark_self_play.py"
echo "├── benchmark_cuda.py"
echo "├── run_test_matrix.py"
echo "└── get_thread.py"
echo ""

