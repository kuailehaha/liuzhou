#!/bin/bash
# 验证迁移后的测试结构是否正常工作
# Usage: bash scripts/verify_migration.sh

set -e
cd "$(dirname "$0")/.."

echo "=========================================="
echo "验证迁移后的测试结构"
echo "=========================================="

echo ""
echo "[1/5] 验证 legacy 测试..."
python -m pytest tests/legacy/test_mcts.py -v --tb=short || echo "⚠️ legacy/test_mcts.py 失败"

echo ""
echo "[2/5] 验证 v0 state_batch 测试..."
python -m pytest tests/v0/test_state_batch.py -v --tb=short || echo "⚠️ v0/test_state_batch.py 失败"

echo ""
echo "[3/5] 验证 v0 mcts 测试..."
python -m pytest tests/v0/test_mcts.py -v --tb=short -k "valid_policy or consistency" || echo "⚠️ v0/test_mcts.py 失败"

echo ""
echo "[4/5] 验证 v0 actions 测试..."
python -m pytest tests/v0/test_actions.py::test_v0_actions_match_legacy -v --tb=short || echo "⚠️ v0/test_actions.py 失败"

echo ""
echo "[5/5] 验证 integration 测试..."
python -m pytest tests/integration/test_self_play.py -v --tb=short || echo "⚠️ integration/test_self_play.py 失败"

echo ""
echo "=========================================="
echo "验证完成！如果所有测试通过，可以运行清理脚本："
echo "  bash scripts/cleanup_after_migration.sh"
echo "=========================================="

