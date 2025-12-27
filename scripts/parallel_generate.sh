#!/bin/bash
# 并行生成自博弈数据 - 多服务模式
# 用法: ./scripts/parallel_generate.sh [num_parallel] [games_per_service]

set -e
cd /home/ubuntu/.cache/liuzhou
export PYTHONPATH=/home/ubuntu/.cache/liuzhou:/home/ubuntu/.cache/liuzhou/v0/build/src:$PYTHONPATH

# ==================== 配置 ====================
NUM_PARALLEL=${1:-4}           # 并行服务数（默认4个）
GAMES_PER_SERVICE=${2:-32}     # 每个服务生成的局数
MCTS_SIMS=800                  # MCTS模拟次数
BATCH_LEAVES=512               # 批量叶子节点（H20显存充足，可以设大）
DEVICE="cuda"
DATA_DIR="./v0/data/self_play"
CHECKPOINT_DIR="./checkpoints_v0"

# ==================== 初始化 ====================
mkdir -p "$DATA_DIR"
mkdir -p logs

# 获取模型参数
BEST_MODEL="$CHECKPOINT_DIR/best_model.pt"
if [ -f "$BEST_MODEL" ]; then
    MODEL_ARG="--model_checkpoint $BEST_MODEL"
    echo "使用模型: $BEST_MODEL"
else
    MODEL_ARG=""
    echo "使用随机初始化模型"
fi

echo "=========================================="
echo "并行生成配置"
echo "  并行服务数: $NUM_PARALLEL"
echo "  每服务局数: $GAMES_PER_SERVICE"
echo "  总局数: $((NUM_PARALLEL * GAMES_PER_SERVICE))"
echo "  MCTS模拟: $MCTS_SIMS"
echo "  Batch Leaves: $BATCH_LEAVES"
echo "=========================================="

# ==================== 并行启动 ====================
PIDS=()
START_TIME=$(date +%s)

for i in $(seq 1 $NUM_PARALLEL); do
    SEED=$((42 + i * 1000000))  # 不同种子避免重复
    PREFIX="gen_${i}"
    LOG_FILE="logs/generate_${i}.log"
    
    echo "启动服务 $i (seed=$SEED) -> $LOG_FILE"
    
    python -m v0.generate_data \
        --num_games $GAMES_PER_SERVICE \
        --mcts_simulations $MCTS_SIMS \
        --batch_leaves $BATCH_LEAVES \
        --device $DEVICE \
        --output_dir "$DATA_DIR" \
        --output_prefix "$PREFIX" \
        --base_seed $SEED \
        $MODEL_ARG \
        > "$LOG_FILE" 2>&1 &
    
    PIDS+=($!)
    
    # 错开启动，避免CUDA初始化冲突
    sleep 2
done

echo ""
echo "所有服务已启动，等待完成..."
echo "PIDs: ${PIDS[*]}"

# ==================== 等待完成 ====================
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait $pid; then
        echo "进程 $pid 失败"
        FAILED=$((FAILED + 1))
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "✅ 全部完成！"
else
    echo "⚠️ $FAILED 个服务失败"
fi
echo "总耗时: ${ELAPSED}秒 ($((ELAPSED / 60))分钟)"
echo "数据目录: $DATA_DIR"
echo "=========================================="

# 显示生成的文件
ls -la "$DATA_DIR"/*.jsonl 2>/dev/null | tail -10

