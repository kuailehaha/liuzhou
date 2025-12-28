#!/bin/bash
# H20 大规模训练脚本
# 用法: ./scripts/large_scale_train.sh [iterations]

set -e
cd /home/ubuntu/.cache/liuzhou
export PYTHONPATH=/home/ubuntu/.cache/liuzhou:/home/ubuntu/.cache/liuzhou/v0/build/src:$PYTHONPATH

# ==================== 配置 ====================
ITERATIONS=${1:-100}           # 总训练迭代次数

# 自博弈配置 (H20 优化)
NUM_PARALLEL=12                 # 12个并行生成服务
GAMES_PER_SERVICE=32           # 每服务32局 → 每迭代128局
MCTS_SIMS=800                  # MCTS模拟次数
BATCH_LEAVES=512               # H20显存大，可以设512

# 训练配置
TRAIN_EPOCHS=10
TRAIN_BATCH_SIZE=128           # 大batch训练
TRAIN_LR=0.001
EVAL_GAMES=50                  # 评估对局数

# 路径
DATA_DIR="./v0/data/self_play"
CHECKPOINT_DIR="./checkpoints_v0"
DEVICE="cuda"

# ==================== 初始化 ====================
mkdir -p "$DATA_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p logs

echo "=========================================="
echo "H20 大规模训练 - 开始"
echo "=========================================="
echo "  总迭代次数: $ITERATIONS"
echo "  每迭代自博弈: $((NUM_PARALLEL * GAMES_PER_SERVICE)) 局"
echo "  预计总局数: $((ITERATIONS * NUM_PARALLEL * GAMES_PER_SERVICE)) 局"
echo ""
echo "  自博弈: ${NUM_PARALLEL}x${GAMES_PER_SERVICE} 并行"
echo "  MCTS: ${MCTS_SIMS} sims, batch_leaves=${BATCH_LEAVES}"
echo "  训练: ${TRAIN_EPOCHS} epochs, batch=${TRAIN_BATCH_SIZE}"
echo "=========================================="

TOTAL_START=$(date +%s)

for ((iter=1; iter<=ITERATIONS; iter++)); do
    echo ""
    echo "==================== 迭代 $iter / $ITERATIONS ===================="
    ITER_START=$(date +%s)
    
    # ========== 阶段1: 并行自博弈 ==========
    echo "[$(date +%H:%M:%S)] 阶段1: 并行自博弈 (${NUM_PARALLEL}x${GAMES_PER_SERVICE}局)..."
    
    BEST_MODEL="$CHECKPOINT_DIR/best_model.pt"
    if [ -f "$BEST_MODEL" ]; then
        MODEL_ARG="--model_checkpoint $BEST_MODEL"
    else
        MODEL_ARG=""
    fi
    
    PIDS=()
    for i in $(seq 1 $NUM_PARALLEL); do
        SEED=$((iter * 10000 + i * 1000))
        PREFIX="iter_$(printf '%04d' $iter)_svc_${i}"
        
        python -m v0.generate_data \
            --num_games $GAMES_PER_SERVICE \
            --mcts_simulations $MCTS_SIMS \
            --batch_leaves $BATCH_LEAVES \
            --device $DEVICE \
            --output_dir "$DATA_DIR" \
            --output_prefix "$PREFIX" \
            --base_seed $SEED \
            $MODEL_ARG \
            > "logs/gen_iter${iter}_svc${i}.log" 2>&1 &
        
        PIDS+=($!)
        sleep 1
    done
    
    # 等待所有生成完成
    for pid in "${PIDS[@]}"; do
        wait $pid
    done
    
    GEN_END=$(date +%s)
    echo "[$(date +%H:%M:%S)] 自博弈完成，耗时 $((GEN_END - ITER_START))秒"
    
    # ========== 阶段2: 训练 ==========
    echo "[$(date +%H:%M:%S)] 阶段2: 训练 (epochs=${TRAIN_EPOCHS})..."
    
    python -m v0.train \
        --data_files "$DATA_DIR"/*.jsonl \
        --iterations 1 \
        --epochs $TRAIN_EPOCHS \
        --batch_size $TRAIN_BATCH_SIZE \
        --lr $TRAIN_LR \
        --device $DEVICE \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --eval_games_vs_random $EVAL_GAMES \
        --eval_games_vs_best 0 \
        > "logs/train_iter${iter}.log" 2>&1
    
    ITER_END=$(date +%s)
    echo "[$(date +%H:%M:%S)] 迭代 $iter 完成，耗时 $((ITER_END - ITER_START))秒"
    
    # 统计
    DATA_COUNT=$(ls -1 "$DATA_DIR"/*.jsonl 2>/dev/null | wc -l)
    echo "  累积数据文件: $DATA_COUNT"
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "  总耗时: ${TOTAL_ELAPSED}秒 ($((TOTAL_ELAPSED / 3600))小时 $((TOTAL_ELAPSED % 3600 / 60))分钟)"
echo "  检查点: $CHECKPOINT_DIR"
echo "  数据: $DATA_DIR"
echo "=========================================="

