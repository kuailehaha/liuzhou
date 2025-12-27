#!/bin/bash
# V0 训练调度脚本：自博弈累积 + 统一训练
# 用法: ./scripts/train_loop.sh [iterations]

set -e

# ==================== 配置参数 ====================
ITERATIONS=${1:-10}                    # 总迭代次数
GAMES_PER_ITER=64                      # 每次迭代自博弈局数
MCTS_SIMS=800                          # MCTS模拟次数
BATCH_LEAVES=256                       # 批量叶子节点数
DEVICE="cuda"                          # 推理设备
DATA_DIR="./v0/data/self_play"         # 数据存储目录
CHECKPOINT_DIR="./checkpoints_v0"      # 检查点目录

# 训练参数
TRAIN_EPOCHS=10
TRAIN_BATCH_SIZE=64
TRAIN_LR=0.001

# ==================== 初始化 ====================
mkdir -p "$DATA_DIR"
mkdir -p "$CHECKPOINT_DIR"

echo "=========================================="
echo "V0 训练循环开始"
echo "总迭代次数: $ITERATIONS"
echo "每次自博弈: $GAMES_PER_ITER 局"
echo "MCTS模拟: $MCTS_SIMS 次/步"
echo "Batch Leaves: $BATCH_LEAVES"
echo "=========================================="

# ==================== 主循环 ====================
for ((i=1; i<=ITERATIONS; i++)); do
    echo ""
    echo "========== 迭代 $i / $ITERATIONS =========="
    
    # 获取当前最佳模型（如果存在）
    BEST_MODEL="$CHECKPOINT_DIR/best_model.pt"
    if [ -f "$BEST_MODEL" ]; then
        MODEL_ARG="--model_checkpoint $BEST_MODEL"
        echo "[自博弈] 使用模型: $BEST_MODEL"
    else
        MODEL_ARG=""
        echo "[自博弈] 使用随机初始化模型"
    fi
    
    # 阶段1: 自博弈生成数据
    echo "[自博弈] 生成 $GAMES_PER_ITER 局棋局..."
    START_TIME=$(date +%s)
    
    python -m v0.generate_data \
        --num_games $GAMES_PER_ITER \
        --mcts_simulations $MCTS_SIMS \
        --batch_leaves $BATCH_LEAVES \
        --device $DEVICE \
        --output_dir "$DATA_DIR" \
        --output_prefix "iter_$(printf '%04d' $i)" \
        $MODEL_ARG
    
    END_TIME=$(date +%s)
    echo "[自博弈] 完成，耗时 $((END_TIME - START_TIME)) 秒"
    
    # 统计当前数据量
    DATA_FILES=$(ls -1 "$DATA_DIR"/*.jsonl 2>/dev/null | wc -l)
    echo "[数据] 当前累积 $DATA_FILES 个数据文件"
    
    # 阶段2: 训练（使用所有累积数据）
    echo "[训练] 开始训练..."
    START_TIME=$(date +%s)
    
    python -m v0.train \
        --data_files "$DATA_DIR"/*.jsonl \
        --epochs $TRAIN_EPOCHS \
        --batch_size $TRAIN_BATCH_SIZE \
        --lr $TRAIN_LR \
        --device $DEVICE \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --iterations 1 \
        --eval_games_vs_random 20 \
        --eval_games_vs_best 0
    
    END_TIME=$(date +%s)
    echo "[训练] 完成，耗时 $((END_TIME - START_TIME)) 秒"
    
done

echo ""
echo "=========================================="
echo "训练完成！"
echo "检查点保存在: $CHECKPOINT_DIR"
echo "数据保存在: $DATA_DIR"
echo "=========================================="

