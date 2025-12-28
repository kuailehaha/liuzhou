#!/bin/bash
# 优化版大规模训练脚本
# 解决：1) I/O过载  2) 小kernel碎片
# 用法: nohup ./scripts/optimized_train.sh 50 > train_output.log 2>&1 &

set -e
cd /home/ubuntu/.cache/liuzhou
export PYTHONPATH=/home/ubuntu/.cache/liuzhou:/home/ubuntu/.cache/liuzhou/v0/build/src:$PYTHONPATH

# ==================== 配置 ====================
ITERATIONS=${1:-100}

# 【优化1】减少并行数，避免I/O争抢
# H20单卡，2-4个并行就够了，更多会互相争抢GPU context
NUM_PARALLEL=4

# 【优化2】每服务生成更多局，减少进程启动开销
GAMES_PER_SERVICE=64           # 4x64=256局/迭代

# 【优化3】增大batch_leaves，减少kernel调用次数
# H20有97GB显存，完全可以开到512甚至更大
MCTS_SIMS=800
BATCH_LEAVES=512              

# 训练配置
TRAIN_EPOCHS=10
TRAIN_BATCH_SIZE=256           # 增大训练batch
TRAIN_LR=0.001
EVAL_GAMES=50

# 【优化4】使用tmpfs作为临时写入目录，最后再复制到磁盘
# 如果/dev/shm空间不够，可以用普通目录但减少写入频率
TMPFS_DIR="/dev/shm/liuzhou_data"
DATA_DIR="./v0/data/self_play"
CHECKPOINT_DIR="./checkpoints_v0"
DEVICE="cuda"

# ==================== 初始化 ====================
mkdir -p "$DATA_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$TMPFS_DIR"
mkdir -p logs

echo "=========================================="
echo "优化版训练 - 开始"
echo "=========================================="
echo "  总迭代次数: $ITERATIONS"
echo "  并行服务: $NUM_PARALLEL (减少I/O争抢)"
echo "  每服务局数: $GAMES_PER_SERVICE"
echo "  batch_leaves: $BATCH_LEAVES (减少小kernel)"
echo "  临时目录: $TMPFS_DIR (内存写入)"
echo "=========================================="

TOTAL_START=$(date +%s)

for ((iter=1; iter<=ITERATIONS; iter++)); do
    echo ""
    echo "==================== 迭代 $iter / $ITERATIONS ===================="
    ITER_START=$(date +%s)
    
    # ========== 阶段1: 自博弈 (写入tmpfs) ==========
    echo "[$(date +%H:%M:%S)] 阶段1: 自博弈 → tmpfs..."
    
    BEST_MODEL="$CHECKPOINT_DIR/best_model.pt"
    if [ -f "$BEST_MODEL" ]; then
        MODEL_ARG="--model_checkpoint $BEST_MODEL"
    else
        MODEL_ARG=""
    fi
    
    # 清空tmpfs
    rm -f "$TMPFS_DIR"/*.jsonl "$TMPFS_DIR"/*.json 2>/dev/null || true
    
    PIDS=()
    for i in $(seq 1 $NUM_PARALLEL); do
        SEED=$((iter * 10000 + i * 1000))
        PREFIX="iter_$(printf '%04d' $iter)_svc_${i}"
        
        python -m v0.generate_data \
            --num_games $GAMES_PER_SERVICE \
            --mcts_simulations $MCTS_SIMS \
            --batch_leaves $BATCH_LEAVES \
            --device $DEVICE \
            --output_dir "$TMPFS_DIR" \
            --output_prefix "$PREFIX" \
            --base_seed $SEED \
            $MODEL_ARG \
            > "logs/gen_iter${iter}_svc${i}.log" 2>&1 &
        
        PIDS+=($!)
        sleep 2  # 错开启动
    done
    
    for pid in "${PIDS[@]}"; do
        wait $pid
    done
    
    GEN_END=$(date +%s)
    echo "[$(date +%H:%M:%S)] 自博弈完成，耗时 $((GEN_END - ITER_START))秒"
    
    # ========== 阶段1.5: 复制数据到磁盘 (顺序写入) ==========
    echo "[$(date +%H:%M:%S)] 复制数据到磁盘..."
    mv "$TMPFS_DIR"/*.jsonl "$DATA_DIR"/ 2>/dev/null || true
    mv "$TMPFS_DIR"/*.json "$DATA_DIR"/ 2>/dev/null || true
    
    # ========== 阶段2: 训练 ==========
    echo "[$(date +%H:%M:%S)] 阶段2: 训练..."
    
    # 【关键修复】加载上一次训练的模型继续训练，而不是从随机模型开始
    # 优先使用 latest_model.pt（每次训练后保存），其次使用 best_model.pt
    LATEST_MODEL="$CHECKPOINT_DIR/latest_model.pt"
    LOAD_MODEL=""
    if [ -f "$LATEST_MODEL" ]; then
        LOAD_MODEL="--load_checkpoint $LATEST_MODEL"
        echo "  继续训练模型: $LATEST_MODEL"
    elif [ -f "$BEST_MODEL" ]; then
        LOAD_MODEL="--load_checkpoint $BEST_MODEL"
        echo "  继续训练模型: $BEST_MODEL"
    else
        echo "  首次训练，从随机模型开始"
    fi
    
    # 【修复】只使用当前迭代生成的数据文件，而不是所有历史数据
    ITER_PREFIX="iter_$(printf '%04d' $iter)"
    CURRENT_DATA_FILES="$DATA_DIR/${ITER_PREFIX}_*.jsonl"
    DATA_FILE_COUNT=$(ls -1 $CURRENT_DATA_FILES 2>/dev/null | wc -l)
    echo "  本次迭代数据文件: ${DATA_FILE_COUNT} 个 (${ITER_PREFIX}_*)"
    
    python -m v0.train \
        --data_files $CURRENT_DATA_FILES \
        --iterations 1 \
        --epochs $TRAIN_EPOCHS \
        --batch_size $TRAIN_BATCH_SIZE \
        --lr $TRAIN_LR \
        --device $DEVICE \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --eval_games_vs_random $EVAL_GAMES \
        --eval_games_vs_best 0 \
        $LOAD_MODEL \
        > "logs/train_iter${iter}.log" 2>&1
    
    # 每次训练后，将 model_iter_1.pt 复制为 latest_model.pt，确保下次迭代可以继续训练
    if [ -f "$CHECKPOINT_DIR/model_iter_1.pt" ]; then
        cp "$CHECKPOINT_DIR/model_iter_1.pt" "$LATEST_MODEL"
        echo "  已保存 latest_model.pt 用于下次迭代"
    fi
    
    # 【优化】删除旧迭代的数据文件，释放磁盘空间（保留最近 N 次迭代的数据）
    KEEP_LAST_ITERS=3  # 保留最近3次迭代的数据
    if [ $iter -gt $KEEP_LAST_ITERS ]; then
        OLD_ITER=$((iter - KEEP_LAST_ITERS))
        OLD_PREFIX="iter_$(printf '%04d' $OLD_ITER)"
        rm -f "$DATA_DIR"/${OLD_PREFIX}_*.jsonl "$DATA_DIR"/${OLD_PREFIX}_*.json 2>/dev/null || true
        echo "  已清理旧数据: ${OLD_PREFIX}_*"
    fi
    
    ITER_END=$(date +%s)
    
    # 统计
    DATA_COUNT=$(ls -1 "$DATA_DIR"/*.jsonl 2>/dev/null | wc -l)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    
    echo "[$(date +%H:%M:%S)] 迭代 $iter 完成，耗时 $((ITER_END - ITER_START))秒 | 数据: ${DATA_COUNT}文件 | GPU: ${GPU_MEM}MB"
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "  总耗时: $((TOTAL_ELAPSED / 3600))小时 $((TOTAL_ELAPSED % 3600 / 60))分钟"
echo "=========================================="

