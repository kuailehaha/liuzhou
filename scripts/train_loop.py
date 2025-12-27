#!/usr/bin/env python3
"""V0 训练调度脚本：自博弈累积 + 统一训练

用法:
    python scripts/train_loop.py --iterations 10 --games-per-iter 64
    python scripts/train_loop.py --generate-only --num-games 128  # 只生成数据
    python scripts/train_loop.py --train-only                      # 只训练
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from datetime import datetime


def run_command(cmd: list[str], description: str) -> int:
    """运行命令并打印输出"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"  命令: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    print(f"  耗时: {elapsed:.1f}秒, 返回码: {result.returncode}")
    return result.returncode


def generate_data(args: argparse.Namespace, iteration: int = 0) -> int:
    """生成自博弈数据"""
    cmd = [
        sys.executable, "-m", "v0.generate_data",
        "--num_games", str(args.games_per_iter),
        "--mcts_simulations", str(args.mcts_sims),
        "--batch_leaves", str(args.batch_leaves),
        "--device", args.device,
        "--output_dir", args.data_dir,
        "--output_prefix", f"iter_{iteration:04d}" if iteration > 0 else "self_play",
    ]
    
    # 使用最佳模型（如果存在）
    best_model = os.path.join(args.checkpoint_dir, "best_model.pt")
    if os.path.exists(best_model):
        cmd.extend(["--model_checkpoint", best_model])
        print(f"  使用模型: {best_model}")
    else:
        print("  使用随机初始化模型")
    
    return run_command(cmd, f"自博弈生成 {args.games_per_iter} 局")


def train_model(args: argparse.Namespace) -> int:
    """训练模型"""
    data_files = glob.glob(os.path.join(args.data_dir, "*.jsonl"))
    if not data_files:
        print(f"  错误: {args.data_dir} 中没有数据文件")
        return 1
    
    print(f"  数据文件数: {len(data_files)}")
    
    cmd = [
        sys.executable, "-m", "v0.train",
        "--data_files", *data_files,
        "--epochs", str(args.train_epochs),
        "--batch_size", str(args.train_batch_size),
        "--lr", str(args.train_lr),
        "--device", args.device,
        "--checkpoint_dir", args.checkpoint_dir,
        "--iterations", "1",
        "--eval_games_vs_random", str(args.eval_games),
        "--eval_games_vs_best", "0",
    ]
    
    return run_command(cmd, "训练模型")


def main():
    parser = argparse.ArgumentParser(description="V0 训练调度脚本")
    
    # 模式选择
    parser.add_argument("--generate-only", action="store_true", help="只生成数据，不训练")
    parser.add_argument("--train-only", action="store_true", help="只训练，不生成新数据")
    
    # 迭代控制
    parser.add_argument("--iterations", type=int, default=10, help="训练迭代次数")
    
    # 自博弈参数
    parser.add_argument("--games-per-iter", type=int, default=64, help="每次迭代自博弈局数")
    parser.add_argument("--mcts-sims", type=int, default=800, help="MCTS模拟次数")
    parser.add_argument("--batch-leaves", type=int, default=256, help="批量叶子节点数")
    
    # 训练参数
    parser.add_argument("--train-epochs", type=int, default=10, help="每次训练epoch数")
    parser.add_argument("--train-batch-size", type=int, default=64, help="训练batch大小")
    parser.add_argument("--train-lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--eval-games", type=int, default=20, help="评估对局数")
    
    # 路径配置
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--data-dir", type=str, default="./v0/data/self_play", help="数据目录")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_v0", help="检查点目录")
    
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("=" * 60)
    print("V0 训练调度器")
    print("=" * 60)
    print(f"  迭代次数: {args.iterations}")
    print(f"  自博弈: {args.games_per_iter} 局/迭代, {args.mcts_sims} sims, batch_leaves={args.batch_leaves}")
    print(f"  训练: {args.train_epochs} epochs, batch={args.train_batch_size}, lr={args.train_lr}")
    print(f"  设备: {args.device}")
    print(f"  数据目录: {args.data_dir}")
    print(f"  检查点: {args.checkpoint_dir}")
    print("=" * 60)
    
    total_start = time.time()
    
    if args.train_only:
        # 只训练模式
        train_model(args)
    elif args.generate_only:
        # 只生成数据模式
        generate_data(args)
    else:
        # 完整循环模式
        for i in range(1, args.iterations + 1):
            print(f"\n{'=' * 60}")
            print(f"迭代 {i} / {args.iterations}")
            print("=" * 60)
            
            # 自博弈
            if generate_data(args, i) != 0:
                print("自博弈失败，跳过训练")
                continue
            
            # 训练
            train_model(args)
    
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"训练完成！总耗时: {total_elapsed / 60:.1f} 分钟")
    print(f"检查点: {args.checkpoint_dir}")
    print(f"数据: {args.data_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

