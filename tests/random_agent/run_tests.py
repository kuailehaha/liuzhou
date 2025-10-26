#!/usr/bin/env python3
"""
测试运行脚本 - 用于执行各种类型的随机智能体测试。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


if __package__:
    from .test_random_agent_enhanced import run_enhanced_tests
    from .test_random_agent_debug import test_single_game, test_multiple_games
else:  # 支持直接执行: python tests/random_agent/run_tests.py
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from tests.random_agent.test_random_agent_enhanced import run_enhanced_tests
    from tests.random_agent.test_random_agent_debug import test_single_game, test_multiple_games


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="随机智能体测试工具")

    parser.add_argument(
        "mode",
        choices=["single", "multiple", "basic", "enhanced"],
        help="测试模式: single(单局详细), multiple(多局简要), basic(基础批量), enhanced(增强批量)",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=10,
        help="测试局数 (multiple 默认 10，basic/enhanced 默认 1000)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="起始随机种子 (默认 42)",
    )
    parser.add_argument(
        "-t",
        "--turns",
        type=int,
        default=200,
        help="每局最大回合数 (默认 200)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="安静模式 (仅 enhanced 模式有效，减少输出)",
    )

    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()

    print(f"==== 随机智能体测试 ({args.mode} 模式) ====")
    print(f"起始随机种子: {args.seed}")

    start_time = time.time()

    if args.mode == "single":
        print(f"测试单局游戏 (最大回合数: {args.turns})")
        test_single_game(args.seed, max_turns=args.turns)

    elif args.mode == "multiple":
        print(f"测试多局游戏 (局数 {args.num}, 最大回合数: {args.turns})")
        test_multiple_games(args.num, args.seed, max_turns=args.turns)

    elif args.mode == "enhanced":
        num = 1000 if args.num == 10 else args.num  # 增强模式默认 1000 局
        print(f"增强大规模测试 (局数 {num}, 最大回合数: {args.turns})")
        run_enhanced_tests(
            num,
            args.seed,
            verbose=not args.quiet,
            max_turns_per_game=args.turns,
        )

    elapsed_time = time.time() - start_time
    print(f"\n测试完成！总运行时间: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    main()
