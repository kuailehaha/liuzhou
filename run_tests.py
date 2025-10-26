#!/usr/bin/env python3
"""
测试运行脚本 - 用于执行各种类型的随机智能体测试
"""

import argparse
import time
import sys
from src.test_random_agent_enhanced import run_enhanced_tests
from src.test_random_agent_debug import test_single_game, test_multiple_games

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='随机智能体测试工具')
    
    # 测试模式
    parser.add_argument('mode', choices=['single', 'multiple', 'basic', 'enhanced'],
                      help='测试模式: single(单局详细), multiple(多局简要), basic(基本大规模), enhanced(增强大规模)')
    
    # 通用参数
    parser.add_argument('-n', '--num', type=int, default=10,
                      help='测试局数 (multiple模式默认10，basic/enhanced模式默认1000)')
    parser.add_argument('-s', '--seed', type=int, default=42,
                      help='起始随机种子 (默认42)')
    parser.add_argument('-t', '--turns', type=int, default=200,
                      help='每局最大回合数 (默认200)')
    
    # enhanced模式特定参数
    parser.add_argument('-q', '--quiet', action='store_true',
                      help='安静模式，减少输出 (仅enhanced模式有效)')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    print(f"==== 随机智能体测试 ({args.mode}模式) ====")
    print(f"起始随机种子: {args.seed}")
    
    start_time = time.time()
    
    if args.mode == 'single':
        print(f"测试单局游戏 (最大回合数: {args.turns})")
        test_single_game(args.seed, max_turns=args.turns)
        
    elif args.mode == 'multiple':
        print(f"测试多局游戏 (局数: {args.num}, 最大回合数: {args.turns})")
        test_multiple_games(args.num, args.seed, max_turns=args.turns)
        
    elif args.mode == 'enhanced':
        num = 1000 if args.num == 10 else args.num  # 增强模式默认1000局
        print(f"增强大规模测试 (局数: {num}, 最大回合数: {args.turns})")
        run_enhanced_tests(num, args.seed, verbose=not args.quiet, max_turns_per_game=args.turns)
    
    elapsed_time = time.time() - start_time
    print(f"\n测试完成！总运行时间: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main() 
