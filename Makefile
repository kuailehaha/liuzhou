# Makefile for 随机智能体测试

.PHONY: test-single test-multiple test-basic test-enhanced test-large test-huge help

# 默认目标：显示帮助信息
help:
	@echo "随机智能体测试工具 Makefile"
	@echo ""
	@echo "可用目标:"
	@echo "  test-single     运行单局详细测试"
	@echo "  test-multiple   运行多局简要测试（10局）"
	@echo "  test-basic      运行基本测试（100局）"
	@echo "  test-enhanced   运行增强测试（100局，带统计信息）"
	@echo "  test-large      运行大规模测试（1000局）"
	@echo "  test-huge       运行超大规模测试（10000局）"
	@echo "  test-all        运行所有测试（从小到大）"
	@echo ""
	@echo "参数设置："
	@echo "  SEED=<数字>     指定随机种子（默认42）"
	@echo "  QUIET=1         使用安静模式（减少输出）"
	@echo ""
	@echo "示例:"
	@echo "  make test-enhanced SEED=100"
	@echo "  make test-large QUIET=1"

# 默认参数
SEED ?= 42
QUIET_FLAG = $(if $(filter 1,$(QUIET)),-q,)

# 单局详细测试
test-single:
	python -m tests.random_agent.run_tests single -s $(SEED)

# 多局简要测试
test-multiple:
	python -m tests.random_agent.run_tests multiple -n 10 -s $(SEED)

# 基本测试（100局）
test-basic:
	python -m tests.random_agent.run_tests basic -n 100 -s $(SEED)

# 增强测试（100局，带统计信息）
test-enhanced:
	python -m tests.random_agent.run_tests enhanced -n 100 -s $(SEED) $(QUIET_FLAG)

# 大规模测试（1000局）
test-large:
	@echo "运行1000局游戏，起始种子: $(SEED)"
	python -m tests.random_agent.run_tests enhanced -n 1000 -s $(SEED) $(QUIET_FLAG)
	@echo "注意: 每局游戏使用不同的随机种子，从$(SEED)到$(shell expr $(SEED) + 999)"

# 超大规模测试（10000局）
test-huge:
	@echo "启动10000局测试，这可能需要较长时间..."
	@mkdir -p test_results
	python -m tests.random_agent.run_tests enhanced -n 10000 -s $(SEED) -q > test_results/test_10k_seed$(SEED).txt
	@echo "测试完成，结果保存在 test_results/test_10k_seed$(SEED).txt"

# 运行所有测试
test-all: test-single test-multiple test-enhanced test-large

# 清理测试结果
clean:
	rm -f *.txt
	rm -rf test_results 
