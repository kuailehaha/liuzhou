#!/bin/bash
# 资源监控脚本 - 独立运行，定期记录GPU/CPU/内存使用情况
# 用法: ./scripts/monitor_resources.sh [interval_seconds]
# 
# 与训练脚本配合使用:
#   nohup ./scripts/large_scale_train.sh 100 > train_output.log 2>&1 &
#   nohup ./scripts/monitor_resources.sh 60 > /dev/null 2>&1 &

INTERVAL=${1:-60}  # 默认每60秒记录一次
LOG_FILE="logs/resource_monitor.log"

mkdir -p logs

echo "========================================" >> "$LOG_FILE"
echo "资源监控开始: $(date)" >> "$LOG_FILE"
echo "监控间隔: ${INTERVAL}秒" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # GPU信息
    GPU_INFO=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
    if [ -n "$GPU_INFO" ]; then
        GPU_MEM_USED=$(echo "$GPU_INFO" | cut -d',' -f1 | tr -d ' ')
        GPU_MEM_TOTAL=$(echo "$GPU_INFO" | cut -d',' -f2 | tr -d ' ')
        GPU_UTIL=$(echo "$GPU_INFO" | cut -d',' -f3 | tr -d ' ')
        GPU_TEMP=$(echo "$GPU_INFO" | cut -d',' -f4 | tr -d ' ')
    else
        GPU_MEM_USED="N/A"
        GPU_MEM_TOTAL="N/A"
        GPU_UTIL="N/A"
        GPU_TEMP="N/A"
    fi
    
    # 系统内存 (MB)
    MEM_INFO=$(free -m | grep Mem)
    MEM_TOTAL=$(echo "$MEM_INFO" | awk '{print $2}')
    MEM_USED=$(echo "$MEM_INFO" | awk '{print $3}')
    MEM_PERCENT=$((MEM_USED * 100 / MEM_TOTAL))
    
    # CPU使用率
    CPU_UTIL=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    
    # Python进程数和内存
    PY_PROCS=$(pgrep -c python 2>/dev/null || echo "0")
    PY_MEM=$(ps aux | grep python | grep -v grep | awk '{sum += $6} END {print int(sum/1024)}')
    
    # 数据文件数
    DATA_FILES=$(ls -1 ./v0/data/self_play/*.jsonl 2>/dev/null | wc -l)
    
    # 格式化输出
    echo "[$TIMESTAMP] GPU: ${GPU_MEM_USED}/${GPU_MEM_TOTAL}MB (${GPU_UTIL}%) ${GPU_TEMP}°C | RAM: ${MEM_USED}/${MEM_TOTAL}MB (${MEM_PERCENT}%) | CPU: ${CPU_UTIL}% | Python: ${PY_PROCS}进程 ${PY_MEM}MB | 数据: ${DATA_FILES}文件" >> "$LOG_FILE"
    
    # 警告检测
    if [ "$GPU_MEM_USED" != "N/A" ] && [ "$GPU_MEM_USED" -gt 38000 ]; then
        echo "[$TIMESTAMP] ⚠️ GPU显存使用超过38GB!" >> "$LOG_FILE"
    fi
    
    if [ "$MEM_PERCENT" -gt 90 ]; then
        echo "[$TIMESTAMP] ⚠️ 系统内存使用超过90%!" >> "$LOG_FILE"
    fi
    
    sleep "$INTERVAL"
done

