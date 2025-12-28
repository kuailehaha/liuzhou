#!/usr/bin/env bash
# 当根磁盘 IO 连续超阈值时，自动杀掉“当前 IO 最大”的进程（默认仅限当前用户）
# 用法：
#   nohup bash ./scripts/kill_top_io.sh   --mbps 150   --interval 2   --grace 3   --scope user   --sample 1   > logs/kill_top_io.nohup 2>&1 &
#
# 参数：
#   --mbps       触发阈值（MB/s），默认150
#   --interval   采样间隔秒（根磁盘总IO），默认2
#   --grace      连续超阈值次数才出手，默认3
#   --scope      user | all   （默认 user：只看/只杀当前用户；all 需要 sudo 才有意义）
#   --sample     采样窗口秒（用于找 top IO pid），默认1
#   --dry-run    只打印不杀

set -euo pipefail

THRESHOLD_MBPS=150
INTERVAL=2
GRACE=3
SCOPE="user"
SAMPLE=1
DRY_RUN=0
LOG_FILE="logs/kill_top_io.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mbps) THRESHOLD_MBPS="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --grace) GRACE="$2"; shift 2 ;;
    --scope) SCOPE="$2"; shift 2 ;;
    --sample) SAMPLE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift 1 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

mkdir -p logs
SELF_PID="$$"

echo "========================================" >> "$LOG_FILE"
echo "启动 kill_top_io: $(date)" >> "$LOG_FILE"
echo "阈值: ${THRESHOLD_MBPS}MB/s, 间隔: ${INTERVAL}s, 连续${GRACE}次, scope=${SCOPE}, sample=${SAMPLE}s, dry_run=${DRY_RUN}" >> "$LOG_FILE"
echo "self_pid=$SELF_PID" >> "$LOG_FILE"

# 更稳的根磁盘识别：用 findmnt + lsblk
get_root_disk() {
  local root_src pk
  root_src=$(findmnt -n -o SOURCE / || true)
  # 例如 /dev/nvme0n1p1 或 /dev/sda1
  if [[ -z "$root_src" ]]; then return 1; fi
  pk=$(lsblk -no PKNAME "$root_src" 2>/dev/null || true)
  if [[ -z "$pk" ]]; then
    # 兜底：去掉 /dev/ 前缀 + 去分区号
    local dev="${root_src#/dev/}"
    dev="${dev%%p[0-9]*}"
    dev="${dev%%[0-9]*}"
    echo "$dev"
  else
    echo "$pk"
  fi
}

DISK_NAME="$(get_root_disk || true)"
if [[ -z "$DISK_NAME" ]]; then
  echo "无法确定根磁盘名" | tee -a "$LOG_FILE"
  exit 1
fi
echo "监控磁盘: $DISK_NAME" >> "$LOG_FILE"

read_diskstats() {
  # fields: 3=name 6=sectors_read 10=sectors_written
  awk -v disk="$DISK_NAME" '$3==disk {print $6, $10}' /proc/diskstats
}

read prev_read prev_write < <(read_diskstats)
over_count=0

read_pid_io_sum() {
  # 返回 read_bytes + write_bytes（字节数），读不到则返回空
  local pid="$1"
  local io="/proc/$pid/io"
  [[ -r "$io" ]] || return 1
  local r w
  r=$(awk '/^read_bytes:/ {print $2}' "$io" 2>/dev/null || true)
  w=$(awk '/^write_bytes:/ {print $2}' "$io" 2>/dev/null || true)
  [[ -n "$r" && -n "$w" ]] || return 1
  echo $((r + w))
}

pick_top_io_pid() {
  # 在 SAMPLE 秒窗口内采样所有候选 pid 的 io 增量，返回 delta 最大的 pid
  local pids=()
  if [[ "$SCOPE" == "all" ]]; then
    mapfile -t pids < <(ps -e -o pid=)
  else
    mapfile -t pids < <(ps -u "$USER" -o pid=)
  fi

  # baseline
  declare -A base
  for pid in "${pids[@]}"; do
    [[ "$pid" == "$SELF_PID" ]] && continue
    local s
    s=$(read_pid_io_sum "$pid" 2>/dev/null || true)
    [[ -n "$s" ]] && base["$pid"]="$s"
  done

  sleep "$SAMPLE"

  local best_pid=""
  local best_delta=0
  for pid in "${!base[@]}"; do
    local s2
    s2=$(read_pid_io_sum "$pid" 2>/dev/null || true)
    [[ -n "$s2" ]] || continue
    local delta=$((s2 - base["$pid"]))
    if (( delta > best_delta )); then
      best_delta="$delta"
      best_pid="$pid"
    fi
  done

  [[ -n "$best_pid" ]] || return 1
  echo "$best_pid $best_delta"
}

kill_pid_safely() {
  local pid="$1"
  local reason="$2"
  local cmd
  cmd=$(ps -p "$pid" -o cmd= 2>/dev/null || true)

  local ts
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$ts] ⚠️ $reason | kill pid=$pid | cmd=$cmd" | tee -a "$LOG_FILE"

  if (( DRY_RUN == 1 )); then
    echo "[$ts] dry-run: 不实际 kill" | tee -a "$LOG_FILE"
    return 0
  fi

  kill -TERM "$pid" 2>/dev/null || true
  sleep 3
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "[$ts] 仍存活，kill -KILL" | tee -a "$LOG_FILE"
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

threshold_bytes_per_sec=$(( THRESHOLD_MBPS * 1024 * 1024 ))

while true; do
  sleep "$INTERVAL"
  read cur_read cur_write < <(read_diskstats) || continue

  delta_read=$((cur_read - prev_read))
  delta_write=$((cur_write - prev_write))
  prev_read=$cur_read
  prev_write=$cur_write

  bytes=$(( (delta_read + delta_write) * 512 ))
  bps=$(( bytes / INTERVAL ))

  ts=$(date '+%Y-%m-%d %H:%M:%S')
  mbps=$(awk -v b="$bytes" -v t="$INTERVAL" 'BEGIN {printf "%.1f", b/1024/1024/t}')
  echo "[$ts] IO=${mbps}MB/s (阈值=${THRESHOLD_MBPS})" >> "$LOG_FILE"

  if (( bps > threshold_bytes_per_sec )); then
    over_count=$((over_count + 1))
    if (( over_count >= GRACE )); then
      if top=$(pick_top_io_pid); then
        top_pid=$(awk '{print $1}' <<< "$top")
        top_delta=$(awk '{print $2}' <<< "$top")
        top_mbps=$(awk -v d="$top_delta" -v s="$SAMPLE" 'BEGIN {printf "%.1f", d/1024/1024/s}')
        kill_pid_safely "$top_pid" "根盘IO超阈值${over_count}次，top IO pid Δ=${top_mbps}MB/s"
      else
        echo "[$ts] 超阈值但未找到可读 /proc/<pid>/io 的候选进程（scope=$SCOPE）。" | tee -a "$LOG_FILE"
      fi
      over_count=0
    fi
  else
    over_count=0
  fi
done


