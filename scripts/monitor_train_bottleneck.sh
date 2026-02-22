#!/usr/bin/env bash
# Monitor v1 train-load bottlenecks without py-spy.
# Usage:
#   bash scripts/monitor_train_bottleneck.sh [duration_sec] [interval_sec] [log_file]
#   bash scripts/monitor_train_bottleneck.sh [duration_sec] [log_file]
#
# Example:
#   bash scripts/monitor_train_bottleneck.sh 7200 180 logs/train_bottleneck_$(date +%Y%m%d_%H%M%S).log
#   bash scripts/monitor_train_bottleneck.sh 7200 logs/train_bottleneck_$(date +%Y%m%d_%H%M%S).log
#
# Notes:
# - Target root process pattern can be overridden via MONITOR_PROC_PATTERN.
# - Script tracks matched root processes and all of their descendants.
# - Default root pattern covers big_train_v1 + v1 staged/eval entry processes.

set -u
set -o pipefail

DURATION_SEC="${1:-7200}"
INTERVAL_SEC="180"
LOG_FILE="logs/train_bottleneck_$(date +%Y%m%d_%H%M%S).log"

if [ "${2:-}" != "" ]; then
    if [[ "${2}" =~ ^[0-9]+$ ]]; then
        INTERVAL_SEC="${2}"
        if [ "${3:-}" != "" ]; then
            LOG_FILE="${3}"
        fi
    else
        LOG_FILE="${2}"
    fi
fi
PROC_PATTERN="${MONITOR_PROC_PATTERN:-big_train_v1.sh|torchrun|train_entry.py|eval_checkpoint.py}"

mkdir -p "$(dirname "$LOG_FILE")"

has_cmd() {
    command -v "$1" >/dev/null 2>&1
}

collect_descendants() {
    # Recursively collect descendants for input pid list.
    # Echoes unique pid list, one per line.
    local -a queue=("$@")
    local pid
    declare -A seen=()

    while [ "${#queue[@]}" -gt 0 ]; do
        pid="${queue[0]}"
        queue=("${queue[@]:1}")

        if [ -z "$pid" ] || [ ! -d "/proc/${pid}" ]; then
            continue
        fi
        if [ "${seen[$pid]+x}" = "x" ]; then
            continue
        fi
        seen["$pid"]=1

        mapfile -t _children < <(pgrep -P "$pid" 2>/dev/null || true)
        if [ "${#_children[@]}" -gt 0 ]; then
            queue+=("${_children[@]}")
        fi
    done

    for pid in "${!seen[@]}"; do
        printf '%s\n' "$pid"
    done | sort -n
}

log_line() {
    printf '%s\n' "$*" >> "$LOG_FILE"
}

read_proc_io_fields() {
    # Output: read_bytes write_bytes rchar wchar
    # shellcheck disable=SC2016
    awk '
        /^read_bytes:/ {rb=$2}
        /^write_bytes:/ {wb=$2}
        /^rchar:/ {rc=$2}
        /^wchar:/ {wc=$2}
        END {printf "%d %d %d %d\n", rb+0, wb+0, rc+0, wc+0}
    ' "$1"
}

START_EPOCH="$(date +%s)"
END_EPOCH=$((START_EPOCH + DURATION_SEC))
SAMPLE_ID=0

declare -A PREV_READ_BYTES
declare -A PREV_WRITE_BYTES
declare -A PREV_EPOCH

log_line "===== monitor_train_bottleneck start ====="
log_line "start_time=$(date '+%F %T %Z')"
log_line "duration_sec=${DURATION_SEC}"
log_line "interval_sec=${INTERVAL_SEC}"
log_line "proc_pattern=${PROC_PATTERN}"
log_line "host=$(hostname 2>/dev/null || echo unknown)"
log_line "kernel=$(uname -sr 2>/dev/null || echo unknown)"
log_line "cmd_available: nvidia-smi=$(has_cmd nvidia-smi && echo yes || echo no), pidstat=$(has_cmd pidstat && echo yes || echo no), iostat=$(has_cmd iostat && echo yes || echo no), vmstat=$(has_cmd vmstat && echo yes || echo no)"
log_line "log_file=${LOG_FILE}"
log_line "==========================================="

on_exit() {
    log_line "===== monitor_train_bottleneck end ====="
    log_line "end_time=$(date '+%F %T %Z')"
}
trap on_exit EXIT

while true; do
    NOW_EPOCH="$(date +%s)"
    if [ "$NOW_EPOCH" -ge "$END_EPOCH" ]; then
        break
    fi

    SAMPLE_ID=$((SAMPLE_ID + 1))
    TS="$(date '+%F %T')"

    log_line ""
    log_line "----- sample=${SAMPLE_ID} ts=${TS} epoch=${NOW_EPOCH} -----"
    log_line "[uptime] $(uptime 2>/dev/null || echo unavailable)"
    log_line "[loadavg] $(cat /proc/loadavg 2>/dev/null || echo unavailable)"

    if has_cmd free; then
        log_line "[free_mb]"
        free -m >> "$LOG_FILE" 2>&1 || true
    fi

    if has_cmd vmstat; then
        log_line "[vmstat_1s]"
        vmstat 1 2 2>/dev/null | tail -n 2 >> "$LOG_FILE" || true
    fi

    if has_cmd iostat; then
        log_line "[iostat_x_1s]"
        iostat -x 1 1 >> "$LOG_FILE" 2>&1 || true
    fi

    if has_cmd nvidia-smi; then
        log_line "[nvidia_gpu_query]"
        nvidia-smi \
            --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
            --format=csv,noheader,nounits >> "$LOG_FILE" 2>&1 || true

        log_line "[nvidia_compute_apps]"
        nvidia-smi \
            --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader,nounits >> "$LOG_FILE" 2>&1 || true

        log_line "[nvidia_pmon_1sample]"
        nvidia-smi pmon -c 1 >> "$LOG_FILE" 2>&1 || true
    fi

    mapfile -t ROOT_PID_ARR < <(pgrep -f "$PROC_PATTERN" 2>/dev/null || true)
    if [ "${#ROOT_PID_ARR[@]}" -eq 0 ]; then
        log_line "[root_pids] none"
        log_line "[tracked_pids] none"
    else
        ROOT_PID_CSV="$(IFS=,; echo "${ROOT_PID_ARR[*]}")"
        log_line "[root_pids] ${ROOT_PID_CSV}"

        mapfile -t PID_ARR < <(collect_descendants "${ROOT_PID_ARR[@]}")
        if [ "${#PID_ARR[@]}" -eq 0 ]; then
            log_line "[tracked_pids] none_after_descendant_scan"
            sleep "$INTERVAL_SEC"
            continue
        fi

        PID_CSV="$(IFS=,; echo "${PID_ARR[*]}")"
        log_line "[tracked_pids] ${PID_CSV}"

        log_line "[ps_tracked_pids]"
        ps -p "$PID_CSV" -o pid,ppid,stat,etime,pcpu,pmem,rss,psr,comm,args --no-headers >> "$LOG_FILE" 2>&1 || true

        if has_cmd pidstat; then
            log_line "[pidstat_dru_1s]"
            pidstat -dru -h -p "$PID_CSV" 1 1 >> "$LOG_FILE" 2>&1 || true
        fi

        for pid in "${PID_ARR[@]}"; do
            if [ ! -d "/proc/${pid}" ]; then
                continue
            fi

            if [ -r "/proc/${pid}/io" ]; then
                read -r rb wb rc wc < <(read_proc_io_fields "/proc/${pid}/io")
                prev_rb="${PREV_READ_BYTES[$pid]:-}"
                prev_wb="${PREV_WRITE_BYTES[$pid]:-}"
                prev_t="${PREV_EPOCH[$pid]:-}"

                if [ -n "${prev_rb}" ] && [ -n "${prev_wb}" ] && [ -n "${prev_t}" ] && [ "$NOW_EPOCH" -gt "$prev_t" ]; then
                    dt=$((NOW_EPOCH - prev_t))
                    dr=$((rb - prev_rb))
                    dw=$((wb - prev_wb))
                    r_mibps="$(awk -v b="$dr" -v t="$dt" 'BEGIN {printf "%.3f", (t>0? b/t/1048576:0)}')"
                    w_mibps="$(awk -v b="$dw" -v t="$dt" 'BEGIN {printf "%.3f", (t>0? b/t/1048576:0)}')"
                    log_line "[proc_io] pid=${pid} read_bytes=${rb} write_bytes=${wb} rchar=${rc} wchar=${wc} delta_sec=${dt} delta_read_bytes=${dr} delta_write_bytes=${dw} read_mib_s=${r_mibps} write_mib_s=${w_mibps}"
                else
                    log_line "[proc_io] pid=${pid} read_bytes=${rb} write_bytes=${wb} rchar=${rc} wchar=${wc} delta=bootstrap"
                fi

                PREV_READ_BYTES[$pid]="$rb"
                PREV_WRITE_BYTES[$pid]="$wb"
                PREV_EPOCH[$pid]="$NOW_EPOCH"
            fi

            if [ -r "/proc/${pid}/wchan" ]; then
                wchan="$(cat "/proc/${pid}/wchan" 2>/dev/null || echo unknown)"
                log_line "[proc_wchan] pid=${pid} wchan=${wchan}"
            fi

            if [ -r "/proc/${pid}/status" ]; then
                log_line "[proc_status] pid=${pid}"
                awk '
                    /^(Name|State|Threads|VmRSS|VmSize|voluntary_ctxt_switches|nonvoluntary_ctxt_switches):/ {print}
                ' "/proc/${pid}/status" | sed "s/^/  /" >> "$LOG_FILE" 2>&1 || true
            fi
        done
    fi

    sleep "$INTERVAL_SEC"
done

log_line "monitor finished normally."
