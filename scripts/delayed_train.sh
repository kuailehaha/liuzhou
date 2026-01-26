#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/.cache/liuzhou"
SESSIONS=(embed llm reranker)
WAIT_PATTERN="batch_run_output_format.py.*--base-url http://127.0.0.1:11000/v1.*--model Qwen3-30B-A3B-Instruct-2507"
WAIT_SLEEP_SEC=30

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

wait_for_pipeline() {
  log "Waiting for pipeline process to finish: $WAIT_PATTERN"
  while pgrep -f "$WAIT_PATTERN" >/dev/null 2>&1; do
    sleep "$WAIT_SLEEP_SEC"
  done
  log "Pipeline process not found; continuing"
}

collect_descendants() {
  local pid="$1"
  local children
  children=$(ps -o pid= --ppid "$pid" | tr -d ' ' || true)
  for child in $children; do
    echo "$child"
    collect_descendants "$child"
  done
}

kill_vllm_in_session() {
  local session="$1"
  if ! tmux has-session -t "$session" 2>/dev/null; then
    log "tmux session '$session' not found; skip"
    return 0
  fi

  local pane_pids
  pane_pids=$(tmux list-panes -t "$session" -F "#{pane_pid}" | tr -d '\r' || true)
  if [[ -z "$pane_pids" ]]; then
    log "tmux session '$session' has no panes; skip"
    return 0
  fi

  declare -A seen=()
  local pid
  for pid in $pane_pids; do
    seen["$pid"]=1
    local child
    while read -r child; do
      [[ -n "$child" ]] && seen["$child"]=1
    done < <(collect_descendants "$pid")
  done

  local killed_any=0
  for pid in "${!seen[@]}"; do
    local cmd
    cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
    if [[ -n "$cmd" && "$cmd" == *vllm* ]]; then
      log "TERM vllm pid $pid (session $session): $cmd"
      kill -TERM "$pid" 2>/dev/null || true
      killed_any=1
    fi
  done

  if [[ "$killed_any" -eq 0 ]]; then
    log "no vllm process found in session '$session'"
  fi

  # Give processes a moment to exit
  sleep 10

  for pid in "${!seen[@]}"; do
    local cmd
    cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
    if [[ -n "$cmd" && "$cmd" == *vllm* ]]; then
      log "KILL vllm pid $pid (session $session): $cmd"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

wait_for_pipeline

log "Starting cleanup of vllm processes in tmux sessions: ${SESSIONS[*]}"
for session in "${SESSIONS[@]}"; do
  kill_vllm_in_session "$session"
done

log "Launching training: ./scripts/toy_train.sh"
cd "$ROOT"
./scripts/toy_train.sh
