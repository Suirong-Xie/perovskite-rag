#!/usr/bin/env bash
# 批量语义 chunk 脚本 - 稳定版本
# 用法: bash run_chunking.sh 2021

set -e
YEAR=${1:-2021}
LOG="/tmp/chunk_run_${YEAR}.log"
PIDFILE="/tmp/chunk_pid_${YEAR}"

echo "[$(date)] Starting chunking for year $YEAR" > "$LOG"

export PYTHONUNBUFFERED=1
export LLM_API_KEY="sk-74b8c682599d4702abc746d7f598673b"
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-chat"

cd /data1/perovskite-rag
exec /data1/perovskite-rag/.RAGenv/bin/python -u pipeline/llm_semantic_chunker.py --year "$YEAR" >> "$LOG" 2>&1 &
PID=$!
echo $PID > "$PIDFILE"
echo "[$(date)] Started PID=$PID" >> "$LOG"
wait $PID
echo "[$(date)] Finished" >> "$LOG"
