#!/bin/bash
# PerovskiteGPT V4 启动脚本
set -e
BASE_DIR="/data1/perovskite-rag"
OLLAMA_BIN="/data1/ollama/bin/ollama"
OLLAMA_MODELS="/data1/ollama/model"
LOG_DIR="$BASE_DIR/logs"
PIDFILE_EMBED="/var/run/ollama_11435.pid"
PIDFILE_V4="/var/run/perovskitegpt_v4.pid"
echo "PerovskiteGPT V4 starting..."
echo "[1/2] Ollama embedding (11435)..."
if [ -f "$PIDFILE_EMBED" ] && kill -0 "$(cat "$PIDFILE_EMBED" 2>/dev/null)" 2>/dev/null; then
    echo " -> already running"
else
    CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS="$OLLAMA_MODELS" nohup "$OLLAMA_BIN" serve > "$LOG_DIR/ollama_11435.log" 2>&1 &
    echo "$!" > "$PIDFILE_EMBED"
    echo " -> started"
fi
for i in $(seq 1 15); do
    if curl -s --max-time 2 "http://127.0.0.1:11435/api/tags" > /dev/null 2>&1; then break; fi
    sleep 1
done
echo "[2/2] V4 Server (8001)..."
if [ -f "$PIDFILE_V4" ] && kill -0 "$(cat "$PIDFILE_V4" 2>/dev/null)" 2>/dev/null; then
    echo " -> already running"
else
    cd "$BASE_DIR/server/v4"
    nohup /data1/perovskite-rag/.RAGenv/bin/python3 server.py --port 8001 > "$LOG_DIR/v4_server.log" 2>&1 &
    echo "$!" > "$PIDFILE_V4"
    echo " -> started"
fi
for i in $(seq 1 10); do
    if curl -s --max-time 2 http://localhost:8001/ > /dev/null 2>&1; then break; fi
    sleep 1
done
echo "V4 ready on http://localhost:8001"
