#!/bin/bash
# PerovskiteGPT 启动脚本
# 按架构文档启动双端口 Ollama + RAG Server

set -e

# ─── 配置 ───
BASE_DIR="/data1/perovskite-rag"
OLLAMA_BIN="/data1/ollama/bin/ollama"
OLLAMA_MODELS="/data1/ollama/model"
LOG_DIR="$BASE_DIR/logs"
PIDFILE_GEN="/var/run/ollama_11434.pid"
PIDFILE_EMBED="/var/run/ollama_11435.pid"
PIDFILE_RAG="/var/run/perovskitegpt.pid"

echo "🚀 PerovskiteGPT 启动中..."

# ─── 创建日志目录 ───
mkdir -p "$LOG_DIR"

# ─── 1. 启动 11434 (生成: llama3-70b-gpu, GPU 0) ───
echo "[1/3] 启动 Ollama 生成实例 (11434, GPU 0)..."
if [ -f "$PIDFILE_GEN" ] && kill -0 $(cat "$PIDFILE_GEN") 2>/dev/null; then
    echo "  → 11434 已在运行"
else
    CUDA_VISIBLE_DEVICES=0 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS="$OLLAMA_MODELS" \
    nohup "$OLLAMA_BIN" serve > "$LOG_DIR/ollama_11434.log" 2>&1 &
    echo $! > "$PIDFILE_GEN"
    echo "  → 已启动 (PID $(cat $PIDFILE_GEN))"
fi

# ─── 2. 启动 11435 (嵌入: mxbai-embed-large, GPU 1) ───
echo "[2/3] 启动 Ollama 嵌入实例 (11435, GPU 1)..."
if [ -f "$PIDFILE_EMBED" ] && kill -0 $(cat "$PIDFILE_EMBED") 2>/dev/null; then
    echo "  → 11435 已在运行"
else
    CUDA_VISIBLE_DEVICES=1 \
    OLLAMA_HOST=127.0.0.1:11435 \
    OLLAMA_MODELS="$OLLAMA_MODELS" \
    nohup "$OLLAMA_BIN" serve > "$LOG_DIR/ollama_11435.log" 2>&1 &
    echo $! > "$PIDFILE_EMBED"
    echo "  → 已启动 (PID $(cat $PIDFILE_EMBED))"
fi

# ─── 等待 Ollama 就绪 ───
echo "  ⏳ 等待 Ollama 就绪..."
for port in 11434 11435; do
    for i in $(seq 1 15); do
        if curl -s --max-time 2 "http://127.0.0.1:$port/api/tags" > /dev/null 2>&1; then
            echo "  → 127.0.0.1:$port 就绪"
            break
        fi
        sleep 1
    done
done

# ─── 3. 启动 RAG Server ───
echo "[3/3] 启动 RAG Server (8000)..."
if [ -f "$PIDFILE_RAG" ] && kill -0 $(cat "$PIDFILE_RAG") 2>/dev/null; then
    echo "  → RAG Server 已在运行"
else
    cd "$BASE_DIR/server/v2"
    rm -f "$BASE_DIR/data/qdrant_data/.lock"
    nohup /data1/perovskite-rag/.RAGenv/bin/python server.py > "$LOG_DIR/server.log" 2>&1 &
    echo $! > "$PIDFILE_RAG"
    echo "  → 已启动 (PID $(cat $PIDFILE_RAG))"
fi

# ─── 等待 RAG 就绪 ───
echo "  ⏳ 等待 RAG Server 就绪..."
for i in $(seq 1 20); do
    if curl -s --max-time 2 http://localhost:8000/ > /dev/null 2>&1; then
        echo "  → RAG Server 就绪 (http://localhost:8000)"
        break
    fi
    sleep 2
done

echo ""
echo "✅ PerovskiteGPT 启动完成！"
echo "   └ 生成 (GPU 0): http://127.0.0.1:11434 (llama3-70b)"
echo "   └ 嵌入 (GPU 1): http://127.0.0.1:11435 (mxbai-embed-large)"
echo "   └ RAG Server:   http://localhost:8000"
echo "   └ 日志目录:     $LOG_DIR"
