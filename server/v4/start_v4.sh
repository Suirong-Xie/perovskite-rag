#!/bin/bash
# V4 启动脚本 — 只启动嵌入模型(11435) + V4 Server(8001)

BASE_DIR="/data1/perovskite-rag"
OLLAMA_BIN="/data1/ollama/bin/ollama"
OLLAMA_MODELS="/data1/ollama/model"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$LOG_DIR"

echo "🚀 PerovskiteGPT V4 启动中..."

# ─── 1. 清理旧的 11434 进程 ───
echo "[1/3] 清理旧进程..."
# 杀掉所有旧 ollama serve 和 runner
pkill -f "ollama serve" 2>/dev/null || true
pkill -f "ollama runner" 2>/dev/null || true
sleep 2

# ─── 2. 启动 11435 (嵌入: mxbai-embed-large, GPU 1) ───
echo "[2/3] 启动嵌入模型 (11435, GPU 1)..."
CUDA_VISIBLE_DEVICES=1 \
OLLAMA_HOST=127.0.0.1:11435 \
OLLAMA_MODELS="$OLLAMA_MODELS" \
nohup "$OLLAMA_BIN" serve > "$LOG_DIR/ollama_11435.log" 2>&1 &
EMBED_PID=$!
echo "  → PID $EMBED_PID"

# 等待就绪
for i in $(seq 1 15); do
    if curl -s --max-time 2 "http://127.0.0.1:11435/api/tags" > /dev/null 2>&1; then
        echo "  → 嵌入模型就绪"
        break
    fi
    sleep 1
done

# ─── 3. 启动 V4 Server (8001) ───
echo "[3/3] 启动 V4 Server (8001)..."
cd "$BASE_DIR/server/v4"
echo P@ssw0rd | sudo -S fuser -k 8001/tcp 2>/dev/null || true 2>/dev/null || true
sleep 1
/data1/perovskite-rag/.RAGenv/bin/python3 server.py --port 8001 > "$LOG_DIR/v4_server.log" 2>&1 &
V4_PID=$!
echo "  → PID $V4_PID"

sleep 3
echo ""
echo "✅ PerovskiteGPT V4 启动完成！"
echo "   └ 嵌入 (GPU 1): http://127.0.0.1:11435 (mxbai-embed-large)"
echo "   └ V4 Server:    http://localhost:8001"
echo "   └ 注意：11434 (llama3-70b) 不再启动"
