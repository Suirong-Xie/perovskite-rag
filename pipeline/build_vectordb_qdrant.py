import json
import warnings
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

warnings.filterwarnings("ignore")

# 嵌入模型 (GPU 1)
embed_model = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://127.0.0.1:11435"
)

# 读取所有 chunks
with open("/data1/perovskite-rag/data/chunked_data/chunks.jsonl", "r", encoding="utf-8") as f:
    chunks_data = [json.loads(line) for line in f]

texts = [item["content"] for item in chunks_data]
metadatas = [item["metadata"] for item in chunks_data]
print(f"加载 {len(texts)} 条数据")

# Qdrant 本地客户端（数据存储在 ./qdrant_data）
client = QdrantClient(path="/data1/perovskite-rag/data/qdrant_data")

# 集合名
collection_name = "perovskite_papers"

# 删除旧集合（如果存在）
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

# 创建新集合 (嵌入维度为 mxbai-embed-large 的输出维度 1024)
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

# 创建 QdrantVectorStore
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embed_model,
)

# 分批插入（每批 2000 条，避免内存过大）
batch_size = 2000
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_metadatas = metadatas[i:i+batch_size]
    vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)
    print(f"已完成 {min(i+batch_size, len(texts))}/{len(texts)}")

print("✅ Qdrant 向量库构建完成")
