import json
import warnings
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus

warnings.filterwarnings("ignore", category=UserWarning)

embed_model = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://127.0.0.1:11435"
)

with open("chunked_data/chunks.jsonl", "r", encoding="utf-8") as f:
    chunks_data = [json.loads(line) for line in f]

texts = [item["content"] for item in chunks_data]
metadatas = [item["metadata"] for item in chunks_data]
print(f"加载 {len(texts)} 条数据")

batch_size = 5000
vectorstore = None

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_metadatas = metadatas[i:i+batch_size]

    if vectorstore is None:
        # 第一次：创建集合并插入第一批
        vectorstore = Milvus.from_texts(
            texts=batch_texts,
            embedding=embed_model,
            metadatas=batch_metadatas,
            connection_args={
                "uri": "./milvus_data/perovskite.db",
                "keepalive_time_ms": 60000,
                "keepalive_timeout_ms": 20000,
            },
            collection_name="perovskite_papers",
            drop_old=True,   # 只对第一次生效
        )
    else:
        # 后续批次直接用 add_texts 追加
        vectorstore.add_texts(batch_texts, batch_metadatas)

    print(f"已完成 {min(i+batch_size, len(texts))}/{len(texts)}")

print("✅ 向量库构建完成")
