from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from rerankers import Reranker
from openai import OpenAI
import os
import logging
from datetime import datetime
from pathlib import Path

# ========== 初始化日志 ==========
LOG_DIR = Path("/root/autodl-tmp/QH/rag_demo_20251026/log")
LOG_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"rag_demo_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"日志初始化成功：{log_file}")


# ========== Step 1. 加载知识文件 ==========
loader = PyPDFLoader("/root/autodl-tmp/QH/rag_demo_20251026/data/GitHubPR操作简明指南.pdf")
docs = loader.load()
logger.info(f"[INFO] 加载文档完成，共 {len(docs)} 页")


# ========== Step 2. 分块 ==========
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
logger.info(f"[INFO] 文档分块完成，共 {len(chunks)} 个 chunk")


# ========== Step 3. 向量化 ==========
embedder = HuggingFaceEmbeddings(model_name="/root/autodl-tmp/QH/rag_demo_20251026/models/bge-large-zh-v1___5")
VECTOR_DIR = Path("/root/autodl-tmp/QH/rag_demo_20251026/vector_database")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# 如果向量数据库已存在，就直接加载
if (VECTOR_DIR / "index.faiss").exists():
    vector_db = FAISS.load_local(str(VECTOR_DIR), embedder, allow_dangerous_deserialization=True)
    logger.info(f"已加载现有向量数据库：{VECTOR_DIR}")
else:
    vector_db = FAISS.from_documents(chunks, embedder)
    vector_db.save_local(str(VECTOR_DIR))
    logger.info(f"新建向量数据库并保存至：{VECTOR_DIR}")


# ========== Step 4. 同时建立 BM25 检索器 ==========
bm25_retriever = BM25Retriever.from_documents(chunks)
vector_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
logger.info(f"[INFO] BM25 与 向量检索器初始化完成")


# ========== Step 5. 混合检索器（带权重） ==========
def hybrid_retrieve(query, w_bm25=0.5, w_vector=0.5, top_k=10):
    bm25_results = bm25_retriever.get_relevant_documents(query)
    vector_results = vector_retriever.get_relevant_documents(query)

    # 获取分数（LangChain文档对象有metadata可用来存放score）
    bm25_scored = [(doc, (1 - i / len(bm25_results)) * w_bm25) for i, doc in enumerate(bm25_results[:top_k])]
    vector_scored = [(doc, (1 - i / len(vector_results)) * w_vector) for i, doc in enumerate(vector_results[:top_k])]

    combined = bm25_scored + vector_scored
    combined = sorted(combined, key=lambda x: x[1], reverse=True)
    docs = [doc for doc, _ in combined[:top_k]]

    logger.info(f"\n[INFO] 检索阶段完成：BM25返回 {len(bm25_results)} 条，向量检索返回 {len(vector_results)} 条，合并取前 {top_k}")
    for i, d in enumerate(docs[:3]):
        logger.info(f"[DOC {i+1}] {d.page_content[:100]}...\n")
    return docs


# ========== Step 6. Reranker 重排 ==========
reranker = Reranker("/root/autodl-tmp/QH/rag_demo_20251026/models/bge-reranker-base")

def rerank_docs(query, docs, top_k=5):
    # 提取文本内容传入 reranker
    texts = [d.page_content for d in docs]
    results = reranker.rank(query, texts)
    
    # results[i] 对应 docs[i]
    reranked_pairs = list(zip(docs, [r.score for r in results]))
    reranked_pairs = sorted(reranked_pairs, key=lambda x: x[1], reverse=True)
    
    reranked_docs = [doc for doc, _ in reranked_pairs[:top_k]]

    logger.info(f"[INFO] 重排完成，选出前 {len(reranked_docs)} 条最相关内容：")
    for i, (doc, score) in enumerate(reranked_pairs[:top_k]):
        logger.info(f"[RERANK-{i+1}] (score={score:.4f}) {doc.page_content[:100]}...\n")

    return reranked_docs


# ========== Step 7. 调用硅基流动 API ==========
os.environ["OPENAI_API_KEY"] = ""   # 你的api密钥（这里是我自己的密钥，大家可以换成自己的哈，money有限哈哈）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.siliconflow.cn/v1")


# ========== Step 8. LLM 直接回答 ==========
def llm_only_answer(query):
    completion = client.chat.completions.create(
        model="Qwen/Qwen3-14B",
        messages=[
            {"role": "system", "content": "你是一个知识问答专家。"},
            {"role": "user", "content": query}
        ],
        temperature=0.7
    )
    return completion.choices[0].message.content


# ========== Step 9. RAG 增强回答 ==========
def rag_answer(query):
    docs = hybrid_retrieve(query)
    reranked = rerank_docs(query, docs)
    context = "\n".join([d.page_content for d in reranked])

    prompt = f"""
             你是一个知识问答助手。
             请根据以下资料回答用户问题，回答要尽量引用资料内容而非臆测。

             资料内容：
             {context}

             用户问题：
             {query}
             """
    completion = client.chat.completions.create(
        model="Qwen/Qwen3-14B",
        messages=[
            {"role": "system", "content": "你是一个知识问答专家。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return completion.choices[0].message.content


# ========== Step 10. 对比演示 ==========
if __name__ == "__main__":
    query = input("\n请输入你的问题：")
    print("\n" + "=" * 80)
    logger.info("=" * 80)
    logger.info(">>> [阶段1] 仅使用 LLM 回答")
    logger.info("=" * 80)

    llm_result = llm_only_answer(query)
    logger.info("\n[LLM ONLY 回答]:\n%s", llm_result)
    print(f"\n[LLM ONLY 回答]:\n{llm_result}")

    print("\n" + "=" * 80)
    logger.info("=" * 80)
    logger.info(">>> [阶段2] 使用 RAG 检索增强回答")
    logger.info("=" * 80)

    rag_result = rag_answer(query)
    logger.info("\n[RAG 回答]:\n%s", rag_result)
    print(f"\n[RAG 回答]:\n{rag_result}")

    print("\n" + "=" * 80)
    print(">>> [对比总结]")
    print("=" * 80)
    summary_text = (
        "LLM 结果：侧重语言生成，但可能不引用具体知识；\n"
        "RAG 结果：引用文档内容，回答更具体、更可靠。"
    )
    logger.info(summary_text)
    print(summary_text)
