import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(Path(__file__).parent / ".env")

DATA_FILE = Path(__file__).parent / "data" / "说说备份_792059284.txt"
FAISS_INDEX_DIR = Path(__file__).parent / "data" / "faiss_index"

MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def build_embeddings() -> OpenAIEmbeddings:
    """创建指向 ModelScope 的 Embedding 实例。

    Returns:
        配置好 ModelScope endpoint 的 `OpenAIEmbeddings`。
    """
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.environ["MODELSCOPE_API_KEY"], # type: ignore
        openai_api_base=MODELSCOPE_BASE_URL, # type: ignore
    )


def build_vectorstore(chunks: list[Document], embeddings: OpenAIEmbeddings) -> FAISS:
    """将 chunks 向量化并存入 FAISS，同时持久化到本地磁盘。

    Args:
        chunks: 已切分的 `Document` 列表。
        embeddings: 用于生成向量的 Embedding 实例。

    Returns:
        构建完成的 `FAISS` 向量库。
    """
    print(f"正在向量化 {len(chunks)} 个 chunks，请稍候...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_INDEX_DIR))
    print(f"向量库已保存至 {FAISS_INDEX_DIR}")
    return vectorstore


def load_vectorstore(embeddings: OpenAIEmbeddings) -> FAISS:
    """从本地磁盘加载已有的 FAISS 向量库。

    Args:
        embeddings: 用于查询时生成向量的 Embedding 实例。

    Returns:
        加载完成的 `FAISS` 向量库。
    """
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_shuoshuo(file_path: Path) -> list[dict]:
    """从QQ空间说说备份文件中读取所有说说条目。

    Args:
        file_path: 备份文件路径。

    Returns:
        每条说说的字典列表，包含 `date` 和 `content` 字段。
    """
    text = file_path.read_text(encoding="utf-8")

    # 匹配 [YYYY年MM月DD日] 后跟内容行
    pattern = re.compile(
        r"\[(\d{4}年\d{2}月\d{2}日)\]\n内容：(.+?)(?=\n\[|\Z)",
        re.DOTALL,
    )

    entries = []
    for match in pattern.finditer(text):
        date_str = match.group(1).strip()
        content = match.group(2).strip()
        entries.append({"date": date_str, "content": content})

    return entries

def TextSplitter(text: str, max_length: int = 100) -> list[str]:
    """将文本分割成不超过 max_length 的片段，尽量在句子边界分割。

    Args:
        text: 待分割的文本。
        max_length: 每个片段的最大长度。

    Returns:
        分割后的文本片段列表。
    """
    if len(text) <= max_length:
        return [text]

    sentences = re.split(r"(?<=[。！？.!?])", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def split_into_chunks(
    entries: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """将说说条目转换为 Document 并用 RecursiveCharacterTextSplitter 切分。

    Args:
        entries: `load_shuoshuo` 返回的条目列表。
        chunk_size: 每个 chunk 的最大字符数。
        chunk_overlap: 相邻 chunk 的重叠字符数。

    Returns:
        切分后的 `Document` 列表，metadata 中保留原始日期。
    """
    docs = [
        Document(page_content=e["content"], metadata={"date": e["date"]})
        for e in entries
        if e["content"]
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["。", "！", "？", "…", "\n", ""],
    )
    return splitter.split_documents(docs)


def analyze_chunk_configs(
    entries: list[dict],
    configs: list[dict],
) -> None:
    """对比不同切分配置的统计数据，不调用任何 API。

    Args:
        entries: `load_shuoshuo` 返回的条目列表。
        configs: 每项包含 `chunk_size` 和 `chunk_overlap` 的配置列表。
    """
    print(f"\n{'配置':^30} {'chunk数':>7} {'最短':>6} {'最长':>6} {'平均':>6}")
    print("-" * 60)
    for cfg in configs:
        chunks = split_into_chunks(entries, **cfg)
        lengths = [len(c.page_content) for c in chunks]
        label = f"size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}"
        print(
            f"{label:^30} {len(chunks):>7} "
            f"{min(lengths):>6} {max(lengths):>6} {sum(lengths)//len(lengths):>6}"
        )


def rebuild_index(
    entries: list[dict],
    embeddings: OpenAIEmbeddings,
    chunk_size: int,
    chunk_overlap: int,
) -> FAISS:
    """用指定配置重新构建向量库（覆盖磁盘上的旧索引）。

    Args:
        entries: `load_shuoshuo` 返回的条目列表。
        embeddings: Embedding 实例。
        chunk_size: 切分块大小。
        chunk_overlap: 切分重叠字符数。

    Returns:
        重建完成的 `FAISS` 向量库。
    """
    chunks = split_into_chunks(entries, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"使用 chunk_size={chunk_size}, overlap={chunk_overlap} → {len(chunks)} 个 chunks")
    return build_vectorstore(chunks, embeddings)


if __name__ == "__main__":
    entries = load_shuoshuo(DATA_FILE)
    print(f"共读取 {len(entries)} 条说说")

    # ── 第一步：先看不同配置的切分统计，零 API 消耗 ──────────────────
    configs = [
        {"chunk_size": 100, "chunk_overlap": 0},
        {"chunk_size": 100, "chunk_overlap": 30},
        {"chunk_size": 300, "chunk_overlap": 50},
        {"chunk_size": 500, "chunk_overlap": 50},   # 当前线上配置
        {"chunk_size": 500, "chunk_overlap": 150},
        {"chunk_size": 1000, "chunk_overlap": 100},
    ]
    analyze_chunk_configs(entries, configs)

    # ── 第二步：加载向量库并检索，观察 chunk 内容对结果的影响 ─────────
    embeddings = build_embeddings()

    # 若想用新配置重建，取消下面注释并修改参数（会覆盖磁盘索引）：
    # vectorstore = rebuild_index(entries, embeddings, chunk_size=100, chunk_overlap=30)

    if FAISS_INDEX_DIR.exists():
        print("\n检测到本地向量库，直接加载...")
        vectorstore = load_vectorstore(embeddings)
    else:
        vectorstore = build_vectorstore(split_into_chunks(entries), embeddings)

    query = "难过"
    print(f"\n{'='*60}")
    print(f"查询：{query!r}   search_type=similarity")
    for doc in vectorstore.similarity_search(query, k=4):
        print(f"  [{doc.metadata['date']}] ({len(doc.page_content):>3}字) {doc.page_content[:60]}")

    print(f"\n查询：{query!r}   search_type=mmr  lambda=0.5")
    for doc in vectorstore.max_marginal_relevance_search(query, k=4, fetch_k=20, lambda_mult=0.5):
        print(f"  [{doc.metadata['date']}] ({len(doc.page_content):>3}字) {doc.page_content[:60]}")

