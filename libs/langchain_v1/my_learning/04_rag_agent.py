import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(Path(__file__).parent / ".env")

NOTES_DIR = Path(__file__).parent / "data" / "有用的知识以及心得"
FAISS_INDEX_DIR = Path(__file__).parent / "data" / "faiss_notes_index"

MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def load_notes(notes_dir: Path) -> list[Document]:
    """递归加载目录下所有 .md 文件，metadata 中自动记录文件路径。

    Args:
        notes_dir: 笔记根目录。

    Returns:
        每个文件对应一个 `Document`，`metadata["source"]` 为文件路径。
    """
    loader = DirectoryLoader(
        str(notes_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    # 把 source 路径改为相对路径，方便阅读
    for doc in docs:
        doc.metadata["source"] = Path(doc.metadata["source"]).relative_to(notes_dir).as_posix()
    return docs


def split_docs(docs: list[Document]) -> list[Document]:
    """将文档切分成适合检索的 chunks。

    Args:
        docs: `load_notes` 返回的文档列表。

    Returns:
        切分后的 `Document` 列表。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", ""],
    )
    return splitter.split_documents(docs)


def build_embeddings() -> OpenAIEmbeddings:
    """创建指向 ModelScope 的 Embedding 实例。

    Returns:
        配置好 ModelScope endpoint 的 `OpenAIEmbeddings`。
    """
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.environ["MODELSCOPE_API_KEY"],  # type: ignore
        openai_api_base=MODELSCOPE_BASE_URL,  # type: ignore
    )


def build_vectorstore(chunks: list[Document], embeddings: OpenAIEmbeddings) -> FAISS:
    """将 chunks 向量化并持久化到本地 FAISS 索引。

    Args:
        chunks: 已切分的 `Document` 列表。
        embeddings: Embedding 实例。

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
        embeddings: Embedding 实例。

    Returns:
        加载完成的 `FAISS` 向量库。
    """
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

@tool
def search_my_notes(query: str) -> str:
    """搜索用户的个人笔记库，包含：个人成长经历与回忆（高中、大学、暑假等各阶段）、
    读书笔记（如《小狗钱钱》）、学习心得（英语、蓝桥杯、CET-6、口播练习等）、
    AI 问答记录、环境搭建教程等。
    当问题涉及用户的个人经历、学习记录或私有知识时，必须调用此工具，不得凭空回答。

    Args:
        query: 用于检索的关键词或问题，尽量简洁具体。
    """
    # 用 MMR 避免多次返回同一文件的内容
    results = vectorstore.max_marginal_relevance_search(query, k=5, fetch_k=20, lambda_mult=0.7)
    if not results:
        return "笔记库中未找到相关内容。"
    return "\n\n---\n\n".join(
        [f"[来源：{doc.metadata['source']}]\n{doc.page_content}" for doc in results]
    )


llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],  # type: ignore
    base_url="https://api.deepseek.com",
)

tools = [search_my_notes]

SYSTEM_PROMPT = (
    "你是用户的私人助理，可以访问用户的个人笔记库。"
    "笔记库中包含用户从高中到大学各阶段的真实经历、读书笔记和学习心得。"
    "回答涉及用户个人经历或私有知识的问题时，必须先调用 search_my_notes 工具检索，"
    "再根据检索结果回答，不得凭空捏造。"
)

agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT, debug=True, checkpointer=MemorySaver())

if __name__ == "__main__":
    # ── 第一步：加载所有 .md 文件 ──────────────────────────────────
    docs = load_notes(NOTES_DIR)
    print(f"\n共加载 {len(docs)} 个文件：")
    for doc in docs:
        print(f"  {doc.metadata['source']}  ({len(doc.page_content)} 字)")

    # ── 第二步：切分 ────────────────────────────────────────────────
    chunks = split_docs(docs)
    print(f"\n切分后共 {len(chunks)} 个 chunks")

    # ── 第三步：向量化（有本地索引就直接加载）──────────────────────
    embeddings = build_embeddings()
    if FAISS_INDEX_DIR.exists():
        print("检测到本地向量库，直接加载...")
        vectorstore = load_vectorstore(embeddings)
    else:
        vectorstore = build_vectorstore(chunks, embeddings)


    # 测试 1：需要调用 Tool
    print("=== 测试 1 ===")
    result = agent.invoke({"messages": [{"role": "user", "content": "我大学做了什么"}]})
    print("最终答案：", result["messages"][-1].content)
    for msg in result["messages"]:
        text = (msg.content[:80] if msg.content else "[no content]")
        print(type(msg).__name__, ":", text.encode("gbk", errors="replace").decode("gbk"))
