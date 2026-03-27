import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(Path(__file__).parent / ".env")

NOTES_DIR = Path(__file__).parent / "data" / "有用的知识以及心得"
FAISS_INDEX_DIR = Path(__file__).parent / "data" / "faiss_notes_index"
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


# ── 向量库相关（复用 04_rag_agent.py 的逻辑）────────────────────────────────

def load_notes(notes_dir: Path) -> list[Document]:
    """递归加载目录下所有 .md 文件。

    Args:
        notes_dir: 笔记根目录。

    Returns:
        每个文件对应一个 `Document`，`metadata["source"]` 为相对路径。
    """
    loader = DirectoryLoader(
        str(notes_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = Path(doc.metadata["source"]).relative_to(notes_dir).as_posix()
    return docs


def split_docs(docs: list[Document]) -> list[Document]:
    """将文档切分成适合检索的 chunks。

    Args:
        docs: 原始文档列表。

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


def init_vectorstore() -> FAISS:
    """加载或构建 FAISS 向量库。

    Returns:
        可用的 `FAISS` 向量库。
    """
    embeddings = build_embeddings()
    if FAISS_INDEX_DIR.exists():
        print("检测到本地向量库，直接加载...")
        return FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    print("构建向量库中...")
    docs = load_notes(NOTES_DIR)
    chunks = split_docs(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(FAISS_INDEX_DIR))
    return vs


# ── 工具定义（search_my_notes 在运行时才访问 vectorstore）────────────────────

# 占位，在 __main__ 里赋值
vectorstore: FAISS

@tool
def search_my_notes(query: str) -> str:
    """搜索用户的个人笔记库，包含：个人成长经历与回忆（高中、大学、暑假等各阶段）、
    读书笔记（如《小狗钱钱》）、学习心得（英语、蓝桥杯、CET-6、口播练习等）、
    AI 问答记录、环境搭建教程等。
    当问题涉及用户的个人经历、学习记录或私有知识时，必须调用此工具。

    Args:
        query: 单个具体的检索关键词，如"大学学习方法"。
    """
    results = vectorstore.max_marginal_relevance_search(query, k=5, fetch_k=20, lambda_mult=0.7)
    if not results:
        return "笔记库中未找到相关内容。"
    return "\n\n---\n\n".join(
        [f"[来源：{doc.metadata['source']}]\n{doc.page_content}" for doc in results]
    )


# ── LLM ─────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],  # type: ignore
    base_url="https://api.deepseek.com",
)


# ── 执行器 Agent：只管检索，无需记忆 ────────────────────────────────────────

EXECUTOR_PROMPT = (
    "你是检索执行器。接收一个具体的查询，调用 search_my_notes 工具检索笔记库，"
    "将检索到的原始内容整理后返回，不要添加额外推断。"
)

executor_agent = create_agent(llm, [search_my_notes], system_prompt=EXECUTOR_PROMPT)


# ── 把执行器包装成 Tool 给规划器调用 ────────────────────────────────────────

@tool
def execute_search(query: str) -> str:
    """向执行器 Agent 发送一个子查询，返回从笔记库检索到的内容片段。
    每次只传入一个具体的查询，不要把多个问题合并在一起。

    Args:
        query: 单个具体的检索查询，如"大一上学期的学习状态"。
    """
    result = executor_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content


# ── 规划器 Agent：拆问题 + 综合结果 + 多轮记忆 ──────────────────────────────

PLANNER_PROMPT = (
    "你是用户的私人助理兼知识规划器。\n"
    "收到用户的问题后，最多调用 2 次 execute_search 工具检索，然后综合结果回答。\n"
    "注意：\n"
    "- 每个子查询要具体，避免重复\n"
    "- 你有对话记忆，可以直接引用之前轮次中已检索到的信息，无需重复检索\n"
    "- 如果问题在之前对话中已有答案，直接回答，不要再调用工具\n"
    "- 严格限制：最多调用 2 次 execute_search，不得超过"
)

planner_agent = create_agent(
    llm,
    [execute_search],
    system_prompt=PLANNER_PROMPT,
    checkpointer=MemorySaver(),  # 多轮记忆
    debug=False,                 # 关掉 debug 减少噪音
)

# recursion_limit 限制 LangGraph 内部节点跳转次数，防止无限循环
# 每次 execute_search 调用约消耗 4 次跳转，上限设 20 约等于最多 4 次工具调用
CHAT_CONFIG: RunnableConfig = {
    "configurable": {"thread_id": "main-session"},
    "recursion_limit": 20,
}


# ── 多轮交互主循环 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    vectorstore = init_vectorstore()

    print("\n笔记库已就绪。输入问题开始对话，输入 'quit' 退出。\n")

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "退出"):
            print("再见！")
            break

        result = planner_agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=CHAT_CONFIG,
        )

        answer = result["messages"][-1].content
        print(f"\nAssistant：{answer}\n")
        print("-" * 60)
