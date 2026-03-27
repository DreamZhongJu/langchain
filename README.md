# LangChain 源码学习路线 —— RAG + Agent 实战专家养成计划

> 本文件是个人学习记录，用于从零开始系统性地理解 LangChain Agent 模块源码。
> 原始 README 已备份至 [README.backup.md](README.backup.md)。
>
> **注意：本项目使用 `langchain_v1`（v1.x），底层基于 LangGraph，与网上大多数教程的旧版 API（`AgentExecutor` + `create_react_agent`）不同。**

---

## 目标

- 深入理解 LangChain v1 Agent 模块的设计与实现
- 掌握 RAG（检索增强生成）的原理与工程实践
- 能够阅读、修改、扩展 LangChain 源码

---

## 学习路线总览

```text
阶段一：环境 & 基础概念        ✅ 完成
阶段二：跑通第一个 Agent       ✅ 完成
阶段三：理解 Agent 执行流程    进行中
阶段四：深入源码               未开始
阶段五：RAG 原理与实践         未开始
阶段六：RAG + Agent 融合       未开始
阶段七：自己扩展与实战         未开始
```

---

## 阶段一：环境搭建 & 基础概念 ✅

### TODO

- [x] **1.1** 安装 Python 3.11+，安装 `uv`（项目使用的包管理工具）

- [x] **1.2** 在 `libs/langchain_v1/` 目录下执行依赖安装

  ```bash
  cd libs/langchain_v1
  uv sync --all-groups
  ```

- [x] **1.3** 注册并获取 DeepSeek API Key，配置到 `my_learning/.env`

- [x] **1.4** 理解以下核心概念：
  - [x] 什么是 LLM？什么是 Chat Model？
  - [x] 什么是 Prompt Template？
  - [x] 什么是 Chain？
  - [x] 什么是 Tool？
  - [x] 什么是 Agent？

---

## 阶段二：跑通第一个 Agent ✅

### TODO

- [x] **2.1** 新建 `libs/langchain_v1/my_learning/` 目录

- [x] **2.2** 写最简单的 LLM 调用，验证 API Key 可用（`my_learning/01_hello_llm.py`）

  ```python
  import os
  from pathlib import Path

  from dotenv import load_dotenv
  from langchain_openai import ChatOpenAI

  load_dotenv(Path(__file__).parent / ".env")

  llm = ChatOpenAI(
      model="deepseek-chat",
      api_key=os.environ["DEEPSEEK_API_KEY"],  # type: ignore
      base_url="https://api.deepseek.com",
  )
  print(llm.invoke("你好，介绍一下你自己"))
  ```

- [x] **2.3** 写一个带两个 Tool 的 Agent（`my_learning/02_react_agent.py`）

  **关键认知：新版 API 和旧版的本质区别**

  | 对比项 | 旧版 `AgentExecutor`（教程常见） | 新版 `create_agent`（本项目） |
  | --- | --- | --- |
  | LLM 决策 | 输出纯文本 `Thought/Action/Action Input` | 输出结构化 `tool_calls` 对象 |
  | Tool 结果 | 文本 `Observation:` | `ToolMessage` 对象 |
  | 终止信号 | LLM 输出 `Final Answer:` | `finish_reason: stop`，`tool_calls=[]` |
  | 底层 | 自定义 Python 循环 | LangGraph 状态图 |

  ```python
  import os
  from datetime import datetime
  from pathlib import Path

  from dotenv import load_dotenv
  from langchain.agents import create_agent
  from langchain_core.tools import tool
  from langchain_openai import ChatOpenAI

  load_dotenv(Path(__file__).parent / ".env")
  ```


  @tool
  def get_current_time(query: str) -> str:
      """获取当前的日期和时间。当用户询问现在几点、今天几号时使用此工具。"""
      return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


  @tool
  def calculate(expression: str) -> str:
      """计算一个数学表达式，例如 '100 * 99' 或 '(3 + 5) * 2'。
      输入必须是合法的数学表达式字符串。"""
      try:
          result = eval(expression)  # noqa: S307
          return str(result)
      except Exception as e:
          return f"计算出错：{e}"


  llm = ChatOpenAI(
      model="deepseek-chat",
      api_key=os.environ["DEEPSEEK_API_KEY"],  # type: ignore
      base_url="https://api.deepseek.com",
  )

  tools = [get_current_time, calculate]
  agent = create_agent(llm, tools, debug=True)

  print("=== 测试 1：需要调用 Tool ===")
  result = agent.invoke({"messages": [{"role": "user", "content": "现在几点？"}]})
  print("最终答案：", result["messages"][-1].content)

  print("\n=== 测试 2：需要计算 ===")
  result = agent.invoke({"messages": [{"role": "user", "content": "100 乘以 99 等于多少？"}]})
  print("最终答案：", result["messages"][-1].content)

  print("\n=== 测试 3：不需要 Tool ===")
  result = agent.invoke({"messages": [{"role": "user", "content": "天空为什么是蓝色的？"}]})
  print("最终答案：", result["messages"][-1].content)
  ```

  运行：

  ```powershell
  cd libs/langchain_v1
  uv run python my_learning/02_react_agent.py
  ```

- [x] **2.4** 用三类问题测试 Agent：需要 Tool / 需要计算 / 不需要 Tool

---

## 阶段三：理解 Agent 执行流程（不读源码，靠观察）

> 先从行为推断机制，再用源码验证。

### TODO

- [x] **3.1** 对照 `debug=True` 的输出，在纸上画出一次完整执行的流程图：
  - `HumanMessage` 进入
  - → `model` 节点（LLM）处理
  - → LLM 返回带 `tool_calls` 的 `AIMessage`
  - → `tools` 节点执行 Tool，返回 `ToolMessage`
  - → `model` 节点再次处理（带上 ToolMessage）
  - → LLM 返回不带 `tool_calls` 的 `AIMessage`（结束）

- [x] **3.2** 回答以下问题（先猜，再从 debug 输出验证）：
  - [x] 每次调用 LLM 时，历史消息是怎么传进去的？（看 `[values]` 里 messages 列表的增长）
  - [x] LLM 怎么"知道"有哪些 Tool 可以用？（Tool 的 docstring 在哪里被发送给 LLM？）
  - [x] `ToolMessage` 的 `tool_call_id` 和上一条 `AIMessage` 的 `tool_calls[0]['id']` 有什么关系？为什么需要这个 id？
  - [x] 测试 3（不需要 Tool）的 debug 输出比测试 1/2 少了哪个节点？

- [x] **3.3** 修改 `02_react_agent.py`，在调用后打印完整的 messages 列表：

  ```python
  for msg in result["messages"]:
      print(type(msg).__name__, ":", msg.content[:80] if msg.content else "[no content]")
  ```

  理解每条消息的类型和顺序。

### 本阶段检验

不看代码，只用文字描述："调用 `agent.invoke(...)` 后，内部经历了哪些步骤，数据是以什么格式流转的？"

---

## 阶段四：深入源码

> 现在你有了足够的"使用者直觉"，开始读源码验证你的猜测。

### TODO

#### 4.1 找到并读 `create_agent` 的实现

- [x] 找到 `create_agent` 函数定义（[factory.py:673](libs/langchain_v1/langchain/agents/factory.py#L673)）
- [x] 理解它返回的是什么类型的对象：`CompiledStateGraph`（蓝图 → `.compile()` → 可运行的机器）
- [x] 找到构建 LangGraph 的关键代码：
  - `StateGraph(...)` 创建空白蓝图（L1020）
  - `graph.add_node("model", ...)` 加 LLM 节点（L1356）
  - `graph.add_node("tools", ...)` 加 Tool 节点（L1360）
  - `graph.add_conditional_edges(...)` 加条件跳转（L1510）
  - `graph.compile(...)` 将蓝图编译为可执行对象（L1637）

#### 4.2 理解 LangGraph 的两个核心节点

- [x] 找到"循环还是终止"的判断逻辑：`_make_model_to_tools_edge`（[factory.py:1684](libs/langchain_v1/langchain/agents/factory.py#L1684)）

  每次 LLM 响应后执行，按顺序判断：

  1. 如果 state 里有 `jump_to`（主动跳转指令）→ 按指令跳
  2. 如果没有 `AIMessage`（消息被清空）→ 退出循环
  3. **如果 `tool_calls` 为空 → 退出循环**（这是正常结束的条件）
  4. 如果有待执行的 `tool_calls` → 发送到 `tools` 节点执行
  5. 如果有结构化输出结果 → 退出循环
  6. 其他（人工注入了 ToolMessage）→ 跳回 `model` 节点

- [x] 找到 `model` 节点的执行函数：`model_node`（[factory.py:1287](libs/langchain_v1/langchain/agents/factory.py#L1287)）
  - 输入：`state["messages"]`（全部历史消息），若有 `system_prompt` 则插到最前
  - 输出：新的 `AIMessage`（含或不含 `tool_calls`）
- [x] 找到 `tools` 节点的执行函数：`ToolNode`（来自 LangGraph，黑盒）
  - 读取 `AIMessage.tool_calls`，找到对应函数执行，返回 `ToolMessage` 列表
  - 若 `available_tools` 为空，`tool_node = None`，图里根本不会有 `tools` 节点

#### 4.3 理解 Tool 是如何注册和被调用的

- [x] `@tool` 装饰器把函数变成了 `BaseTool` 对象（[tools/convert.py:88](libs/core/langchain_core/tools/convert.py#L88)），`name` 取函数名，`description` 取 docstring
- [x] LLM 通过 `bind_tools` 知道有哪些工具：[factory.py:1219](libs/langchain_v1/langchain/agents/factory.py#L1219) 把 `final_tools` 绑定到 LLM，每次请求都会携带
- [x] docstring 最终转成 JSON Schema 随 API 请求发给 LLM，可用以下代码验证：

  ```python
  import json
  print(json.dumps(llm.bind_tools(tools).kwargs["tools"], ensure_ascii=False, indent=2))
  ```

#### 4.4 对比：为什么新版不需要 Output Parser？

- [x] 旧版需要 OutputParser：LLM 输出纯文本 `"Action: calculate\nAction Input: 100*99"`，需要正则解析才能知道调哪个工具
- [x] 新版不需要：LLM 直接返回结构化 `tool_calls` 字段，`model_node` 里的 `_handle_model_output`（factory.py:1326）直接读取，无需解析

### 本阶段检验

不看文档，只看源码，能回答："调用 `agent.invoke({"messages": [...]})` 时，从入口到最终返回，代码走过了哪些函数？"

**参考答案（用自己的话写一遍）：**

```text
agent.invoke({"messages": [...]})
  → StateGraph 路由到 model 节点
  → model_node：取 state["messages"]，bind_tools 后调用 LLM
  → LLM 返回 AIMessage（含 tool_calls）
  → _make_model_to_tools_edge 判断：tool_calls 非空 → 路由到 tools 节点
  → ToolNode：执行对应函数，返回 ToolMessage
  → 路由回 model 节点，把 ToolMessage 加入 messages 再次调用 LLM
  → LLM 返回 AIMessage（tool_calls 为空）
  → _make_model_to_tools_edge 判断：tool_calls 为空 → 路由到 END
  → 返回最终 state["messages"]
```

---

## 阶段五：RAG 原理与实践

### TODO

#### 5.1 理解 RAG 的核心概念

- [x] 什么是 Embedding（向量嵌入）？为什么能用来做相似度搜索？
- [x] 什么是 Vector Store（向量数据库）？
- [x] RAG 解决了 LLM 的什么核心问题（知识截止 / 私有数据）？

#### 5.2 动手实现一个最简单的 RAG（`my_learning/03_basic_rag.py`）

- [x] 选择一个本地文档（QQ空间说说备份，593 条，2011–2026 年）
- [x] 用 `RecursiveCharacterTextSplitter` 把文档切成 chunks
- [x] 用 Embedding 模型生成向量，存入 `FAISS`（ModelScope Qwen3-Embedding-0.6B）
- [x] 给定问题，检索最相关 chunks，再让 LLM 生成答案

#### 5.3 读 RAG 相关源码

- [x] 找到 `RecursiveCharacterTextSplitter` 的实现，理解切分规则

  **切分规则**（`libs/text-splitters/langchain_text_splitters/character.py:107`）：

  两步机制——①贪心选分隔符：遍历 `separators` 列表，找第一个在文本中能匹配到的，
  用它切一刀；切完仍然太大的片段，换下一个分隔符递归处理，最终兜底用 `""` 逐字切。
  ②滑动窗口合并（`base.py:152` `_merge_splits`）：把小片段逐个装入窗口，超过
  `chunk_size` 就输出；输出后从窗口头部 pop 直到长度 ≤ `chunk_overlap`，
  剩余内容作为下一 chunk 的开头，这就是"重叠"的物理来源。

- [x] 找到 `VectorStoreRetriever.invoke` 方法，理解检索流程

  ```text
  retriever.invoke(query)                    # retrievers.py:179
    → 配置回调（LangSmith 追踪）
    → _get_relevant_documents()              # vectorstores/base.py:1040
        ├─ similarity       → similarity_search()
        ├─ mmr              → max_marginal_relevance_search()
        └─ score_threshold  → similarity_search_with_relevance_scores()
    → 触发 on_retriever_end 回调 → 返回 list[Document]
  ```

  `vectorstore.similarity_search()` 是绕过 Retriever 层的直接调用；
  接 `create_retrieval_chain()` 时必须用 `as_retriever()` 形式。

- [x] 理解 `similarity_search` 和 `max_marginal_relevance_search`（MMR）的区别

  | | similarity_search | MMR |
  | --- | --- | --- |
  | 选择标准 | 按余弦距离降序取前 K 个 | `λ × 相关性 − (1−λ) × 冗余惩罚` |
  | 结果特点 | 最相关，但可能重复 | 相关且多样 |
  | 关键参数 | `k` | `k`, `fetch_k`, `lambda_mult` |

  MMR 算法（`vectorstores/utils.py:149`）：先捞 `fetch_k` 个候选，然后贪心逐个挑选，
  每轮把"与已选文档的最大相似度"作为惩罚项；`lambda_mult=1` 退化为纯相似度排序，
  `lambda_mult=0` 只追求多样性。

#### 5.4 实验：RAG 参数对结果的影响

- [x] 调整 `chunk_size` 和 `chunk_overlap`，观察检索质量变化

  测试数据：547 条说说，平均原始长度约 **124 字**。

  | chunk_size | chunk_overlap | chunk 数 | 平均长度 |
  | --- | --- | --- | --- |
  | 100 | 0 | 1184 | 56 字 |
  | 100 | 30 | 1227 | 60 字 |
  | 300 | 50 | 632 | 109 字 |
  | **500** | **50** | **562** | **121 字**（当前配置） |
  | 500 | 150 | 562 | 122 字 |
  | 1000 | 100 | 548 | 124 字 |

  **结论：**
  1. `chunk_size` 要小于原文平均长度才有实际切分效果。本数据集说说均长 124 字，`chunk_size=500` 相当于几乎不切，从 500 增大到 1000 chunk 数几乎不变（562→548）。
  2. `chunk_overlap` 在短文本上效果有限。overlap 从 50 改到 150，chunk 数完全不变；overlap 在长文档（论文、手册）中才更有价值。
  3. chunk 太大会导致返回重复内容。`similarity_search` 会返回同一条说说的多个相邻 chunk，可用 MMR 或缩小 `chunk_size` 缓解。
  4. 对说说这类情感同质化强的数据，MMR 更适合——能跨年份召回多样结果，避免返回一堆同时期的雷同内容。

- [ ] 调整 `k`（返回的 chunk 数量），观察 LLM 回答的变化

### 本阶段检验

能独立搭建一个"问自己笔记"的 RAG 系统，并解释每一步在做什么。

---

## 阶段六：RAG + Agent 融合

### TODO

- [x] **6.1** 把 RAG 检索封装成一个 Tool（`my_learning/04_rag_agent.py`）：

  ```python
  @tool
  def search_my_notes(query: str) -> str:
      """搜索用户的个人笔记库，包含：个人成长经历与回忆（高中、大学、暑假等各阶段）、
      读书笔记（如《小狗钱钱》）、学习心得（英语、蓝桥杯、CET-6、口播练习等）、
      AI 问答记录、环境搭建教程等。
      当问题涉及用户的个人经历、学习记录或私有知识时，必须调用此工具，不得凭空回答。
      """
      results = vectorstore.max_marginal_relevance_search(query, k=5, fetch_k=20, lambda_mult=0.7)
      return "\n\n---\n\n".join(
          [f"[来源：{doc.metadata['source']}]\n{doc.page_content}" for doc in results]
      )
  ```

  **调试过程中发现的问题：**
  - tool docstring 模糊（只写"学习笔记"）→ LLM 不确定何时调用，导致 Agent 绕过 Tool 自行回答
  - `k=3` 不够 → 增大到 5，且改用 MMR 避免反复返回同一个文件的内容
  - 极短文件（`环境搭建.md`，105 字）切成单个小 chunk，余弦相似度对各种查询偏高，导致检索被干扰
  - 加 `system_prompt` 明确告知 Agent"这是你的私人笔记库，涉及个人经历必须先检索"后触发率显著提高

- [x] **6.2** 把这个 Tool 加入 `create_agent`，Agent 自主决定什么时候检索
- [x] **6.3** 测试：让 Agent 回答一个只有笔记里才有答案的问题
- [x] **6.4** 实验：修改 Tool 的 docstring，观察对检索触发率的影响

### 本阶段检验

构建一个能回答私有知识库问题的 RAG + Agent 系统，并理解每一行代码。

---

## 阶段七：扩展与实战

### TODO

- [x] **7.1** 实现一个真实有用的自定义 Tool（查天气、读本地文件、调用某个 API）

- [x] **7.2** 给 Agent 添加对话历史记忆（使用 LangGraph 的 `checkpointer`）：

  ```python
  from langgraph.checkpoint.memory import MemorySaver

  agent = create_agent(llm, tools, checkpointer=MemorySaver())
  # 用 config={"configurable": {"thread_id": "..."}} 区分不同会话
  ```

- [x] **7.3** 用 `system_prompt` 参数给 Agent 设定角色和行为限制

- [x] **7.4** 实现一个多 Agent 协作的系统（一个 Agent 负责规划，一个负责执行）

- [x] **7.5** 阅读 `factory.py` 的 `create_agent` 源码，自己用 LangGraph 从零手写一个等价的 Agent

---

## 阶段八：源码改造实验

> 真正读懂源码的标志：能改动它，并预测改动的效果。

### TODO

- [ ] **8.1** 写一份"LangChain v1 Agent 执行流程源码解析"文档
  - 从 `create_agent` 到 `model_node`、`tools_node`、条件边判断的完整路径
  - 画出 LangGraph 状态流转图，解释每一步的设计意图
  - 这是面试时展示"源码级理解"的核心素材

- [ ] **8.2** 做一个源码小改动并验证
  - 修改 `_make_model_to_tools_edge` 的退出逻辑（[factory.py:1684](libs/langchain_v1/langchain/agents/factory.py#L1684)），让 Agent 连续调用工具超过 N 次后强制停止
  - 重新编译运行测试，验证效果
  - 目标：证明自己不只是"读懂了"，而是"能改动它"

---

## 阶段九：RAG 系统打磨

> 从"能跑"到"能用"。

### TODO

- [ ] **9.1** 增强 `search_my_notes` Tool 的鲁棒性
  - 用 `similarity_search_with_score` 获取相关性分数
  - 当最高分低于阈值时，返回"笔记中没有相关信息"，而不是低质量的检索结果
  - 处理检索结果为空的边界情况

- [ ] **9.2** 接入 `create_history_aware_retriever`
  - 解决传统 RAG 不感知对话历史的问题
  - 测试场景："我去年暑假做了什么？" → "那后来呢？" 验证 Agent 能否正确理解"那"指代的是暑假经历

- [ ] **9.3** 测试并优化多轮对话场景
  - 当前问题：第二轮会把全部历史 token 重新发送，成本高
  - 探索：planner 如何判断"这一轮能直接用上一轮的检索结果回答"

---

## 阶段十：完整项目与技术输出

> 把学习笔记变成作品集，把研究变成影响力。

### TODO

#### 10.1 个人知识库 Agent（核心项目）

- [ ] 结合 KG-RAG 方向，设计"向量检索 + 知识图谱"双轨检索架构
  - 向量检索（FAISS）：回答"我高中做了什么"这类语义问题
  - 知识图谱（Neo4j）：回答"我哪一年拿了蓝桥杯国奖"这类精确关系查询
  - Agent 自主决定调哪个 Tool
- [ ] 技术栈：LangChain v1 + Neo4j + FAISS + LangGraph + Streamlit（界面）
- [ ] 产出指标：检索准确率、Agent 任务完成率

#### 10.2 开源与公开

- [ ] 创建 GitHub 仓库，整理现有代码和学习笔记，写清晰的 README（含架构图）
- [ ] 写技术文章：《LangChain v1 Agent 源码深度解析》，发布到掘金 / 知乎
- [ ] 写技术文章：《手把手教你构建一个能聊天的个人知识库 Agent》
- [ ] 尝试向 LangChain 提交一个 PR（改进文档或修复小 bug）

#### 10.3 面试素材准备

- [ ] 整理"复杂项目"的面试表述：需求 → 难点 → 解决方案 → 效果
- [ ] 准备 LangGraph 状态流转图，能在面试中白板讲清楚调用链

---

## 学习笔记区

### 疑问

- （在这里记录你读代码时不理解的地方）

### 关键发现

- `create_agent` 本质是用 LangGraph 画了一张状态图：`StateGraph` 是蓝图，`.compile()` 才是可运行的对象
- Agent 循环的终止条件在 `_make_model_to_tools_edge`（factory.py:1684）：**LLM 返回的 `tool_calls` 为空时退出**
- `loop_exit_node` 不是真实节点，是个变量，指向"每次循环最后经过的节点名"（默认是 `"model"`）
- `pending_tool_calls` 是过滤掉已执行的 tool_calls 后剩下的——防止重复执行同一个 Tool

### 踩过的坑

- DeepSeek API Key 不能硬编码在代码里，改用 `.env` + `load_dotenv`
- `langchain_v1` v1.x 的 `create_agent` 底层是 LangGraph，与网上旧版教程 API 不同
- VS Code Pylance 报错不代表运行出错，需确认 Python 解释器指向 `.venv`

### 传统 RAG 的局限性

传统 RAG（向量检索 + LLM 生成）本质上是**关键词语义匹配**，存在三个固有缺陷：

1. **依赖 Embedding 模型的质量**
   检索效果的上限由向量模型决定。小模型（如 0.6B）对中文语义的对齐能力弱，
   同样的问题用 4B 模型就能一次命中，用 0.6B 需要 Agent 反复重试。
   向量模型无法理解问题的"意图"，只能计算字面语义的余弦距离。

2. **缺乏上下文感知能力**
   每次检索都是独立的单次查询，不感知对话历史。
   用户问"他后来怎么了"，RAG 不知道"他"指谁，只能用这四个字去检索，必然失败。
   需要额外的 `create_history_aware_retriever` 先让 LLM 把问题改写成独立查询，才能解决。

3. **chunk 切分破坏了长程上下文**
   文档被切成固定大小的 chunk 后，跨 chunk 的推理关系就断了。
   比如"第一章提到的概念在第三章的应用"这类问题，单个 chunk 无法覆盖，
   靠 `chunk_overlap` 只能缓解相邻 chunk 的断裂，跨越多 chunk 的语义无能为力。

---

## 参考资源

- [LangChain 官方文档](https://docs.langchain.com/oss/python/langchain/overview)
- [LangChain API Reference](https://reference.langchain.com/python)
- [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph/overview)
- 核心源码入口：[libs/langchain_v1/langchain/agents/factory.py](libs/langchain_v1/langchain/agents/factory.py)
