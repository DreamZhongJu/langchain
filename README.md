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

- [ ] **3.1** 对照 `debug=True` 的输出，在纸上画出一次完整执行的流程图：
  - `HumanMessage` 进入
  - → `model` 节点（LLM）处理
  - → LLM 返回带 `tool_calls` 的 `AIMessage`
  - → `tools` 节点执行 Tool，返回 `ToolMessage`
  - → `model` 节点再次处理（带上 ToolMessage）
  - → LLM 返回不带 `tool_calls` 的 `AIMessage`（结束）

- [ ] **3.2** 回答以下问题（先猜，再从 debug 输出验证）：
  - [ ] 每次调用 LLM 时，历史消息是怎么传进去的？（看 `[values]` 里 messages 列表的增长）
  - [ ] LLM 怎么"知道"有哪些 Tool 可以用？（Tool 的 docstring 在哪里被发送给 LLM？）
  - [ ] `ToolMessage` 的 `tool_call_id` 和上一条 `AIMessage` 的 `tool_calls[0]['id']` 有什么关系？为什么需要这个 id？
  - [ ] 测试 3（不需要 Tool）的 debug 输出比测试 1/2 少了哪个节点？

- [ ] **3.3** 修改 `02_react_agent.py`，在调用后打印完整的 messages 列表：

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

- [ ] 什么是 Embedding（向量嵌入）？为什么能用来做相似度搜索？
- [ ] 什么是 Vector Store（向量数据库）？
- [ ] RAG 解决了 LLM 的什么核心问题（知识截止 / 私有数据）？

#### 5.2 动手实现一个最简单的 RAG（`my_learning/03_basic_rag.py`）

- [ ] 选择一个本地文档（任意 .txt 文件）
- [ ] 用 `RecursiveCharacterTextSplitter` 把文档切成 chunks
- [ ] 用 Embedding 模型生成向量，存入 `FAISS` 或 `Chroma`
- [ ] 给定问题，检索最相关 chunks，再让 LLM 生成答案

#### 5.3 读 RAG 相关源码

- [ ] 找到 `RecursiveCharacterTextSplitter` 的实现，理解切分规则
- [ ] 找到 `VectorStoreRetriever.invoke` 方法，理解检索流程
- [ ] 理解 `similarity_search` 和 `max_marginal_relevance_search`（MMR）的区别

#### 5.4 实验：RAG 参数对结果的影响

- [ ] 调整 `chunk_size` 和 `chunk_overlap`，观察检索质量变化
- [ ] 调整 `k`（返回的 chunk 数量），观察 LLM 回答的变化

### 本阶段检验

能独立搭建一个"问自己笔记"的 RAG 系统，并解释每一步在做什么。

---

## 阶段六：RAG + Agent 融合

### TODO

- [ ] **6.1** 把 RAG 检索封装成一个 Tool（`my_learning/04_rag_agent.py`）：

  ```python
  @tool
  def search_my_notes(query: str) -> str:
      """从我的学习笔记中检索相关信息。当问题涉及特定知识时使用此工具。"""
      # 调用你在阶段五实现的检索逻辑
      ...
  ```

- [ ] **6.2** 把这个 Tool 加入 `create_agent`，Agent 自主决定什么时候检索
- [ ] **6.3** 测试：让 Agent 回答一个只有笔记里才有答案的问题
- [ ] **6.4** 实验：修改 Tool 的 docstring，观察对检索触发率的影响

### 本阶段检验

构建一个能回答私有知识库问题的 RAG + Agent 系统，并理解每一行代码。

---

## 阶段七：扩展与实战

### TODO

- [ ] **7.1** 实现一个真实有用的自定义 Tool（查天气、读本地文件、调用某个 API）

- [ ] **7.2** 给 Agent 添加对话历史记忆（使用 LangGraph 的 `checkpointer`）：

  ```python
  from langgraph.checkpoint.memory import MemorySaver

  agent = create_agent(llm, tools, checkpointer=MemorySaver())
  # 用 config={"configurable": {"thread_id": "..."}} 区分不同会话
  ```

- [ ] **7.3** 用 `system_prompt` 参数给 Agent 设定角色和行为限制

- [ ] **7.4** 实现一个多 Agent 协作的系统（一个 Agent 负责规划，一个负责执行）

- [ ] **7.5** 阅读 `factory.py` 的 `create_agent` 源码，自己用 LangGraph 从零手写一个等价的 Agent

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

---

## 参考资源

- [LangChain 官方文档](https://docs.langchain.com/oss/python/langchain/overview)
- [LangChain API Reference](https://reference.langchain.com/python)
- [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph/overview)
- 核心源码入口：[libs/langchain_v1/langchain/agents/factory.py](libs/langchain_v1/langchain/agents/factory.py)
