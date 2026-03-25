# LangChain 源码学习路线 —— RAG + Agent 实战专家养成计划

> 本文件是个人学习记录，用于从零开始系统性地理解 LangChain Agent 模块源码。
> 原始 README 已备份至 [README.backup.md](README.backup.md)。

---

## 目标

- 深入理解 LangChain Agent 模块的设计与实现
- 掌握 RAG（检索增强生成）的原理与工程实践
- 能够阅读、修改、扩展 LangChain 源码

---

## 学习路线总览

```
阶段一：环境 & 基础概念        （约 1-2 天）
阶段二：跑通第一个 Agent       （约 1-2 天）
阶段三：理解 Agent 执行流程    （约 2-3 天）
阶段四：深入源码               （约 3-5 天）
阶段五：RAG 原理与实践         （约 3-5 天）
阶段六：RAG + Agent 融合       （约 2-3 天）
阶段七：自己扩展与实战         （持续）
```

---

## 阶段一：环境搭建 & 基础概念

### TODO

- [ ] **1.1** 安装 Python 3.11+，安装 `uv`（项目使用的包管理工具）
  - 验证：`uv --version` 能正常输出
- [ ] **1.2** 在 `libs/langchain_v1/` 目录下执行依赖安装
  ```bash
  cd libs/langchain_v1
  uv sync --all-groups
  ```
- [ ] **1.3** 注册并获取一个 LLM API Key（推荐 OpenAI 或 DeepSeek，后者更便宜）
- [ ] **1.4** 理解以下核心概念（先查文档，后续用源码验证）：
  - [ ] 什么是 LLM？什么是 Chat Model？
  - [ ] 什么是 Prompt Template？
  - [ ] 什么是 Chain？
  - [ ] 什么是 Tool？
  - [ ] 什么是 Agent？

### 本阶段检验

能用 3 句话解释："LangChain 里的 Agent 是什么，它和普通的 LLM 调用有什么区别？"

---

## 阶段二：跑通第一个 Agent

### TODO

- [ ] **2.1** 在 `libs/langchain_v1/` 下新建 `my_learning/` 目录，作为你的实验场
- [ ] **2.2** 写一个最简单的 LLM 调用（不带 Agent），验证 API Key 可用
  ```python
  # my_learning/01_hello_llm.py
  from langchain_openai import ChatOpenAI
  llm = ChatOpenAI(model="gpt-4o-mini")
  print(llm.invoke("你好，介绍一下你自己"))
  ```
- [ ] **2.3** 写一个带两个 Tool 的 ReAct Agent
  - Tool 1：能查询当前时间
  - Tool 2：能做简单的数学计算
  - 开启 `verbose=True`，完整观察每一步的输入输出
- [ ] **2.4** 用不同的问题测试你的 Agent，记录它的行为：
  - [ ] 问它"现在几点？"，看它是否调用了正确的 Tool
  - [ ] 问它"100 乘以 99 等于多少？"
  - [ ] 问它一个不需要 Tool 的问题，看它如何处理

### 本阶段检验

截图或粘贴一次完整的 Agent 运行日志，能指出日志中每个部分的含义。

---

## 阶段三：理解 Agent 执行流程（不读源码，靠观察）

> 先从行为推断机制，再用源码验证你的猜测——这是读源码最高效的方式。

### TODO

- [ ] **3.1** 手画（或用任何工具画）一张 Agent 的执行流程图，包含：
  - 用户输入 → LLM → 输出解析 → 判断是否调用 Tool → 调用 Tool → 结果返回给 LLM → 循环 or 结束
- [ ] **3.2** 回答以下问题（先猜，后验证）：
  - [ ] Agent 的"记忆"（上下文）是怎么传递给 LLM 的？
  - [ ] LLM 的输出是纯文本，Agent 怎么知道它要调用哪个 Tool？
  - [ ] 如果 LLM 调用了一个不存在的 Tool，会发生什么？
  - [ ] `max_iterations` 这个参数是做什么用的？
- [ ] **3.3** 给 Agent 加上 `return_intermediate_steps=True`，打印中间步骤，理解数据结构

### 本阶段检验

能用流程图 + 文字准确描述 Agent 的一次完整执行过程。

---

## 阶段四：深入源码

> 现在你有了足够的"使用者直觉"，可以开始读源码了。

### TODO

#### 4.1 找到关键文件（自己用搜索工具找，不要看答案）

- [ ] 找到 `AgentExecutor` 类的定义文件
- [ ] 找到 `create_react_agent` 函数的定义文件
- [ ] 找到 `Tool` / `BaseTool` 类的定义文件
- [ ] 找到 ReAct 的 Prompt Template 定义

#### 4.2 读 AgentExecutor 的核心循环

- [ ] 找到主执行方法（`invoke` 或 `_call`）
- [ ] 理解循环的终止条件有哪些（至少找到 3 个）
- [ ] 找到"把 Tool 结果拼回 LLM 输入"的代码位置
- [ ] 理解 `agent_scratchpad` 是什么，在哪里被构建

#### 4.3 读 Output Parser

- [ ] 找到 ReAct Agent 的输出解析器（OutputParser）
- [ ] 理解它如何从 LLM 的纯文本输出中提取 Tool 名称和参数
- [ ] 对比 OpenAI Function Calling Agent 的输出解析器有何不同

#### 4.4 对比两种 Agent 实现

- [ ] ReAct Agent：基于文本解析
- [ ] Tool Calling Agent：基于 LLM 的结构化输出
- [ ] 总结：它们的核心区别在哪里？各自适合什么场景？

### 本阶段检验

不看文档，只看源码，能回答："当我调用 `agent_executor.invoke({"input": "..."})` 时，代码执行的完整路径是什么？"

---

## 阶段五：RAG 原理与实践

### TODO

#### 5.1 理解 RAG 的核心概念

- [ ] 什么是 Embedding（向量嵌入）？
- [ ] 什么是 Vector Store（向量数据库）？
- [ ] 什么是相似度检索？
- [ ] RAG 解决了 LLM 的什么问题？

#### 5.2 动手实现一个最简单的 RAG

- [ ] 选择一个本地文档（任意 .txt 文件）
- [ ] 用 `TextSplitter` 把文档切分成 chunks
- [ ] 用 Embedding 模型生成向量，存入 `FAISS` 或 `Chroma`
- [ ] 给定一个问题，检索最相关的 chunks，再让 LLM 生成答案

#### 5.3 读 RAG 相关源码

- [ ] 找到 `RecursiveCharacterTextSplitter` 的实现，理解切分逻辑
- [ ] 找到 `VectorStoreRetriever` 的 `invoke` 方法
- [ ] 理解 `similarity_search` 和 `mmr`（最大边际相关性）的区别

#### 5.4 RAG 质量优化实验

- [ ] 调整 `chunk_size` 和 `chunk_overlap`，观察检索质量变化
- [ ] 尝试不同的相似度阈值，理解 `score_threshold` 参数

### 本阶段检验

能独立搭建一个"问你自己写的笔记"的 RAG 系统。

---

## 阶段六：RAG + Agent 融合

### TODO

- [ ] **6.1** 把你的 RAG 系统封装成一个 Tool
  ```python
  @tool
  def search_my_notes(query: str) -> str:
      """从我的笔记中检索相关信息"""
      # 调用你的 RAG 检索逻辑
      ...
  ```
- [ ] **6.2** 把这个 Tool 加入 Agent，让 Agent 能"自己决定"什么时候查笔记
- [ ] **6.3** 测试：让 Agent 回答一个只有你的笔记里才有答案的问题
- [ ] **6.4** 深入理解：Agent 是怎么决定要不要调用检索 Tool 的？

### 本阶段检验

构建一个能回答私有知识库问题的 RAG + Agent 系统，并理解它的每一行代码。

---

## 阶段七：扩展与实战

### TODO

- [ ] **7.1** 实现一个自定义 Tool（做一件实际有用的事，比如查天气、读文件）
- [ ] **7.2** 实现一个自定义 Agent（继承 `BaseSingleActionAgent` 或使用 LCEL）
- [ ] **7.3** 为 Agent 添加记忆（Memory），让它能记住对话历史
- [ ] **7.4** 阅读 LangGraph 的基本概念（LangChain 的下一代 Agent 框架）
- [ ] **7.5** 用 LangGraph 重写你的 RAG + Agent 系统

---

## 学习笔记区

> 在这里记录你的疑问、发现和"啊哈时刻"

### 疑问

- （在这里记录你读代码时不理解的地方）

### 关键发现

- （在这里记录你自己推断出来、后来被源码验证的结论）

### 踩过的坑

- （在这里记录错误和解决方案）

---

## 参考资源

- [LangChain 官方文档](https://docs.langchain.com/oss/python/langchain/overview)
- [LangChain API Reference](https://reference.langchain.com/python)
- [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph/overview)
- 本仓库源码：`libs/langchain_v1/` 是当前维护的主包入口
