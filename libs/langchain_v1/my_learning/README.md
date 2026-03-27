# LangChain 个人学习笔记

数据集：QQ空间说说备份（`data/说说备份_792059284.txt`，593 条，2011–2026 年）

---

## 03_basic_rag.py — RAG 流程实验

### 流程概览

```
txt 文件
  → load_shuoshuo()          按 [YYYY年MM月DD日] 解析，得到 547 条 dict
  → split_into_chunks()      RecursiveCharacterTextSplitter 切分成 Document
  → build_vectorstore()      OpenAIEmbeddings (ModelScope) + FAISS 向量化存储
  → similarity_search / MMR  检索
```

---

### 5.3 源码阅读笔记

#### RecursiveCharacterTextSplitter 切分规则

源码位置：`libs/text-splitters/langchain_text_splitters/character.py:107`

核心是两步：

**第一步 — 贪心选分隔符**：遍历 `separators` 列表，找第一个能在文本中匹配到的，
用它切一刀；切完仍然太大的片段，换列表里的下一个分隔符递归处理。

```python
# 默认 separators（中文场景建议替换）
["\n\n", "\n", " ", ""]

# 本项目使用
["。", "！", "？", "…", "\n", ""]
```

**第二步 — 滑动窗口合并**（`base.py:152` `_merge_splits`）：
把切好的小片段逐个装入窗口，超过 `chunk_size` 就输出；
输出后从窗口头部 pop，直到窗口长度 ≤ `chunk_overlap`，
剩余内容作为下一个 chunk 的开头，这就是"重叠"的来源。

#### 检索流程

```
vectorstore.as_retriever()
  → 创建 VectorStoreRetriever(search_type="similarity")

retriever.invoke(query)          # retrievers.py:179
  → 配置回调（LangSmith 追踪）
  → _get_relevant_documents()   # vectorstores/base.py:1040
      ├─ similarity       → similarity_search()
      ├─ mmr              → max_marginal_relevance_search()
      └─ score_threshold  → similarity_search_with_relevance_scores()
  → 触发 on_retriever_end 回调
  → 返回 list[Document]
```

`vectorstore.similarity_search()` 是绕过 Retriever 层的直接调用，
接 `create_retrieval_chain()` 时必须用 `as_retriever()` 形式。

#### similarity_search vs MMR

| | similarity_search | MMR |
|---|---|---|
| 选择标准 | 只看与 query 的余弦距离 | `λ × 相关性 − (1−λ) × 冗余惩罚` |
| 结果特点 | 最相关，但可能重复 | 相关且多样 |
| 关键参数 | `k` | `k`, `fetch_k`, `lambda_mult` |

MMR 算法（`vectorstores/utils.py:149`）：
先捞 `fetch_k` 个候选，然后贪心地逐个挑选——每轮把"与已选文档最大相似度"
作为惩罚项，`lambda_mult=0` 最多样，`lambda_mult=1` 退化为纯相似度排序。

---

### chunk_size / chunk_overlap 实验结论

测试数据：547 条说说，平均原始长度约 **124 字**。

| chunk_size | chunk_overlap | chunk 数 | 平均长度 |
|------------|--------------|---------|---------|
| 100 | 0 | 1184 | 56 字 |
| 100 | 30 | 1227 | 60 字 |
| 300 | 50 | 632 | 109 字 |
| **500** | **50** | **562** | **121 字** |
| 500 | 150 | 562 | 122 字 |
| 1000 | 100 | 548 | 124 字 |

**结论：**

1. **`chunk_size` 要小于原文平均长度才有意义**。本数据集说说平均 124 字，
   `chunk_size=500` 相当于几乎不切，从 500 增大到 1000 chunk 数几乎不变（562→548）。
   想真正影响粒度，需要调到 100 左右。

2. **`chunk_overlap` 在短文本上效果有限**。原文本本身就短，相邻 chunk 本来就少，
   overlap 从 50 改到 150 时 chunk 数完全没变。overlap 在长文档（论文、手册）中才更有价值。

3. **chunk 太大会导致检索返回重复内容**。`chunk_size=500` 时 similarity_search
   会返回同一条说说的多个相邻 chunk，造成结果列表中出现重复。
   可用 MMR 或缩小 `chunk_size` 缓解。

4. **MMR 更适合这份数据**。说说情感同质化强（同一时期写的内容高度相似），
   MMR 能跨年份召回多样结果，避免返回一堆同时期的雷同内容。
