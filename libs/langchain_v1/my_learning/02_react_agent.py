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

# 新 API：create_agent 基于 LangGraph，不再需要 AgentExecutor 和 prompt
agent = create_agent(llm, tools, debug=True)

# 测试 1：需要调用 Tool
print("=== 测试 1 ===")
result = agent.invoke({"messages": [{"role": "user", "content": "现在几点？"}]})
print("最终答案：", result["messages"][-1].content)
for msg in result["messages"]:
    print(type(msg).__name__, ":", msg.content[:80] if msg.content else "[no content]")

# 测试 2：需要计算
print("\n=== 测试 2 ===")
result = agent.invoke({"messages": [{"role": "user", "content": "100 乘以 99 等于多少？"}]})
print("最终答案：", result["messages"][-1].content)
for msg in result["messages"]:
    print(type(msg).__name__, ":", msg.content[:80] if msg.content else "[no content]")

# 测试 3：不需要 Tool
print("\n=== 测试 3 ===")
result = agent.invoke({"messages": [{"role": "user", "content": "天空为什么是蓝色的？"}]})
print("最终答案：", result["messages"][-1].content)
for msg in result["messages"]:
    print(type(msg).__name__, ":", msg.content[:80] if msg.content else "[no content]")
