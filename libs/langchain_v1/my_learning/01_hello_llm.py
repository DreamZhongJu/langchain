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
