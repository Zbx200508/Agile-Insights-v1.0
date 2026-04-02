import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("API_KEY")
api_base = os.getenv("API_BASE")
model_name = os.getenv("MODEL_NAME")

print("API_BASE:", api_base)
print("MODEL_NAME:", model_name)
print("API_KEY exists:", bool(api_key))

if not api_key or not api_base or not model_name:
    raise ValueError("请检查 .env，API_KEY / API_BASE / MODEL_NAME 不能为空")

client = OpenAI(
    api_key=api_key,
    base_url=api_base,
)

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "你是一个简洁的助手。"},
        {"role": "user", "content": "请只回复：API测试成功"}
    ],
    temperature=0.1,
)

print("\n模型返回结果：")
print(response.choices[0].message.content)