import os
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark

load_dotenv()

api_key = os.getenv("ARK_API_KEY") or os.getenv("API_KEY")
model_name = "doubao-embedding-vision-251215"

print("API_KEY exists:", bool(api_key))
print("MODEL:", model_name)

if not api_key:
    raise ValueError("请在 .env 中配置 ARK_API_KEY 或 API_KEY")

client = Ark(api_key=api_key)

try:
    print("----- text-only multimodal embeddings request -----")

    resp = client.multimodal_embeddings.create(
        model=model_name,
        input=[
            {
                "type": "text",
                "text": "产品经理为什么要关注交易模型？"
            }
        ]
    )

    print("\n调用成功")
    print(resp)

    data = getattr(resp, "data", None)
    if data is not None:
        embedding = getattr(data, "embedding", None)
        if embedding is not None:
            print("embedding dim:", len(embedding))

except Exception as e:
    print("\n调用失败")
    print("error_type:", type(e).__name__)
    print("error_message:", str(e))