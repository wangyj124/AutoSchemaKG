from openai import OpenAI

# ========== 1. 构造符合你数据集的问题（替换成plantstation_100相关问题） ==========
message = [
    {
        'role':'system',
        'content':'你是熟知燃气轮机电厂测点知识的智能问答助手，仅回答燃气轮机测点相关的问题，基于知识图谱给出精准答案。'
    },
    {
        'role':'user', 
        'content':"""给出位于CHANGXING_1 电厂的三个测点的名称"""
    }
]

# ========== 2. 初始化客户端（适配你的10089端口服务） ==========
# 关键修改：
# - base_url：替换为你运行服务的机器IP（0.0.0.0是监听地址，调用时需填实际IP，比如10.67.74.45）
# - api_key：替换为你实际的Qwen API Key（不是EMPTY）
plant_base_url = "http://10.8.2.63:10089/v1/"  # 替换为服务端实际IP（如10.67.74.45）
plant_client = OpenAI(
    api_key="sk-6aP9xT2mQv1sRb8Zc4NyUe0Lf7Hk3Wd5",  # 你的Qwen API Key
    base_url=plant_base_url
)

# ========== 3. 调用你的RAG服务（核心：模型名替换为Qwen） ==========
# 关键修改：model从"llama"改为你部署的Qwen模型名
response = plant_client.chat.completions.create(
    model="qwen3-30B-A3B-Instruct",  # 必须和你服务端的模型名一致
    messages=message,
    max_tokens=2048,
    temperature=0.5,
)

# ========== 4. 打印回答结果 ==========
print("RAG服务回答：\n", response.choices[0].message.content)