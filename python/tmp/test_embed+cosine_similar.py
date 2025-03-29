"""
示意性代码，展示如何利用 LangChain 框架中的新版模块实现：
   1. 对所有示例文档先计算 embeddings 并生成向量库
   2. 后续拿到查询文本后，计算查询的 embeddings，并在向量层面计算相似度（采用余弦相似度）
   3. 找出最相似的几个文档，并拼接进 prompt 送给 LLM 生成回答

注意：
   需要先安装依赖库：
       pip install -U langchain-openai langchain-community numpy openai
       
   同时需要准备 config.py 文件，并在其中定义以下变量：
       closeai_base_url      # 例如："https://api.openai.com/v1"
       closeai_api_key       # 替换为你的 API key
       closeai_chatgpt_model # 例如："gpt-3.5-turbo"

另外，BM25 是基于词频和逆文档频率的检索算法，不适用于密集向量间的相似度计算，推荐使用余弦相似度等方法进行向量检索。
"""

import os
import sys
import numpy as np
from langchain.docstore.document import Document

# 导入新版 embeddings 模块
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# 添加项目根目录到 sys.path 中，以便载入 config.py 文件
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 1. 准备示例代码文档集合（后续可扩展为大规模代码库）
documents = [
    Document(page_content="def foo():\n    print('Hello World')", metadata={"title": "foo"}),
    Document(page_content="def bar(x):\n    return x * 2", metadata={"title": "bar"}),
    Document(page_content="def baz(x, y):\n    return x + y", metadata={"title": "baz"}),
]

# 2. 使用新版 embeddings 方法计算所有文档的向量，并构建向量库
embeddings_model = OpenAIEmbeddings(openai_api_key=config.closeai_api_key, openai_api_base=config.closeai_base_url)

# 生成文档 embeddings，并存入内存（向量库）
doc_embeddings = []
for doc in documents:
    embed_vector = embeddings_model.embed_query(doc.page_content)
    doc_embeddings.append(embed_vector)
    # print(f"文档 {doc.metadata['title']} 的 embedding（前5维）： {embed_vector[:5]}")

# 3. 定义函数，通过向量检索找到与查询最相似的文档
def cosine_similarity(vec_a, vec_b):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def embedding_retrieve(query: str, top_k: int = 2):
    """
    对查询文本计算 embedding 后，在向量库中通过余弦相似度找到最相似的 top_k 文档
    """
    query_embedding = embeddings_model.embed_query(query)
    similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    similarities = np.array(similarities)
    ranked_indices = np.argsort(similarities)[::-1]  # 从最高相似度到最低排序
    print("余弦相似度：", similarities)
    retrieved_docs = [documents[i] for i in ranked_indices[:top_k]]
    for doc in retrieved_docs:
        print(f"检索到文档: {doc.metadata['title']}\n内容: {doc.page_content}\n")
    return retrieved_docs

# 4. 示例：对一个查询文本进行 embeddings 检索
example_query = "def foo(): print('Hello World')"
retrieved_docs = embedding_retrieve(example_query, top_k=2)

# 5. 使用 OpenAI ChatCompletion API 生成回答
# 组合检索到的文档和查询构造生成提示（prompt）
retrieved_text = "\n---\n".join([doc.page_content for doc in retrieved_docs])
prompt_content = (
    f"以下是与查询代码最相似的代码片段：\n{retrieved_text}\n\n"
    f"请基于以上代码片段，为查询代码提供改进建议或补充说明。\n"
    f"查询代码为：\n{example_query}\n你的回答："
)
print("\n生成的提示（Prompt）：\n", prompt_content)

client = OpenAI(
    base_url = config.closeai_base_url,
    api_key = config.closeai_api_key,
)

messages = [
    {"role": "user", "content": prompt_content}
]

response = client.chat.completions.create(
    model = config.closeai_chatgpt_model,
    messages = messages
)

llm_response = response.choices[0].message.content
print("\nLLM生成的回答：\n", llm_response)

if __name__ == "__main__":
    # 示例代码执行完毕
    pass