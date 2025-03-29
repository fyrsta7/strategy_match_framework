import os
import numpy as np
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi

# 导入新版embeddings和llm模块
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


# 1. 准备示例代码文档集合（后续可扩展为大规模代码库）
documents = [
    Document(page_content="def foo():\n    print('Hello World')", metadata={"title": "foo"}),
    Document(page_content="def bar(x):\n    return x * 2", metadata={"title": "bar"}),
    Document(page_content="def baz(x, y):\n    return x + y", metadata={"title": "baz"}),
]

# 2. 构造BM25索引（此处采用简单分词方法）
corpus = [doc.page_content for doc in documents]
tokenized_corpus = [text.split() for text in corpus]
bm25 = BM25Okapi(tokenized_corpus)

def bm25_retrieve(query: str, top_k: int = 2):
    """
    使用BM25进行检索
    query: 用户查询代码或描述
    top_k: 返回最相似的文档数量
    """
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argsort(scores)[::-1]  # 按得分降序排序
    retrieved_docs = [documents[i] for i in ranked_indices[:top_k]]
    print("BM25 得分：", scores)
    for doc in retrieved_docs:
        print(f"检索到文档: {doc.metadata['title']}\n内容: {doc.page_content}\n")
    return retrieved_docs

# 3. 示例：检索最相似的代码片段
example_query = "def foo(): print('Hello World')"
retrieved_docs = bm25_retrieve(example_query, top_k=2)

# 4. 使用新版embeddings方法计算代码的embedding
# 这里将openai_api_key参数传入，避免环境变量缺失的问题
embeddings_model = OpenAIEmbeddings(openai_api_key=config.closeai_api_key, openai_api_base=config.closeai_base_url)
for doc in documents:
    embed_vector = embeddings_model.embed_query(doc.page_content)
    # 仅打印前5维，展示embedding结构
    print(f"文档 {doc.metadata['title']} 的embedding（前5维）： {embed_vector[:5]}")

# 5. RAG流程示例：利用LLM生成回答
# 使用config模块中的配置参数调用CloseAI的LLM接口

# 组合检索到的文档和查询构造生成提示（prompt）
retrieved_text = "\n---\n".join([doc.page_content for doc in retrieved_docs])
prompt = (
    f"以下是与查询代码最相似的代码片段：\n{retrieved_text}\n\n"
    f"请基于以上代码片段，为查询代码提供改进建议或补充说明。\n"
    f"查询代码为：\n{example_query}\n你的回答："
)
print("\n生成的提示（Prompt）：\n", prompt)

client = OpenAI(
    base_url = config.closeai_base_url,
    api_key = config.closeai_api_key,
)

messages = [
    {
        "role": "user",
        "content": prompt
    }
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
