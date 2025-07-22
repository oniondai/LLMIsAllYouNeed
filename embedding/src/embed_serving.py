from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Query, Body
from typing import Dict, List
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

model = '/opt/meituan/dolphinfs_daicong/2.llm/huggingface.co/Qwen/Qwen3-Embedding-8B'
model = SentenceTransformer(model)
print(model.encode(['你好']))

class QueryRequest(BaseModel):
    queries: List[str]

app = FastAPI()

@app.post("/llm_embedding")
async def build_llm_embed_result(request: QueryRequest):
    print(request.queries)
    query_embeddings = model.encode(request.queries)

    return query_embeddings.tolist()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)


