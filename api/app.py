from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query import embed_text, query_index, generate_rag_answer

app = FastAPI(title="Multi-Modal RAG API")

# CORS (allow your UI to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(req: QueryRequest):
    try:
        query_vec = embed_text(req.question)
        retrieved = query_index(query_vec)
        answer = generate_rag_answer(retrieved, req.question)
        return {"answer": answer, "sources": retrieved}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
