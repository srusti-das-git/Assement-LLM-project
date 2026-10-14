from fastapi import FastAPI
from router import router as rag_router
app=FastAPI(title="RAG API with Google Gemini LLM")
app.include_router(rag_router, prefix="/api", tags=["RAG"])
@app.get("/")
def root():
    return {"message": "Welcome to the RAG API with Google Gemini LLM!"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)