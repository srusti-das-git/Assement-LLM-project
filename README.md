# RAG API with Google Gemini - Complete Guide

## What This Project Does

It's a web application that answers your questions using Google's AI (Gemini). You ask a question, it searches through some knowledge it has stored, finds the most relevant information, and gives you an answer based on that. This technique is called **RAG** (Retrieval-Augmented Generation).

---

## The 5 Files Explained

### 1. `requirement.txt` - The Shopping List
Lists all the libraries (tools) this project needs. Think of it as a shopping list for the kitchen.

| Library | What It Does (Simple) |
|---|---|
| **fastapi** | Builds the web server. It's like the restaurant's front-of-house that takes orders and serves food. |
| **uvicorn** | The actual server that runs and listens for incoming requests. It's the engine that keeps the restaurant open. |
| **langchain** | The main framework that connects AI models together. It's the "glue" that makes all AI pieces work together. |
| **langchain-google-genai** | The bridge that lets LangChain talk to Google's Gemini AI. It's the translator between our code and Google's servers. |
| **langchain-community** | Extra community-made tools for LangChain (like a toolbox with extra features). |
| **chromadb** | A local database that stores text and can find similar text quickly. It's like a smart library catalog. |
| **pydantic-settings** | Helps read settings (like the API key) from configuration files. |
| **python-dotenv** | Reads the `.env` file that contains secret settings (like your API key). |

---

### 2. `config.py` - The Key Holder
This file safely loads your **Google API key** from the `.env` file.

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Loads the .env file

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

settings = Settings()
```

**What happens:**
1. `load_dotenv()` reads the `.env` file
2. It looks for `GOOGLE_GEMINI_API_KEY` in that file
3. If not found, it falls back to `GOOGLE_API_KEY`
4. If neither exists, it uses an empty string `""`
5. The key is stored in `settings.GOOGLE_API_KEY`

**Why it matters:** Your API key is a secret password. It's stored in `.env` (not in the code) so you don't accidentally share it with others.

---

### 3. `services.py` - The Brain (RAG Logic)
This is where the AI magic happens. It contains three main parts:

#### Part A: Initialize the AI Models (lines 9-19)
```python
llm = ChatGoogleGenerativeAI(model="models/gemini-3.6-flash", ...)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", ...)
```

- **`llm`** = The AI that generates text answers (like ChatGPT)
- **`embeddings`** = The AI that converts text into numbers so the computer can understand similarity

**Analogy:** `llm` is the writer, `embeddings` is the librarian who organizes books.

#### Part B: Build the Knowledge Base (lines 22-35)
```python
def get_vector_store():
    sample_docs = [
        "FastAPI is a modern, fast web framework...",
        "LangChain is a framework for creating applications using LLMs...",
        "Google Gemini is a family of multimodal LLMs...",
        "RAG is a technique for enhancing LLM responses..."
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(sample_docs)
    
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore
```

**What happens step by step:**
1. We have 4 pieces of text (our "knowledge")
2. `RecursiveCharacterTextSplitter` breaks text into chunks (like cutting a long article into paragraphs). `chunk_size=500` means each chunk is up to 500 characters. `chunk_overlap=50` means adjacent chunks share 50 characters so we don't lose context at the boundaries.
3. `Chroma.from_documents()` takes those chunks, converts each to numbers (embeddings), and stores them in a local database
4. Returns the database so we can search it later

**Analogy:** This is like taking 4 textbooks, tearing them into manageable sections, and putting them in a smart filing cabinet that can find related topics.

#### Part C: The RAG Chain (lines 38-60)
```python
def ask_rag(question: str) -> str:
    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(question)
    return response
```

**What happens when someone asks a question:**

1. **Retrieve:** The question goes to the Chroma database. It finds the **2 most relevant chunks** (`k=2` means "give me the top 2 matches"). This is the "Retrieval" part of RAG.

2. **Format:** Those 2 chunks are joined together with newlines into one big text block called `context`.

3. **Prompt:** The `context` and the original `question` are combined into a template that says: "Answer the question based ONLY on this context."

4. **Generate:** The combined prompt goes to the Gemini AI (`llm`), which reads the context and generates an answer. This is the "Generation" part of RAG.

5. **Output:** The AI's response is converted from a complex object to a plain string by `StrOutputParser()`.

**Analogy:** You're a student. The question is on a test. You go to the library (Chroma), find 2 relevant pages (retrieval), read them, then write your answer based only on what you read (generation).

---

### 4. `router.py` - The Receptionist (API Endpoints)
This file defines what URLs (endpoints) exist and how they respond.

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import ask_rag

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@router.post("/query", response_model=QueryResponse)
def query_rag_endpoint(request: QueryRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
        answer = ask_rag(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**What this means:**

- **`APIRouter()`** = Creates a group of related routes (like a department in a restaurant)
- **`QueryRequest`** = Defines what a request should look like. It must have a `question` field that's a string. This is like an order form.
- **`QueryResponse`** = Defines what the response looks like. It returns an `answer` field that's a string. This is like the receipt.
- **`@router.post("/query")`** = Creates an endpoint at `/api/query` (the `/api` prefix comes from `main.py`) that only accepts POST requests (sending data, not just asking)
- **`HTTPException`** = Handles errors. If the question is empty, it returns error 400. If something else fails, it returns error 500.

**Analogy:** This is the menu. It tells customers: "If you send a POST request to `/api/query` with a `question`, you'll get back an `answer`."

---

### 5. `main.py` - The Restaurant Manager
This is the entry point that starts everything.

```python
from fastapi import FastAPI
from router import router as rag_router

app = FastAPI(title="RAG API with Google Gemini LLM")
app.include_router(rag_router, prefix="/api", tags=["RAG"])

@app.get("/")
def root():
    return {"message": "Welcome to the RAG API with Google Gemini LLM!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
```

**What each line does:**

1. **`FastAPI(title=...)`** = Creates the main application object (the restaurant)
2. **`include_router(rag_router, prefix="/api")`** = Adds all routes from `router.py` but makes them start with `/api`. So `/query` becomes `/api/query`
3. **`@app.get("/")`** = Creates a GET endpoint at the root URL `/` that just returns a welcome message
4. **`uvicorn.run("main:app", ...)`** = Starts the server:
   - `"main:app"` = Look in `main.py` file, find the `app` object
   - `host="127.0.0.1"` = Only accept connections from this computer (localhost)
   - `port=8000` = Listen on port 8000 (like a specific street address)
   - `reload=True` = If you change code, the server automatically restarts (like a self-driving car that re-parks when the map updates)

---

## How It All Works Together - Step by Step

When you run `python main.py` and someone asks a question:

```
1. YOU send: POST /api/query with {"question": "What is FastAPI?"}
       ↓
2. router.py receives the request
       ↓
3. It validates the request (must have a non-empty question)
       ↓
4. It calls ask_rag() from services.py
       ↓
5. services.py creates a new Chroma database with the 4 sample documents
       ↓
6. The question goes to Chroma, which finds the 2 most similar chunks
       ↓
7. Those chunks become the "context"
       ↓
8. The context + question are combined into a prompt
       ↓
9. The prompt is sent to Google's Gemini AI
       ↓
10. Gemini generates an answer based ONLY on the context
       ↓
11. The answer is converted to a plain string
       ↓
12. router.py wraps it in {"answer": "..."} and sends it back to you
```

---

## How to Run It

### Step 1: Install dependencies
```bash
pip install -r requirement.txt
```

### Step 2: Make sure `.env` has your API key
The `.env` file should look like:
```
GOOGLE_GEMINI_API_KEY=your_api_key_here
```

### Step 3: Start the server
```bash
python main.py
```

### Step 4: Test it

**Terminal A (server keeps running):**
The server will print:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

**Terminal B (send a request):**
```bash
# Test the welcome endpoint
curl http://127.0.0.1:8000/

# Test the RAG query
curl -X POST http://127.0.0.1:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is FastAPI?"}'
```

### Step 5: View documentation
Open in your browser: **http://127.0.0.1:8000/docs**

This gives you an interactive API documentation page where you can test endpoints directly.

---

## Common Issues and Fixes

| Problem | Cause | Solution |
|---|---|---|
| **404 Not Found on `/`** | Missing `@` decorator in `main.py:5` | Add `@` before `app.get("/")` |
| **404 models/embedding-001 not found** | Wrong embedding model name | Use `models/gemini-embedding-001` |
| **404 models/gemini-1.5-flash not found** | Model deprecated/renamed | Use `models/gemini-3.6-flash` |
| **"ALTS creds ignored" warnings** | Harmless Google Cloud message | Can be ignored - doesn't affect functionality |
| **"reload" warning** | uvicorn needs import string | Use `uvicorn.run("main:app", ...)` not `uvicorn.run(app, ...)` |

---

## Key Concepts Summary

- **API** = Application Programming Interface. A set of rules for programs to talk to each other.
- **REST API** = A type of API that uses HTTP methods (GET, POST) and URLs.
- **POST** = HTTP method for sending data to the server (like submitting a form).
- **GET** = HTTP method for retrieving data from the server (like reading a page).
- **JSON** = A lightweight data format that's easy for both humans and computers to read. Looks like `{"key": "value"}`.
- **Vector Database (Chroma)** = A database that stores text as numbers (vectors) so it can find similar items quickly using math.
- **Embeddings** = Converting words/sentences into lists of numbers that capture their meaning.
- **LLM** = Large Language Model. An AI trained on massive text data that can understand and generate human-like text.
- **RAG** = Retrieval-Augmented Generation. A technique where an AI first searches for relevant information, then generates an answer based on that information. This makes the AI more accurate and grounded in facts.
- **Temperature** = Controls how "creative" the AI is. Lower (0.3) = more focused and deterministic. Higher (1.0+) = more random and creative.
- **Chunking** = Breaking long text into smaller pieces so they can be stored and retrieved efficiently.