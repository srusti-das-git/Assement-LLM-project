from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings

# 1. Initialize Gemini LLM and Embeddings
llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.6-flash",
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0.3
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=settings.GOOGLE_API_KEY
)

# 2. Initialize or load a local Chroma vector store with sample data
def get_vector_store():
    sample_docs = [
        "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints.",
        "LangChain is a framework designed to simplify the creation of applications using large language models.",
        "Google Gemini is a family of multimodal large language models developed by Google.",
        "Retrieval-Augmented Generation (RAG) is a technique for enhancing LLM responses with external data."
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(sample_docs)
    
    # Store locally in memory/disk
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore

# 3. Build RAG Chain
def ask_rag(question: str) -> str:
    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
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