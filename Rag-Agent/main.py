from fastapi import FastAPI
from pydantic import BaseModel
from config import PERSIST_DIR,PDF_PATH
from vectorstore.chroma_store import get_chroma_vectorstore
from models.llm_provider import get_openai_llm
from chains.retrieval_chain import build_retrieval_chain
import glob
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

app = FastAPI()

# Load static options
FRAMEWORKS = ['LangChain', 'CrewAI', 'LlamaIndex']
MODELS = ['OpenAI', 'Anthropic', 'Cohere']
VECTORSTORES = ['Chroma', 'Weaviate', 'Pinecone']



embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Load & split


vectordb   = get_chroma_vectorstore(PERSIST_DIR, embeddings, PDF_PATH)
retriever = vectordb.as_retriever()
llm = get_openai_llm()
rag_chain = build_retrieval_chain(llm, retriever)

# Request schema
class RAGQuery(BaseModel):
    framework: str
    model: str
    vector_store: str
    prompt: str

@app.post("/api/rag-query")
def rag_query(request: RAGQuery):
    print(f"Received request with framework={request.framework}, model={request.model}, vector_store={request.vector_store}")
    result = rag_chain.invoke({"input": request.prompt})
    return {"answer": result["answer"]}

# Optional health check
@app.get("/")
def health_check():
    return {"message": "FastAPI RAG API running"}