import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader


    

def get_chroma_vectorstore(persist_dir: str,embedding_function,pdf_path: str) -> Chroma:

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    """
    Returns a Chroma vector store, loading from disk if it exists,
    otherwise loading PDFs, splitting them, embedding, persisting.
    """
    # If index already persisted, just load it
    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_function,
        )

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

 

    vectordb = Chroma.from_documents(
        split_docs,
        embedding=embedding_function,
        persist_directory=persist_dir,
    )
    return vectordb