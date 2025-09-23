

import os
import time
import json
import hashlib
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

# -------------------- Setup --------------------
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = 'RAG Chatbot v4-meta tags'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PDF_PATH = "DIABETES.pdf"
INDEX_ROOT = Path(".indices")

if not os.path.exists(INDEX_ROOT):
    INDEX_ROOT.mkdir(exist_ok=True)
    logging.info(f"📁 Created index root directory at: {INDEX_ROOT.resolve()}")
else:
    logging.info(f" Index root directory already exists at: {INDEX_ROOT.resolve()}")

# -------------------- Fingerprinting Logic --------------------
def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {
        "sha256": h.hexdigest(),
        "size": p.stat().st_size,
        "mtime": int(p.stat().st_mtime)
    }

def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model: str) -> str:
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model,
        "version": "v1"
    }
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    return key

# -------------------- PDF Pipeline --------------------
@traceable(name="load_pdf", tags=['pdf', 'loader'], metadata={'loader': 'PyPDFLoader'})
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    logging.info(f"Loaded {len(docs)} pages from PDF.")
    return docs

@traceable(name="split_documents", tags=['splitter', 'splits'])
def split_documents(docs, chunk_size=500, chunk_overlap=30):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)
    logging.info(f"Split into {len(splits)} chunks.")
    return splits

@traceable(name="build_vectorstore", tags=['embedding', 'faiss'], metadata={"embedding_model": "all-MiniLM-L6-v2"})
def build_vectorstore(splits, embed_model: str, index_dir: Path):
    embeddings = HuggingFaceEmbeddings(model=embed_model)
    vs = FAISS.from_documents(splits, embeddings)
    vs.save_local(str(index_dir))
    logging.info(f"Saved FAISS index to: {index_dir}")
    return vs

@traceable(name="load_vectorstore", tags=['embedding', 'faiss_load'])
def load_vectorstore(index_dir: Path, embed_model: str):
    embeddings = HuggingFaceEmbeddings(model=embed_model)
    vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    logging.info(f"Loaded FAISS index from: {index_dir}")
    return vs

# -------------------- Load or Build Index --------------------
@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(pdf_path: str, chunk_size=500, chunk_overlap=30, embed_model="all-MiniLM-L6-v2", force_rebuild=False):
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embed_model)
    index_dir = INDEX_ROOT / key
    if index_dir.exists() and not force_rebuild:
        vs = load_vectorstore(index_dir, embed_model)
    else:
        docs = load_pdf(pdf_path)
        splits = split_documents(docs, chunk_size, chunk_overlap)
        vs = build_vectorstore(splits, embed_model, index_dir)
        with open(index_dir / "meta.json", "w") as f:
            json.dump({
                "pdf_path": os.path.abspath(pdf_path),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_model": embed_model
            }, f, indent=2)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever

# -------------------- Main RAG Chain --------------------
@traceable(name="run_chat_loop", tags=["interface", "chat"])
def run_chat_loop():
    retriever = setup_pipeline(PDF_PATH)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a document QA assistant.
        Only answer using the information provided in the context below.
        If the answer is not explicitly present in the context, reply with:
        "I don't know based on the context provided."

        Do NOT use prior knowledge.
        Do NOT guess.
        Do NOT fabricate.
        Respond only based on the given context."""),
        ("user", "Question: {question}\n\nContext:\n{context}")
    ])

    llm = ChatGroq(model="groq/compound-mini", temperature=0.4)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }) | prompt | llm | StrOutputParser()

    print("PDF RAG Chatbot ready. Ask your questions (type 'exit' to quit).")

    while True:
        try:
            q = input("\nQ: ").strip()
            if q.lower() in {"exit", "quit"}:
                print("Exiting.")
                break
            if not q:
                print("Please ask something.")
                continue

            config = {
                "run_name": "pdf_rag_query",
                "tags": ["chat", "pdf", "rag"],
                "metadata": {"input_length": len(q)}
            }

            answer = chain.invoke(q, config=config)
            print("\nA:", answer)

        except KeyboardInterrupt:
            print("\n Interrupted. Exiting.")
            break

# -------------------- Optional: Single Query --------------------
@traceable(name="run_single_query", tags=["query", "single"])
def run_single_query(question: str):
    retriever = setup_pipeline(PDF_PATH)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
        ("user", "Question: {question}\n\nContext:\n{context}")
    ])

    llm = ChatGroq(model="groq/compound-mini", temperature=0)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }) | prompt | llm | StrOutputParser()

    config = {
        "run_name": "pdf_rag_single_query",
        "tags": ["single_query", "pdf", "rag"],
        "metadata": {"input_length": len(question)}
    }

    answer = chain.invoke(question, config=config)
    return answer

# -------------------- CLI Entry --------------------
if __name__ == "__main__":
    run_chat_loop()
