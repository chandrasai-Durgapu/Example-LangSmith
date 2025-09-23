

import os
import time
import logging
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
# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = 'RAG Chatbot v2'

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PDF_PATH = "DIABETES.pdf"

# ---------------------------------------------
# Traced Steps
# ---------------------------------------------

@traceable(name="load_pdf")
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    logging.info(f"Loaded {len(docs)} pages from PDF.")
    if docs:
        logging.info(f"Sample content: {docs[0].page_content[:100]}...")
    return docs


@traceable(name="split_documents")
def split_documents(docs, chunk_size=500, chunk_overlap=30):
    start_time = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)
    logging.info(f"Split into {len(splits)} chunks.")
    end_time = time.perf_counter()
    logging.info(f"Splitting time: {end_time - start_time:.4f} seconds")
    return splits


@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    try:
        start_time = time.perf_counter()
        emb = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
        db_path = "faiss_index"

        if os.path.exists(db_path):
            vs = FAISS.load_local(db_path, emb, allow_dangerous_deserialization=True)
            logging.info("Loaded FAISS index from disk.")
        else:
            vs = FAISS.from_documents(splits, emb)
            vs.save_local(db_path)
            logging.info("Saved new FAISS index to disk.")

        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        end_time = time.perf_counter()
        logging.info(f"Vector store setup time: {end_time - start_time:.4f} seconds")
        return retriever

    except Exception as e:
        logging.error("Vector store error: %s", str(e))
        raise


@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    retriever = build_vectorstore(splits)
    return retriever


# ---------------------------------------------
# Main RAG Chain
# ---------------------------------------------

@traceable(name="run_chat_loop")
def run_chat_loop():
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

    print(" PDF RAG Chatbot ready. Ask your questions (type 'exit' to quit).")

    while True:
        try:
            q = input("\nQ: ").strip()
            if q.lower() in {"exit", "quit"}:
                print("Exiting.")
                break
            if not q:
                print("Please ask something.")
                continue

            # === Add run_name and optional tags/metadata here ===
            config = {
                "run_name": "pdf_rag_query",
                "tags": ["chat_loop", "pdf", "rag"],
                "metadata": {"input_length": len(q)}
            }

            answer = chain.invoke(q, config=config)
            print("\nA:", answer)

        except KeyboardInterrupt:
            print("\n Interrupted. Exiting.")
            break



# ---------------------------------------------
# Single Query Example (optional)
# ---------------------------------------------
# If you want to run a one-shot query instead of a loop, use this:

@traceable(name="run_single_query")
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


# Uncomment to test single query:
# print(run_single_query("What are the symptoms of diabetes?"))


if __name__ == "__main__":
    run_chat_loop()
