
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

# Load environment variables ()
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = 'RAG Chatbot v1'

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 1) Load PDF
PDF_PATH = "DIABETES.pdf"  
logging.info(f"PDF path: {PDF_PATH}")

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
if len(docs) > 1:
    logging.info(f"Sample page content:\n{docs[1].page_content[:20]}...")

# 2) Split into chunks
try:
    start_time = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    logging.info(f"Total number of chunks: {len(splits)}")
    logging.info(f"Sample chunk:\n{splits[1].page_content[:300]}...")
    end_time = time.perf_counter()
    logging.info(f"Time taken to split: {end_time - start_time:.4f} seconds")
except Exception as e:
    logging.error("Error during splitting: %s", str(e))
    raise

# 3) Embeddings + FAISS vectorstore
retriever = None
try:
    start_time = time.perf_counter()
    emb = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    db_path = "faiss_index"

    if os.path.exists(db_path):
        vs = FAISS.load_local(db_path, emb, allow_dangerous_deserialization=True)
        logging.info("Loaded existing FAISS index from disk.")
    else:
        vs = FAISS.from_documents(splits, emb)
        vs.save_local(db_path)
        logging.info(f"Saved FAISS index to '{db_path}'")

    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    end_time = time.perf_counter()
    logging.info(f"Time taken to setup vectorstore: {end_time - start_time:.4f} seconds")

except Exception as e:
    logging.error("Error in embedding or vectorstore: %s", str(e))
    raise

# Fail early if retriever wasn't initialized
if retriever is None:
    raise RuntimeError("Retriever was not initialized. Exiting.")

# 4) Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("user", "Question: {question}\n\nContext:\n{context}")
])

# 5) RAG Chain
llm = ChatGroq(model="groq/compound-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions in a loop
print("PDF RAG Chatbot ready. Ask your question (type 'exit' to quit).")

try:
    while True:
        q = input("\nQ: ").strip()
        if q.lower() in ["exit", "quit"]:
            print(" Goodbye!")
            break
        if not q:
            print(" Please enter a question.")
            continue

        ans = chain.invoke(q)
        print("\nA:", ans)

except KeyboardInterrupt:
    print("\n Interrupted by user. Exiting.")
