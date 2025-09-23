# Example-LangSmith
# Example-LangSmith

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Example-LangSmith demonstrates how to integrate Groq LLMs with LangChain and LangSmith to build various language model workflows, including:

- Simple LLM calls  
- Sequential chains  
- Retrieval-Augmented Generation (RAG)  
- Agent workflows  
- LangGraph orchestration  

All examples are instrumented with LangSmith for tracing and monitoring.

---


---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/chandrasai-Durgapu/Example-LangSmith.git
cd Example-LangSmith



---
conda create -n example-langsmith python=3.13 -y
conda activate example-langsmith

pip install -r requirements.txt

python 1_simple_llm_call.py

python 2_sequential_chain.py


python 3_rag_v1.py
python 3_rag_v2.py
python 3_rag_v3.py
python 3_rag_v4.py

python 4_agent.py

python 5_langgraph.py

export LANGCHAIN_API_KEY="your_langsmith_api_key"
export LANGCHAIN_PROJECT="example-langsmith"
Or use a .env file with python-dotenv.

 
Other features of Langsmith --- Monitoring and Alerting

