# Example-LangSmith

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Example-LangSmith demonstrates how to integrate the Groq LLM with LangChain and LangSmith to build and monitor various LLM workflows. The repository includes examples of:

- Simple LLM calls
- Sequential chains
- Retrieval-Augmented Generation (RAG) workflows
- Agent workflows
- LangGraph orchestration pipelines

LangSmith is used throughout to trace, monitor, and debug LLM and chain executions.

---

## Repository Structure

Example-LangSmith/
├── 1_simple_llm_call.py # Simple Groq LLM prompt example
├── 2_sequential_chain.py # Sequential chaining of prompts
├── 3_rag_v1.py # Basic RAG example with PDF retrieval
├── 3_rag_v2.py # RAG with improved indexing
├── 3_rag_v3.py # Enhanced RAG version
├── 3_rag_v4.py # Advanced RAG with custom retriever
├── 4_agent.py # Agent with tool usage example
├── 5_langgraph.py # LangGraph graph orchestration example
├── DIABETES.pdf # Sample document for RAG demos
├── requirements.txt # Python dependencies
└── LICENSE # MIT License


---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/chandrasai-Durgapu/Example-LangSmith.git
cd Example-LangSmith

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

 
Other features of Langsmith --- Monitoring and Alerting
