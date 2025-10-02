# Example-LangSmith


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview



---
# 🧪 Example LangSmith Integration

This project demonstrates how to integrate **LangSmith** into a LangChain-based application to enable **LLM tracing**, **debugging**, and **example management**. It showcases how to monitor, analyze, and evaluate large language model workflows using LangSmith's observability tools.

- Simple LLM calls  
- Sequential chains  
- Retrieval-Augmented Generation (RAG)  
- Agent workflows  
- LangGraph orchestration
- Langsmith --- Monitoring and Alerting
[View LangSmith Projects](https://smith.langchain.com/o/b19e13a8-e01b-4367-8e67-7c4ae54a2821/projects)

All examples are instrumented with LangSmith for tracing and monitoring.
---

## 🚀 Features

- 🔍 **LLM Tracing**: Automatically track LLM calls, inputs, outputs, and latency using LangSmith.
- 🧪 **Evaluation**: Compare model responses to labeled examples for quality testing.
- 🧠 **Few-Shot Example Selection**: Dynamically fetch relevant examples from LangSmith datasets.
- 🛠️ **Tool Usage Logging**: See how tools (e.g., search, calculator) are used inside chains/agents.
- 📈 **Production Monitoring**: Improve and debug your LangChain agents more effectively.

---

## 🧰 Prerequisites

Before running the project, ensure you have the following:

- ✅ Python 3.8 or higher
- ✅ [LangSmith account](https://www.langchain.com/langsmith)
- ✅ LangSmith API key
- ✅ (Optional) OpenAI API key or any other LLM provider credentials

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/chandrasai-Durgapu/Example-LangSmith.git
cd Example-LangSmith


---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/chandrasai-Durgapu/Example-LangSmith.git
cd Example-LangSmith
```
---
## Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---
## Install dependencies
```bash
pip install -r requirements.txt
```
---
Environment Setup

Create a .env file using the provided example:
```bash
cp .env.example .env
```
Then add your API keys and config to .env:
```bash
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=LangSmith-Demo
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.langchain.plus

```
---
# run seperately every python file
```bash
python 1_simple_llm_call.py
```
```bash
python 2_sequential_chain.py
```
```bash
python 3_rag_v1.py
```
```bash
python 3_rag_v2.py
```
```bash
python 3_rag_v3.py
```
```bash
python 3_rag_v4.py
```
```bash
python 4_agent.py
```
```bash
python 5_langgraph.py
```
---
## How It Works

The LangChain chain or agent is wrapped using LangSmith’s traceable decorators or environment-based tracing.

As the chain runs, each step is automatically logged:

LLM calls (prompts + responses)

Tool usage

Intermediate inputs/outputs

Optionally, examples can be retrieved from LangSmith datasets and injected into prompts (for few-shot learning).
 
Other features of Langsmith --- Monitoring and Alerting
[View LangSmith Projects](https://smith.langchain.com/o/b19e13a8-e01b-4367-8e67-7c4ae54a2821/projects)

