# Example-LangSmith

A sample project demonstrating how to use [LangSmith](https://smith.langchain.com/) for debugging, evaluating, and monitoring LangChain applications.

---

## 📌 Overview

This repository provides a practical example of integrating **LangSmith** with **LangChain** projects. LangSmith is a platform for:
- **Debugging** LLM chains and agents
- **Evaluating** model performance
- **Monitoring** production applications

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- A [LangSmith account](https://smith.langchain.com/) (sign up for free)
- Basic familiarity with LangChain

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chandrasai-Durgapu/Example-LangSmith.git
   cd Example-LangSmith


Install dependencies:
 Copypip install -r requirements.txt


Set your LangSmith API key:
 Copyexport LANGCHAIN_API_KEY="your-api-key-here"
Or set it in a .env file:
 CopyLANGCHAIN_API_KEY=your-api-key-here



📂 Project Structure
FileDescriptionexample.pyExample LangChain + LangSmith scriptrequirements.txtPython dependencies.env.exampleExample environment variables

🔧 Usage
Running the Example


Run the example script:
 Copypython example.py


View your traces and evaluations:

Visit the LangSmith dashboard
All runs, traces, and evaluations will appear under your project



Customizing the Example

Modify example.py to include your own LangChain components (chains, agents, prompts).
Add custom evaluators or datasets in LangSmith for more advanced use cases.

