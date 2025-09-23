from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize the Groq LLM
llm_model = ChatGroq(model_name="llama-3.3-70b-versatile",temperature=0.7)
# Simple one-line prompt
prompt = ChatPromptTemplate.from_messages([("user", "{question}")])

# model = ChatGroq()
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | llm_model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)
