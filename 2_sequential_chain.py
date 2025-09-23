from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# set project name in langsmith
os.environ["LANGCHAIN_PROJECT"]='Sequential LLM App'


prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are an expert researcher. Your task is to generate a detailed report."),
    ("user", "Generate a detailed report on {topic}")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are an expert summarizer. Your task is to provide a 5-point summary."),
    ("user", "Generate a 5-point summary from the following text:\n\n{text}")
])


model1 = ChatGroq(model_name="groq/compound-mini",temperature=0.7)
model2 = ChatGroq(model_name="llama-3.3-70b-versatile",temperature=0.5)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config={
    'tags':['llm app','report generation', 'summarization'], 
    'metadata':{'model1':'groq/compound-mini', 'model1_temp':0.7,'parser':'stroutputparser'}
        }

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)
