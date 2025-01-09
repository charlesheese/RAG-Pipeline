import os 
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import openai
from llama_index.core import Document
import numpy as np
import textwrap
import re

# load env variables 
load_dotenv()

# api key setup
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

documents=SimpleDirectoryReader("./data").load_data()

question = "who is the 10-k you have from"
query = f"You are a LLM tasked with analyzing and finding information from 10-k SEC filings. The context is the data you have recieved. The question you are answering is {question}. If you are unsure about the question you are answering or do not have enough context say so, do not makeup answers."
index=VectorStoreIndex.from_documents(documents, show_progress=True)

query_engine=index.as_query_engine(similarity_top_k=3)

response=query_engine.query(query)
print(response)






