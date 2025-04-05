from langchain_openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct")  # Change the model to use other llm

result = llm.invoke("What is the capital of India?")

print(result)