from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(model="gpt-4", temperature=0.8, max_completion_tokens=500)

result = chat_model.invoke("write a 5 line poem on cricket")

print(result.content)