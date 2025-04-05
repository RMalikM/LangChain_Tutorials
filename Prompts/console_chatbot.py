from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI()

while True:
    user_input = input('You: ')
    if user_input=='exit':
        break
    result = chat_model.invoke(user_input)
    print("AI: ", result.content)
