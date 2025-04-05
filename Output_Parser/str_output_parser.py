from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI()

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=['text']
)

# Example: No Str Parser
prompt1 = template1.invoke({'topic': "Black Hole"})
result1 = chat_model.invoke(prompt1)
prompt2 = template2.invoke({'text': result1.content})
result2 = chat_model.invoke(prompt2)
print("\nOutput No Str Parser: ", result2.content)

# Using StrOutputParser
chat_parser = StrOutputParser()
chain = template1 | chat_model | chat_parser | template2 | chat_model | chat_parser
result = chain.invoke({'topic': 'Black Hole'})
print("\nOutput with StrOutputParser: ", result)