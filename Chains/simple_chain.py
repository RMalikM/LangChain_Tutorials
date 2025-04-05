from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template="Tell 5 not so common facts about {topic}",
    input_variables=['topic']
)

chat_model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt | chat_model | parser

# Visualize Chain
chain.get_graph().print_ascii()
result = chain.invoke({'topic': 'illuminati'})
print(result)