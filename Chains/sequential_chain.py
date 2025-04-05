from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Give a detailed report on the {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate the 5 point summary of the {report}",
    input_variables=['report']
)

chat_model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt1 | chat_model | parser| prompt2 | chat_model | parser

# Visualize Chain
chain.get_graph().print_ascii()

result = chain.invoke({'topic': 'Global Warming'})
print("\n", result)