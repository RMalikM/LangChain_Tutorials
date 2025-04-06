from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke - {joke}',
    input_variables=['joke']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, chat_model, parser, prompt2, chat_model, parser)

print(chain.invoke({'topic':'Jupitor'}))