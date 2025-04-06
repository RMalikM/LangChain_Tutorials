from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI()

prompt1 = PromptTemplate(
    template="Write a report about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Genrate a short summary of the following - {report}',
    input_variables=['report']
)

parser = StrOutputParser()

report_chain = RunnableSequence(prompt1, chat_model, parser)

parallel_chain = RunnableParallel({
    'repot': RunnablePassthrough(),
    'summary': RunnableSequence(prompt2, chat_model, parser)
})

final_chain = RunnableSequence(report_chain, parallel_chain)
print(final_chain.invoke({'topic':'Jupitor'}))

