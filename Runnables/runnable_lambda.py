from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

from dotenv import load_dotenv
load_dotenv()

def word_count(text):
    return len(text.split())

chat_model = ChatOpenAI()

prompt = PromptTemplate(
    template="Write a report about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

report_chain = RunnableSequence(prompt, chat_model, parser)

parallel_chain = RunnableParallel({
    'repot': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(report_chain, parallel_chain)
print(final_chain.invoke({'topic':'Jupitor'}))

