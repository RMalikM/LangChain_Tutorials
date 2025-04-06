from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel

from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI()

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a LinkedIn post about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, chat_model, parser),
    'linkedin': RunnableSequence(prompt2, chat_model, parser)
})

result = parallel_chain.invoke({'topic': 'Quantum Computing'})

print("Tweet: ", result['tweet'])
print("LinkedIn: ", result['linkedin'])