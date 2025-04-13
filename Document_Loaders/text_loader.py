from langchain_community.document_loaders import TextLoader

loader = TextLoader(file_path="data_files/text.txt", encoding='utf-8')

docs = loader.load()

print(type(docs))
print(docs)

# print("\n", docs[0].page_content)
# print("\n", docs[0].metadata)


# Using Text Loader with LLM
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI()

prompt = PromptTemplate(
    template='Write a summary for the following report- \n {report}',
    input_variables=['report']
)

parser = StrOutputParser()

chain = prompt | chat_model | parser

print("\n\nSummary: \n", chain.invoke({'report':docs[0].page_content}))
