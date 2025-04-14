from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text = """
The story of human history begins approximately 300,000 years ago with the emergence of Homo sapiens in Africa.
Early humans lived as hunter-gatherers, relying on nature for food and shelter.

Around 10,000 BCE, the Agricultural Revolution marked a major turning point as humans began farming and domesticating animals.
Permanent settlements formed, leading to the rise of early civilizations such as Mesopotamia, Ancient Egypt, and the Indus Valley.

Writing systems emerged around 3,000 BCE, allowing the recording of laws, trade, and culture.
Classical civilizations like Greece, Rome, and China flourished, contributing to philosophy, science, and governance.
"""

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

# 1. Using split_text() 
# result = splitter.split_text(text)
# print(result)


# 2. Using split_documents() --- using text splitter and document loader together
loader = PyPDFLoader('data_files/Learning SQL-2-18.pdf')
docs = loader.load()

result1 = splitter.split_documents(docs)
print(result1[1].page_content)