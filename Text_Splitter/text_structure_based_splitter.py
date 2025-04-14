from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text = """
The story of human history begins approximately 300,000 years ago with the emergence of Homo sapiens in Africa.
Early humans lived as hunter-gatherers, relying on nature for food and shelter.

Around 10,000 BCE, the Agricultural Revolution marked a major turning point as humans began farming and domesticating animals.
Permanent settlements formed, leading to the rise of early civilizations such as Mesopotamia, Ancient Egypt, and the Indus Valley.

Writing systems emerged around 3,000 BCE, allowing the recording of laws, trade, and culture.
Classical civilizations like Greece, Rome, and China flourished, contributing to philosophy, science, and governance.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0
)

chunks = splitter.split_text(text)
print(len(chunks))

print(chunks)
