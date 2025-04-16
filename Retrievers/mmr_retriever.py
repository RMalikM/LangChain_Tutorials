from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


documents = [
    Document(page_content="Mount Everest is the highest mountain peak in the world, located in the Himalayas on the border of Nepal and China."),
    Document(page_content="The Amazon Rainforest is known for its vast biodiversity and is often referred to as the lungs of the Earth."),
    Document(page_content="The Sahara Desert is the largest hot desert in the world, covering much of North Africa."),
    Document(page_content="The Great Barrier Reef, located off the coast of Australia, is the largest coral reef system in the world."),
    Document(page_content="Lake Baikal in Russia is the world's deepest freshwater lake and holds about 20%  of the Earth's unfrozen fresh water."),
    Document(page_content="The Nile River is the longest river in the world, flowing through northeastern Africa and emptying into the Mediterranean Sea."),
    Document(page_content="Antarctica is the coldest continent on Earth, covered almost entirely by ice and home to unique wildlife like penguins and seals."),
]

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

# Enable MMR in the retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)

query = "What makes Antarctica unique compared to other continents?"
results = retriever.invoke(query)
for i, doc in enumerate(results):
    print(f"\n*** Result {i+1} ***")
    print(doc.page_content)