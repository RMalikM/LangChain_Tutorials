from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(page_content="Mount Everest is the highest mountain peak in the world, located in the Himalayas on the border of Nepal and China."),
    Document(page_content="The Amazon Rainforest is known for its vast biodiversity and is often referred to as the lungs of the Earth."),
    Document(page_content="The Sahara Desert is the largest hot desert in the world, covering much of North Africa."),
    Document(page_content="The Great Barrier Reef, located off the coast of Australia, is the largest coral reef system in the world."),
    Document(page_content="Lake Baikal in Russia is the world's deepest freshwater lake and holds about 20%  of the Earth's unfrozen fresh water.")
]

embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="world_geography"
)

query = "Tell me about the largest desert in Africa"

# Using Vector Store as Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
results = retriever.invoke(query)
for i, doc in enumerate(results):
    print(f"\n*** Result {i+1} ***")
    print(doc.page_content)

# Using Similarity search
results1 = vectorstore.similarity_search(query, k=2)
for i, doc in enumerate(results1):
    print(f"\n*** Result {i+1} ***")
    print(doc.page_content)