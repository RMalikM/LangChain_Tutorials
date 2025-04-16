from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Recreate the document objects from the previous data
documents = [
    Document(page_content=(
        """Mount Kilimanjaro is the highest peak in Africa.
        It is a dormant volcano with three cones: Kibo, Mawenzi, and Shira.
        Many climbers attempt to reach its summit each year. Snow can be found at its top despite its location near the equator."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """The Great Wall of China was built over several dynasties to protect against invasions.
        It stretches over 13,000 miles. 
        Ancient bricks and stones make up the structure. It is visible from space under certain conditions."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Leonardo da Vinci was a Renaissance artist and inventor.
        His famous works include the Mona Lisa and The Last Supper.
        He also sketched designs for flying machines and war devices.
        Da Vinci’s notebooks reveal a brilliant and curious mind."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """Volcanoes are openings in the Earth's surface through which magma erupts.
        Active volcanoes can be found in the Pacific Ring of Fire.
        Ash clouds from eruptions can affect air travel. Lava cools and forms new land over time."""
    ), metadata={"source": "Doc4"})
]


# Create a FAISS vector store from the documents
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embedding_model)

# Create a base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Set up the compressor using an LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")
compressor = LLMChainExtractor.from_llm(llm)

# Create the contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# Query the retriever
query = "What happens when a volcano erupts?"
compressed_results = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_results):
    print(f"\n*** Result {i+1} ***")
    print(doc.page_content)