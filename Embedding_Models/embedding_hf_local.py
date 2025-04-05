from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Single sentence query
result_single = embedding.embed_query("Delhi is the capital of India")
print("\nSingle query embedding: ",  result_single)

# Document embeddings
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]
result_doc = embedding.embed_documents(documents)
print("\nDocument Embeddings: ", result_doc)