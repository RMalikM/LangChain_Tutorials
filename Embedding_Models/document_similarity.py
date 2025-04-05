from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=256)

documents = [
    "Artificial intelligence is rapidly transforming various industries.",
    "The Amazon rainforest plays a crucial role in global climate regulation.",
    "Quantum computing has the potential to solve problems intractable for classical computers.",
    "Ancient Egyptian civilization left behind remarkable architectural achievements.",
    "The culinary traditions of Italy are diverse and regionally specific."
]

query = "What are the latest advancements in natural language processing?"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

similarity_scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Get the index of most relevant doc
index, score = sorted(list(enumerate(similarity_scores)), key=lambda x:x[1])[-1]

print("Query: ", query)
print("Most similary doc: ", documents[index])
print("Similarity Score is: ", score)