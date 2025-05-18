from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

def create_docs():
    doc1 = Document(
        page_content="Albert Einstein developed the theory of relativity, fundamentally changing our understanding of space, time, and gravity. " \
                "His equation E=mc² is one of the most famous in physics.",
        metadata={"field": "Theoretical Physics"}
    )
    doc2 = Document(
            page_content="Marie Curie was a pioneering physicist and chemist who discovered radioactivity. She was the first woman to win a Nobel Prize "
                    "and remains the only person to win in two scientific fields.",
            metadata={"field": "Physics and Chemistry"}
        )
    doc3 = Document(
            page_content="Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics. " \
                    "His work remains a cornerstone of physics and mathematics.",
            metadata={"field": "Classical Mechanics"}
        )
    doc4 = Document(
            page_content="Charles Darwin introduced the theory of evolution by natural selection, explaining the diversity of life on Earth. His " \
                "book 'On the Origin of Species' revolutionized biology.",
            metadata={"field": "Evolutionary Biology"}
        )
    doc5 = Document(
            page_content="Nikola Tesla was an inventor and electrical engineer known for his contributions to alternating current (AC) electricity systems. " \
                "His innovations helped shape the modern electric world.",
            metadata={"field": "Electrical Engineering"}
        )
    
    docs = [doc1, doc2, doc3, doc4, doc5]
    return docs

def get_vector_store(vstore_dir:str, collection_name:str):
    vector_store = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=vstore_dir,
        collection_name=collection_name
    )

    return vector_store

def add_docs(vector_store, docs):
    try:
        vector_store.add_documents(docs)
        print("Documents Added Successfully...")
    except Exception as e:
        print("Added Documents failed with Exception: ", e)

def get_docs(vector_store):
    return vector_store.get(include=['embeddings','documents', 'metadatas'])

def search_docs(vector_store, query, k=2):
   return vector_store.similarity_search(query=query, k=2)

def similarity_search(vector_store, query, k=2):
    return vector_store.similarity_search_with_score(query=query, k=2)

def metadata_filter(vector_store, filter:dict):
    return vector_store.similarity_search_with_score(query="", filter=filter)

def update_doc(vector_store, doc_id:str, document):
    try:
        vector_store.update_document(document_id=doc_id, document=document)
        print("Document Updated Successfully")
    except Exception as e:
        print("Document updation failed with exception: ", e)

def delete_doc(vector_store, doc_ids:list):
    vector_store.delete(ids=doc_ids)


if __name__=='__main__':
    # Create Docs
    docs = create_docs()

    # Create Vector Store
    vector_store = get_vector_store(
        vstore_dir="my_chroma_db",
        collection_name="sample"
    )

    # Adding documents to vector store
    add_docs(vector_store=vector_store, docs=docs)

    # Check the added documents
    db_docs1 = get_docs(vector_store=vector_store)
    print("\n", db_docs1)

    # Similarity Search
    result1 = search_docs(
        vector_store=vector_store,
        query="Who among these are an electric engineer?",
        k=2
    )
    print("\n", result1)

    # Similarity Search with score
    result2 = similarity_search(
        vector_store=vector_store,
        query="Who among these are an electric engineer?",
        k=2
    )
    print("\n", result2)

    # Filter with metadata
    filtered_data = metadata_filter(
        vector_store=vector_store,
        filter={"field": "Physics and Chemistry"}
    )
    print("\n", filtered_data)

    # update documents
    updated_doc1 = Document(
        page_content="Albert Einstein, one of the most influential physicists of the 20th century, is best known for developing the theory of relativity. " \
        "His groundbreaking equation, E=mc², revealed the relationship between mass and energy and laid the foundation for modern physics. " \
        "Einstein’s work not only transformed scientific thought but also influenced philosophical discussions about space and time. " \
        "Beyond his scientific contributions, he was a vocal advocate for peace, civil rights, and education.",
        metadata={"field": "Theoretical Physics"}
    )

    update_doc(
        vector_store=vector_store,
        doc_id="6bc0e53e-bbf3-4feb-b0f5-d4f8fa59adc4",  # Replace it with the doc id in your db
        document=updated_doc1
    )

    # Check the updated documents
    db_docs2 = get_docs(vector_store=vector_store)
    print("\n", db_docs2)

    # Similarly, we can use methods. 