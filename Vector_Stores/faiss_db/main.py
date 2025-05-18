from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os

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

def get_vector_store(index_dir:str):
    # If the index already exists, load it
    if os.path.exists(f"{index_dir}/index.faiss") and os.path.exists(f"{index_dir}/index.pkl"):
        vector_store = FAISS.load_local(
            folder_path=index_dir,
            embeddings=OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )
        print("Loaded existing FAISS index")
    else:
        # If the index doesn't exist, create a new one with empty documents
        vector_store = FAISS.from_documents(
            documents=[],
            embedding=OpenAIEmbeddings()
        )
        # Create directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)
        # Save the empty index
        vector_store.save_local(index_dir)
        print("Created new FAISS index")

    return vector_store

def add_docs(vector_store, docs):
    try:
        vector_store.add_documents(docs)
        print("Documents Added Successfully...")
    except Exception as e:
        print("Added Documents failed with Exception: ", e)

def get_docs(vector_store):
    # FAISS doesn't have a direct get_all method like Chroma
    # We can retrieve docs by doing a generic search
    all_docs = vector_store.similarity_search(
        query="", k=1000  # Use a large k to get all documents
    )
    return all_docs

def search_docs(vector_store, query, k=2):
    return vector_store.similarity_search(query=query, k=k)

def similarity_search(vector_store, query, k=2):
    docs_and_scores = vector_store.similarity_search_with_score(query=query, k=k)
    return docs_and_scores

def metadata_filter(vector_store, filter:dict):
    # In FAISS, we need to do filtering manually after retrieval
    # First, get all documents
    all_docs = get_docs(vector_store)
    
    # Then filter by metadata
    filtered_docs = []
    for doc in all_docs:
        match = True
        for key, value in filter.items():
            if key not in doc.metadata or doc.metadata[key] != value:
                match = False
                break
        if match:
            filtered_docs.append(doc)
    
    # Return the filtered documents with a placeholder score of 1.0
    return [(doc, 1.0) for doc in filtered_docs]

def update_doc(vector_store, doc_id:str, document):
    try:
        # FAISS doesn't support direct updates by ID
        # We need to delete and re-add the document
        
        # First, get all documents
        all_docs = get_docs(vector_store)
        
        # Create a new FAISS instance without the document to update
        updated_docs = [doc for doc in all_docs if doc.metadata.get("id") != doc_id]
        
        # Add the new document with the same ID
        document.metadata["id"] = doc_id
        updated_docs.append(document)
        
        # Create a new vector store with the updated documents
        new_vector_store = FAISS.from_documents(
            documents=updated_docs,
            embedding=OpenAIEmbeddings()
        )
        
        # Replace the old vector store with the new one
        vector_store = new_vector_store
        print("Document Updated Successfully")
        return vector_store
    except Exception as e:
        print("Document updation failed with exception: ", e)
        return vector_store

def delete_doc(vector_store, metadata_filter:dict):
    try:
        # FAISS doesn't support direct deletion by ID
        # We need to create a new index without the documents to delete
        
        # First, get all documents
        all_docs = get_docs(vector_store)
        
        # Filter out documents that match the filter
        new_docs = []
        for doc in all_docs:
            should_keep = True
            for key, value in metadata_filter.items():
                if key in doc.metadata and doc.metadata[key] == value:
                    should_keep = False
                    break
            if should_keep:
                new_docs.append(doc)
        
        # Create a new vector store without the deleted documents
        new_vector_store = FAISS.from_documents(
            documents=new_docs,
            embedding=OpenAIEmbeddings()
        )
        
        # Replace the old vector store with the new one
        vector_store = new_vector_store
        print(f"Documents matching filter {metadata_filter} deleted successfully")
        return vector_store
    except Exception as e:
        print("Document deletion failed with exception: ", e)
        return vector_store

def save_vector_store(vector_store, index_dir:str):
    try:
        vector_store.save_local(index_dir)
        print(f"Vector store saved to {index_dir}")
    except Exception as e:
        print(f"Failed to save vector store: {e}")


if __name__=='__main__':
    # Create Docs
    docs = create_docs()

    # Create or load Vector Store
    index_dir = "faiss_index"
    vector_store = get_vector_store(index_dir=index_dir)

    # Adding documents to vector store
    add_docs(vector_store=vector_store, docs=docs)
    
    # Save the vector store
    save_vector_store(vector_store, index_dir)

    # Check the added documents
    db_docs1 = get_docs(vector_store=vector_store)
    print("\nAll documents:")
    for doc in db_docs1[:2]:  # Print just a few to avoid clutter
        print(f"- {doc.page_content[:50]}...")
    print(f"Total: {len(db_docs1)} documents")

    # Similarity Search
    result1 = search_docs(
        vector_store=vector_store,
        query="Who among these are an electric engineer?",
        k=2
    )
    print("\nSimilarity search results:")
    for doc in result1:
        print(f"- {doc.page_content[:50]}...")

    # Similarity Search with score
    result2 = similarity_search(
        vector_store=vector_store,
        query="Who among these are an electric engineer?",
        k=2
    )
    print("\nSimilarity search with scores:")
    for doc, score in result2:
        print(f"- Score: {score:.4f} | {doc.page_content[:50]}...")

    # Filter with metadata
    filtered_data = metadata_filter(
        vector_store=vector_store,
        filter={"field": "Physics and Chemistry"}
    )
    print("\nMetadata filter results:")
    for doc, score in filtered_data:
        print(f"- {doc.page_content}")

    # Update documents
    # First, assign IDs to the documents
    for i, doc in enumerate(get_docs(vector_store)):
        doc.metadata["id"] = f"doc_{i}"
    
    # Now update a document
    updated_doc1 = Document(
        page_content="Albert Einstein, one of the most influential physicists of the 20th century, is best known for developing the theory of relativity. " \
        "His groundbreaking equation, E=mc², revealed the relationship between mass and energy and laid the foundation for modern physics. " \
        "Einstein's work not only transformed scientific thought but also influenced philosophical discussions about space and time. " \
        "Beyond his scientific contributions, he was a vocal advocate for peace, civil rights, and education.",
        metadata={"field": "Theoretical Physics", "id": "doc_0"}
    )

    vector_store = update_doc(
        vector_store=vector_store,
        doc_id="doc_0",
        document=updated_doc1
    )
    
    # Save after update
    save_vector_store(vector_store, index_dir)

    # Check the updated documents
    db_docs2 = get_docs(vector_store=vector_store)
    print("\nAfter update - first document:")
    print(db_docs2[0].page_content)
    
    # Delete a document
    vector_store = delete_doc(
        vector_store=vector_store,
        metadata_filter={"field": "Evolutionary Biology"}
    )
    
    # Save after deletion
    save_vector_store(vector_store, index_dir)
    
    # Check remaining documents
    remaining_docs = get_docs(vector_store=vector_store)
    print(f"\nRemaining documents after deletion: {len(remaining_docs)}")
    for doc in remaining_docs:
        print(f"- Field: {doc.metadata.get('field')}")