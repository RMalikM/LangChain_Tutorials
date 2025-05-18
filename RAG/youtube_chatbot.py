from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()


def get_transcript(video_id, languages=["en"]):
    """
    Fetches the transcript for a given YouTube video ID.
    Args:
        video_id (str): The YouTube video ID.
        languages (list): List of languages to fetch the transcript in.
    """
    try:
        obj = YouTubeTranscriptApi()
        fetched_transcript  = obj.fetch(video_id=video_id, languages=languages)

        # Flatten it to plain text
        transcript = " ".join(snippet.text for snippet in fetched_transcript )
        return transcript
    except TranscriptsDisabled:
        print("No captions available for this video.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def split_transcript(transcript, chunk_size=1000, chunk_overlap=100):
    """
    Splits the transcript into smaller chunks.
    Args:
        transcript (str): The full transcript text.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents([transcript])
    return chunks

def index_document(doc_chunks):
    """
    Indexes the document chunks using FAISS and OpenAI embeddings.
    Args:
        doc_chunks (list): List of document chunks to index.
    """
    # Initialize the FAISS vector store with OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(doc_chunks, embeddings)
    return vector_store

def retrieve_document(vector_store, question):
    """
    Queries the indexed document chunks.
    Args:
        vector_store (FAISS): The FAISS vector store containing the indexed documents.
        question (str): The question to ask.
    """
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(question)
    return retrieved_docs

def query_document_with_context(retrieved_docs, question):
    """
    Queries the indexed document chunks with context.
    Args:
        retrieved_docs: List of the relevant documents.
        question (str): The question to ask.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables = ['context', 'question']
    )
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = prompt.invoke({"context": context_text, "question": question})
    result = llm.invoke(final_prompt)
    return result


if __name__ == "__main__":
    video_id = "ZXiruGOCn9s"  # Replace with your YouTube video ID

    print("Fetching transcript...")
    transcript = get_transcript(video_id)
    print("Transcript fetched successfully.")
    if transcript:
        print("Transcript: ", transcript[:1000])
    else:
        print("Transcript not available. Exiting.")
        exit(1)
    
    if transcript:
        print("Splitting transcript into chunks...")
        doc_chunks = split_transcript(
            transcript,
            chunk_size=1000,
            chunk_overlap=100
        )
        print("Transcript split into chunks successfully.")
        print("Document chunks: ", doc_chunks[:2])
        print("Indexing document chunks...")
        vector_store = index_document(doc_chunks)
    else:
        print("Transcript not available. Exiting.")
        exit(1)

    # Example question
    question = "what is transformers?"
    retrieved_docs = retrieve_document(vector_store, question)
    print("Retrieved documents: ", retrieved_docs)
    if not retrieved_docs:
        print("No relevant documents found.")
        exit(1)

    answer = query_document_with_context(retrieved_docs, question)
    print("Answer: \n", answer.content)
