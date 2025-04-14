from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

text = """
Children were playing in the park, laughing and running around as the evening breeze gently blew through the trees. 
The sky turned orange as the sun began to set, and birds returned to their nests. The Olympic Games are one of the 
biggest sporting events in the world. Athletes from many countries come together to compete and make their nations proud.

Pollution is a serious threat to our environment and health. It makes the air dirty, harms animals, and affects the climate. 
When rivers and forests are polluted, nature suffers greatly. To stop pollution, we need clean habits, strong rules, and help 
from everyone who wants a better and greener world.
"""

docs = text_splitter.create_documents([text])
print(len(docs))
print(docs)

