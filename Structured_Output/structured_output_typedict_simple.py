from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI()

class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = chat_model.with_structured_output(Review)

result = structured_model.invoke(
    """The **XYZ Smartwatch** is a sleek and feature-packed device that offers excellent health tracking, 
    smooth performance, and a vibrant display. Its battery life is impressive, lasting up to a week on a 
    single charge, and the intuitive interface makes navigation effortless. With accurate heart rate and 
    sleep monitoring, it caters well to fitness enthusiasts. However, the lack of third-party app support 
    might be a drawback for some users. Overall, it's a great choice for those seeking a stylish and reliable smartwatch.
"""
)

print(result)