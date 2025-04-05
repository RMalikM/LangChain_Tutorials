from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI()

class Review(BaseModel):
    key_themes: List[str] = Field(description="Write dowm the key themes discussed in the review")
    summary: str = Field(description="A brief history of the review")
    sentiment: Literal["Pos", "Neg", "Neu"] = Field(description="Return sentiment of the review as either Negative, Positive, or Neutral")
    pros: Optional[List[str]] = Field(description="Write down all the pros inside a list only if pros are available in "
    "the review otherwise do not include")
    cons: Optional[List[str]] = Field(description="Write down all the cons inside a list only if cons are available in "
    "the review otherwise do not include")
    name: Optional[str] = Field(description="Write the name of the reviewer only if present")

structured_model = chat_model.with_structured_output(Review, method='function_calling')

result = structured_model.invoke(
    """The XYZ Smartwatch is a sleek and feature-packed device that offers excellent health tracking, smooth performance, 
    and a vibrant display. Its battery life is impressive, lasting up to a week on a single charge, and the intuitive 
    interface makes navigation effortless. With accurate heart rate and sleep monitoring, it caters well to fitness 
    enthusiasts. However, the lack of third-party app support might be a drawback for some users. Overall, it's a great 
    choice for those seeking a stylish and reliable smartwatch.

    Pros: Long battery life, accurate health tracking, smooth performance, vibrant display.
    Cons: Limited third-party app support, no built-in GPS.
"""
)

print(result.model_dump())