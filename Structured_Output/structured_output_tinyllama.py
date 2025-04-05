"""
IMPORTANT: This code will not work as Tiny_llama does not support with_structured_output method.
This is an example to show that there are some models that do not support with_structured_output
and we need to write OUTPUT PARSER for these models.
"""
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# schema
class Review(BaseModel):

    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")
    

structured_model = model.with_structured_output(Review)

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

print(result)