from typing import TypedDict, Annotated, List, Optional, Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI()

class Review(TypedDict):
    key_themes: Annotated[List[str], "Write dowm the key themes discussed in the review"]
    summary: Annotated[str, "A brief history of the review"]
    sentiment: Annotated[Literal["Pos", "Neg", "Neu"], "Return sentiment of the review as either Negative, Positive, or Neutral"]
    pros: Annotated[Optional[List[str]], "Write down all the pros inside a list only if pros are available in "
    "the review otherwise do not include"]
    cons: Annotated[Optional[List[str]], "Write down all the cons inside a list only if cons are available in "
    "the review otherwise do not include"]
    name: Annotated[Optional[str], "Write the name of the reviewer only if present"]

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

print(result)