from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI()

# schema
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write dowm the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["Pos", "Neg", "Neu"],
      "description": "Return sentiment of the review as either Negative, Positive, or Neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros in a list only if present otherwise null"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons in a list only if present otherwise null"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

structured_model = chat_model.with_structured_output(schema=json_schema, method='function_calling')

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