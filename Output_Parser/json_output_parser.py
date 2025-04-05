from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-2-2b-it",
#     task="text-generation"
# )

# chat_model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the 5 facts about the {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# chain = template | chat_model | parser

# result = chain.invoke({'topic': 'Black Hole'})
# print(result)
# print(type(result))

# Could not run gemma. Access to model is restricted
# Instead using Gpt
chat_model_gpt = ChatOpenAI()

chain = template | chat_model_gpt | parser

result = chain.invoke({'topic': 'Black Hole'})
print(result)
print(type(result))
