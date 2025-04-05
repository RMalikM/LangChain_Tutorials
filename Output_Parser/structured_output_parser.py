from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()



# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-2-2b-it",
#     task="text-generation"
# )

# chat_model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description="Fact 1 about the topoic"),
    ResponseSchema(name='fact_2', description="Fact 2 about the topoic"),
    ResponseSchema(name='fact_3', description="Fact 3 about the topoic"),
    ResponseSchema(name='fact_4', description="Fact 4 about the topoic"),
    ResponseSchema(name='fact_5', description="Fact 5 about the topoic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

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
