from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

from dotenv import load_dotenv

load_dotenv()

chat_model1 = ChatOpenAI()

chat_model2 = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

prompt1 = PromptTemplate(
    template="Generate simple and short notes for the following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 5 short questions answers for a quiz from the following text \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Create a single document for notes and quiz using the following: \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

# Sometimes HuggingFace Models API are unavailable. Use chat_model1 in place if chat_model2 as well.
parallel_chain = RunnableParallel({
    'notes': prompt1 | chat_model1 | parser,
    'quiz': prompt2 | chat_model2 | parser
})

merge_chain = prompt3 | chat_model1 | parser

final_chain = parallel_chain | merge_chain

text = """Popular AI Agent Architectures
AI agent architectures come in various flavors, each with its own strengths and weaknesses:

1. Reactive Architectures: These agents are purely reactive, making decisions based solely on the current 
situation without considering past experiences. They are simple but effective for tasks with limited complexity. 
A classic example is a thermostat, which simply turns the heating or cooling on or off based on the current temperature.

2. Deliberative Architectures: These agents maintain an internal model of the world, enabling them to plan and reason 
about the future consequences of their actions. They are more sophisticated than reactive agents but can be computationally 
demanding. Chess-playing AIs often use deliberative architectures to evaluate potential moves and their long-term implications.

3. Belief-Desire-Intention (BDI) Architectures: A type of deliberative architecture, BDI agents explicitly model their beliefs, 
desires, and intentions. This allows for more complex reasoning and decision-making, often found in intelligent personal assistants 
or virtual characters in video games.

4. Hybrid Architectures: These architectures combine elements of both reactive and deliberative approaches, balancing the need for 
quick reactions with the ability to plan for the future. Autonomous driving systems, for instance, must react quickly to changing 
traffic conditions while also planning a route to a destination.
"""

# Visualize Chain
final_chain.get_graph().print_ascii()

result = final_chain.invoke({'text': text})
print(result)