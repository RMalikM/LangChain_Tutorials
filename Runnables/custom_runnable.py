"""
This code implementation dipicts the idea behind Runnables in LangChain.
Contains a simple custom runnable.
"""

from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
import random

# Define Custom Runnable Class
class CustomRunnable(ABC):
    @abstractmethod
    def invoke():
        pass

# Define a Custom LLM Class with dummy responses
class CustomLLM(CustomRunnable):
  def __init__(self):
    print('LLM created')

  def invoke(self, prompt):
    response_list = [
        "Why don't skeletons fight each other? They don't have the guts.",
        "Why did the scarecrow win an award? Because he was outstanding in his field.",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "Why did the math book look sad? Because it had too many problems.",
        "I asked the librarian if the library had any books on paranoia. She whispered, 'They're right behind you...'"
    ]
    return {'response': random.choice(response_list)}

# Create Custom Prompt Class
class CustomPromptTemplate(CustomRunnable):
  def __init__(self, template, input_variables):
    self.template = template
    self.input_variables = input_variables    

  def invoke(self, input_dict):
    # Check for missing variables
    for var in self.input_variables:
        if var not in input_dict:
            raise ValueError(f"Missing required variable: {var}")

    # Check for extra variables
    for key in input_dict.keys():
        if key not in self.input_variables:
            raise ValueError(f"Unexpected variable: {key}")
    return self.template.format(**input_dict)

# Create Custom String Output Parser
class CustomStrOutputParser(CustomRunnable):
  def __init__(self):
    pass

  def invoke(self, input_data):
    return input_data['response']

# Create Runnable Connector
class RunnableConnector(CustomRunnable):
  def __init__(self, runnable_list):
    self.runnable_list = runnable_list

  def invoke(self, input_data):
    for runnable in self.runnable_list:
      input_data = runnable.invoke(input_data)
    return input_data
  
# ---------- Single Chain Example ----------
template = CustomPromptTemplate(
   template="write a {length} joke about {topic}",
   input_variables=['length', 'topic']
)

model = CustomLLM()
parser = CustomStrOutputParser()
chain = RunnableConnector([template, model, parser])
result = chain.invoke({'length': 'short', 'topic': 'jupitor'})  # Doesn't matter here as we have defined some fixed responses
print("\nSingle Chain Result: ", result)


# ---------- Multiple Chains Example ----------
template1 = CustomPromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

template2 = CustomPromptTemplate(
    template='Explain the following joke {response}',
    input_variables=['response']
)

chain1 = RunnableConnector([template1, model])
chain2 = RunnableConnector([template2, model, parser])
final_chain = RunnableConnector([chain1, chain2])
result1 = final_chain.invoke({'topic':'cricket'})
print("\nMulti-Chain Result: ", result1)