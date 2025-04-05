from langchain_core.prompts import ChatPromptTemplate


chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),         # here: system ~ SystemMessage
    ('human', 'Explain in simple terms, what is {topic}')    # human ~ HumanMessage
])

prompt = chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)
