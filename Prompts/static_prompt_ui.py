from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

chat_model = ChatOpenAI()

st.header("AI Research Tool")

user_input = st.text_input("Enter your query")

if st.button("Summarize"):
    result = chat_model.invoke(user_input)

    st.write(result.content)
    
