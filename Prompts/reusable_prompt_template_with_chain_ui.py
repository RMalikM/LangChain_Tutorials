from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

chat_model = ChatOpenAI()

st.header("AI Research Tool")

paper_input = st.selectbox( 
    "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", \
                                   "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"]
    )
style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]) 
length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

# Load Prompt Template: reusable property
template = load_prompt("template.json")

if st.button("Summarize"):
    chain = template | chat_model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)
    
