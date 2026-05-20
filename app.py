import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Therapist",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Therapist")
st.caption("A simple AI therapist chatbot built with Streamlit + LangChain + Ollama")

# ---------------- LOAD MODEL ---------------- #

llm = ChatOllama(
    model="qwen2.5:1.5b",   # change model if needed
    temperature=0.7
)

# ---------------- PROMPT TEMPLATE ---------------- #

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a calm, empathetic AI therapist.

Your job is to:
- Listen carefully
- Respond warmly
- Ask thoughtful follow-up questions
- Help users reflect on emotions
- Avoid being judgmental
- Keep responses concise and comforting

Do NOT claim to be a real licensed therapist.
If the user mentions self-harm or suicide, encourage them to seek immediate professional help.
"""
        ),
        ("human", "{input}")
    ]
)

# ---------------- CHAT HISTORY ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- USER INPUT ---------------- #

user_input = st.chat_input("How are you feeling today?")

if user_input:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    chain = prompt | llm | StrOutputParser()

    ai_response = chain.invoke(
        {
            "input": user_input
        }
    )

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(ai_response)

    # Save AI response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )