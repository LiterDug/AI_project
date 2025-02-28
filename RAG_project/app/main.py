import streamlit as st


home = st.Page(
    "views/home.py",
    title="Home",
    icon="🏠",
    default=True
)

chatbot = st.Page(
    "views/chatbot.py",
    title="Chatbot",
    icon="🤖",
)

pg = st.navigation(
    {
        "Home": [home],
        "Chatbot": [chatbot]

    }
)

pg.run()