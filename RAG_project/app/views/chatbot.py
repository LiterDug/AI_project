import os
import streamlit as st
import sys
sys.path.append(".")
st.markdown(os.listdir())


def setup_sidebar():
    with st.sidebar:
        st.title("Add Documents")
        st.header("Upload a Document")
        uploaded_file = st.file_uploader("Choose a document...", type=["pdf"])
        if uploaded_file is not None:
            st.write("File uploaded!")
            st.write(uploaded_file)
        else:
            st.write("No file uploaded.")

setup_sidebar()

from chatbot_core.rag_base import RAG

# init model
if "model" not in st.session_state:
    st.session_state.model = RAG(user_infor="Manh", session_id="1234")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Chào anh/chị ạ. Em có thể giúp gì cho anh/chị không ạ?"}]

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Nhập câu hỏi của bạn"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        client = st.session_state["model"]
        response = client.run(prompt)
        st.markdown(response)
    
    st.session_state["messages"].append({"role": "assistant", "content": response})
            










