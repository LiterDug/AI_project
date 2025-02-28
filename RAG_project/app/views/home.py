import streamlit as st

@st.dialog("Đăng kí")   # dialog decorator
def login_form():
    with st.form("login_form"):             # with => manage context
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Submit")

    if submit:
        if not username: 
            st.error("username is empty")
            st.stop()
        if not password:
            st.error("password is empty")
            st.stop()

@st.dialog("Đăng nhập")
def signup_form():
    with st.form("signup_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        re_password = st.text_input("Re-enter Password", type="password")
        submit = st.form_submit_button("Submit")
    if submit:
        if not username: 
            st.error("Username is empty")
            st.stop()
        if not password:
            st.error("Password is empty")
            st.stop()

        if password != re_password:
            st.error("Passwords do not match")
            st.stop()
        else:
            st.success("Passwords matched")


column1, column2 = st.columns(2,gap="medium", vertical_alignment= "center")
with column1: 
    if st.button("Login"):
        login_form()
    if st.button("Sign Up"):
        signup_form()
with column2:
    st.title("My chatbot")

