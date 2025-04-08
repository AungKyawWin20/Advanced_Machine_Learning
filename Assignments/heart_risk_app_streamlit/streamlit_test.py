import streamlit as st

st.title("First Attempt")

st.write("This is my first streamlit.")

import streamlit as st

st.title("This is a title")
st.title("_Streamlit_ is :blue[cool] :sunglasses:")

option = st.selectbox(
    "Heart rate",
    ("high", "medium", "low"),
)
st.write("You selected:", option)
if st.button('Click'):
    st.write('button is clicked.')

