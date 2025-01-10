import streamlit as st
from vc_chain import create_chain

# Load the LangChain pipeline
@st.cache_resource
def load_chain():
    return create_chain()

chain = load_chain()

# Streamlit app UI
st.title("Your Personal Virtual Companion 🧑‍🤝‍🧑")
st.write("We are here to support you during your outreach to make your volunteering experience smoother. With us you will always have a volunteering companion! 🤝😊")

# Input field for the user to enter their question
user_question = st.text_input("Your Question:", placeholder="Type your question here...")

if st.button("Ask"):
    if user_question.strip():
        # Process the user's question with the chain
        with st.spinner("Generating response..."):
            response = chain.invoke({"question": user_question})
        st.success("Response:")
        st.write(response)
    else:
        st.warning("Please enter a question to get a response!")
