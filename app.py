import chat
import streamlit as st
from streamlit_chat import message




# # The page configuration for the app
# st.set_page_config(page_title="🦜️🔗Langchain PDF Chatbot 🤖", layout='centered')

# The title with HTML content
title_html = """
<h1>BuildRegulationsBot: LLM-Powered 🦜️🔗 Merged Docs Assistance 
<i class="fa-sharp fa-solid fa-user-helmet-safety"></i></h1>
"""
st.markdown(title_html, unsafe_allow_html=True)

# Subheader to provide additional information
st.subheader("Guiding You through Merged Approved Documents of Building Regulations using Langchain🦜️🔗")


# Initialize session_state variables to store chat history
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# Function to clear the input text
def clear_input_text():
    global input_text
    input_text = ""

# Get the user's input by calling the get_text function
def get_text():
    global input_text
    input_text = st.text_input("Ask your Question", key="input", on_change=clear_input_text)
    return input_text

def main():

    # Get user input
    user_input = get_text()

    if user_input:
        # Get the bot's response using the chat.answer function
        output = chat.answer(user_input)

        # Store the user's input and the bot's output in session_state
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

     # Display chat history
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            # Display bot's response
            message(st.session_state["generated"][i], key=str(i))
            # Display user's input
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

# Run the app
if __name__ == "__main__":
    main()

