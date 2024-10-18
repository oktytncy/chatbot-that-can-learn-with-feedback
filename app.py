import os
import json
import requests
import streamlit as st
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(levelname)s:%(message)s')

# Constants for model options
EMBEDDING_MODEL_OPTIONS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002"
]
LANGUAGE_MODEL_OPTIONS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-125"
]

# Configuration and tweaks files
CONFIG_FILE_PATH = os.path.join('conf', 'config.json')
TWEAKS_FILE_PATH = os.path.join('tweaks', 'tweaks.json')

@st.cache_data
def load_config_from_file(filepath: str) -> dict:
    """ Load configuration from the JSON file """
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error loading configuration from {filepath}: {e}")

@st.cache_data
def load_tweaks_from_file(filepath: str) -> dict:
    """ Load tweaks from the JSON file """
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        st.warning(f"Tweaks file not found at {filepath}. Continuing without tweaks.")
        return {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing tweaks JSON from {filepath}: {e}")

# Load configuration and tweaks
config = load_config_from_file(CONFIG_FILE_PATH)
TWEAKS = load_tweaks_from_file(TWEAKS_FILE_PATH)

# Constants for API
BASE_API_URL = "https://api.langflow.astra.datastax.com"
LANGFLOW_ID = config.get('langflow_id')
APPLICATION_TOKEN = config.get('application_token')
ENDPOINT = config.get('flow_id')

EMBEDDING_COMPONENT_ID = config.get('embedding_component_id')
LANGUAGE_MODEL_COMPONENT_ID = config.get('language_model_component_id')

def run_flow(
    message: str,
    endpoint: str,
    output_type: str = "chat",
    input_type: str = "chat",
    tweaks: Optional[dict] = None,
    application_token: Optional[str] = None
) -> dict:
    """Send the request to Langflow API and return the response."""
    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{endpoint}"
    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type
    }

    if tweaks:
        payload["tweaks"] = tweaks

    headers = {"Content-Type": "application/json"}
    if application_token:
        headers["Authorization"] = f"Bearer {application_token}"

    response = requests.post(api_url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

def main():
    st.markdown("<h1 style='text-align: center; color: grey;'>Langflow and AstraDB Vector Integration</h1>", unsafe_allow_html=True)

    # Add an image at the top of the page
    st.image("images/chatbot.png", caption="", use_column_width=True)
    # If using a URL:
    # st.image("https://example.com/your_image.png", caption="Your Image Caption", use_column_width=True)
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Empty list to store the messages

    # Sidebar: Session parameters
    with st.sidebar:
        st.header("Session Parameters")
        embedding_model = st.selectbox(
            "Embedding Model",
            options=EMBEDDING_MODEL_OPTIONS,
            index=0
        )
        language_model = st.selectbox(
            "Language Model",
            options=LANGUAGE_MODEL_OPTIONS,
            index=0
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Temperature controls the randomness of the model's responses. Low values make responses more focused and realistic, while high values make them more creative or fantastical."
        )

    # Display the chat history using st.chat_message
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input at the bottom using st.chat_input
    user_input = st.chat_input("Enter your message here:")

    if user_input:
        # Add user's message to the session state and display it
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Update tweaks with user selections
        tweaks = TWEAKS.copy()
        tweaks[EMBEDDING_COMPONENT_ID] = {'model_name': embedding_model}
        tweaks[LANGUAGE_MODEL_COMPONENT_ID] = {
            'model_name': language_model,
            'temperature': temperature
        }

        try:
            # Send the message to the Langflow API
            response = run_flow(
                message=user_input,
                endpoint=ENDPOINT,
                tweaks=tweaks,
                application_token=APPLICATION_TOKEN
            )

            # Extract the answer
            answer = response['outputs'][0]['outputs'][0]['results']['message']['data']['text']

            # Add assistant's message to the session state and display it
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()