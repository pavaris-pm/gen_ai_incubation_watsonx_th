# Import environment loading library
from dotenv import load_dotenv
# Import IBMGen Library 
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain.llms.base import LLM
# Import lang Chain Interface object
from langChainInterface import LangChainInterface
# Import langchain prompt templates
from langchain.prompts import PromptTemplate
# Import system libraries
import os
# Import streamlit for the UI 
import streamlit as st
import re

from function import *

# Load environment vars
load_dotenv()

# Define credentials 
api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)


if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }



# Define generation parameters 
params = {
    GenParams.DECODING_METHOD: "sample",
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 300,
    GenParams.TEMPERATURE: 0.2,
    # GenParams.TOP_K: 100,
    # GenParams.TOP_P: 1,
    GenParams.REPETITION_PENALTY: 1
}

models = {
    "granite_chat":"ibm/granite-13b-chat-v2",
    "flanul": "google/flan-ul2",
    "llama2": "meta-llama/llama-2-70b-chat",
    "mixstral": 'ibm-mistralai/mixtral-8x7b-instruct-v01-q'
}
# define LangChainInterface model
llm = LangChainInterface(model=models["mixstral"], credentials=creds, params=params, project_id=project_id)

def prompt_template(question):
    text = f"""[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

QUESTION: {question} [/INST] ANSWER:"""
    return text

# Title for the app
st.title('🤖 Our First Q&A Front End')
# Prompt box 
prompt = st.text_input('Enter your prompt here')
print(prompt)
# If a user hits enter
if prompt:
    users_language = language_detector(prompt)
    # Pass the prompt to the llm
    text_to_model =  translate_to_thai(prompt, False)
    print('text_to_model')
    print(text_to_model)
    response_from_model = llm(prompt_template(text_to_model))
    print(response_from_model)

    if users_language == "th":
        translated_response = translate_to_thai(response_from_model, True)
        text_to_user = translated_response
    else:
        text_to_user = response_from_model
    # Write the output to the screen
    st.write(text_to_user)