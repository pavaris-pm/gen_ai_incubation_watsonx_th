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
    'decoding_method': "greedy",
    'min_new_tokens': 1,
    'max_new_tokens': 300,
    'random_seed': 42,
    # 'temperature': 0.2,
    # GenParams.TOP_K: 100,
    # GenParams.TOP_P: 1,
    'repetition_penalty': 1.05
}

models = {
    "granite_chat":"ibm/granite-13b-chat-v2",
    "flanul": "google/flan-ul2",
    "llama2": "meta-llama/llama-2-70b-chat",
    "mixstral": 'ibm-mistralai/mixtral-8x7b-instruct-v01-q'
}

def detect_language(text):
    thai_chars = set("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮะัาำิีึืุูเแโใไ็่้๊๋์")
    if any(char in thai_chars for char in text):
        return "th"
    else:
        return "en"

# define LangChainInterface model
llm = LangChainInterface(model=models["mixstral"], credentials=creds, params=params, project_id=project_id)

def prompt_template(question, lang="en"):
    if lang == "en":
        text = f"""[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>

    QUESTION: {question} [/INST] ANSWER:"""
    elif lang == "th":
        text = f"""[INST] <<SYS>>
    คุณเป็นผู้ช่วยที่เป็นประโยชน์และให้ความเคารพ โปรดตอบอย่างเป็นประโยชน์ที่สุดเท่าที่จะเป็นไปได้เสมอโดยต้องปลอดภัย คำตอบของคุณไม่ควรมีเนื้อหาที่เป็นอันตราย ผิดจรรยาบรรณ เหยียดเชื้อชาติ เหยียดเพศ เป็นพิษ เป็นอันตราย หรือผิดกฎหมาย โปรดตรวจสอบให้แน่ใจว่าคำตอบของคุณมีลักษณะเป็นกลางทางสังคมและมีลักษณะเชิงบวก

     หากคำถามไม่สมเหตุสมผลหรือไม่สอดคล้องกันตามข้อเท็จจริง ให้อธิบายเหตุผลแทนที่จะตอบสิ่งที่ไม่ถูกต้อง หากคุณไม่ทราบคำตอบสำหรับคำถาม โปรดอย่าเปิดเผยข้อมูลที่เป็นเท็จ
    
     คุณจะได้รับคำถามจากผู้ใช้ใน ''' ด้านล่าง กรุณาตอบคำถามเป็นภาษาไทย
    <</SYS>>
    ```
    คำถาม: {question}
    ```
    คำถาม: {question} [/INST] คำตอบ:"""
    return text

# Title for the app
st.title('🤖 Our First Q&A Front End')
# Prompt box 
prompt = st.text_input('Enter your prompt here')
# If a user hits enter
if prompt: 
    # Pass the prompt to the llm
    users_language = detect_language(prompt)
    if users_language == "th":
        current_prompt = prompt_template(prompt, lang="th")
    elif users_language == "en":
        current_prompt = prompt_template(prompt, lang="en")
    response = llm(current_prompt)
    # Write the output to the screen
    st.write(response)

# เมืองหลวงประเทษไทย คือที่ไหน
