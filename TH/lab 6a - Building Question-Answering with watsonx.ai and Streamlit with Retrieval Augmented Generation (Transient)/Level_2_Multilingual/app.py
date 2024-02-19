import logging
import os
import pickle
import tempfile
import streamlit as st
from dotenv import load_dotenv
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain.callbacks import StdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import (HuggingFaceHubEmbeddings)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from PIL import Image
from huggingface_hub import login
from langChainInterface import LangChainInterface
from function import generate_prompt, translate_to_thai, translate_large_text, language_detector

# Most GENAI logs are at Debug level.
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

st.set_page_config(
    page_title="Retrieval Augmented Generation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.header("Retrieval Augmented Generation with watsonx.ai 💬")
# chunk_size=1500
# chunk_overlap = 200

load_dotenv()

handler = StdOutCallbackHandler()

api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)
neural_seek_url = os.getenv("NEURAL_SEEK_URL", None)
neural_seek_api_key = os.getenv("NEURAL_SEEK_API_KEY", None)
hgface_token = os.environ["HGFACE_TOKEN"]

login(token=hgface_token, add_to_git_credential=False, write_permission=False)

if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }



# Sidebar contents
with st.sidebar:
    st.title("RAG App")
    st.markdown('''
    ## About
    This app is an LLM-powered RAG built using:
    - [IBM Generative AI SDK](https://github.com/IBM/ibm-generative-ai/)
    - [HuggingFace](https://huggingface.co/)
    - [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) LLM model
 
    ''')
    st.write('Powered by [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai)')
    image = Image.open('watsonxai.jpg')
    st.image(image, caption='Powered by watsonx.ai')
    max_new_tokens= st.number_input('max_new_tokens',1,1024,value=300)
    min_new_tokens= st.number_input('min_new_tokens',0,value=15)
    repetition_penalty = st.number_input('repetition_penalty',1,2,value=1)
    decoding = st.text_input(
            "Decoding",
            "greedy",
            key="placeholder",
        )
    
params = {
    'decoding_method': decoding,
    'min_new_tokens': min_new_tokens,
    'max_new_tokens': max_new_tokens,
    'temperature': 0.0,
    'stop_sequences': ['END_KEY'],
    # GenParams.TOP_K: 100,
    # GenParams.TOP_P: 1,
    'repetition_penalty': repetition_penalty
}
    
uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True)

@st.cache_data
def read_pdf(uploaded_files, chunk_size=600, chunk_overlap=60):
    translated_docs = []

    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
            # Write content to the temporary file
            temp_file.write(bytes_data)
            filepath = temp_file.name

            with st.spinner('Waiting for the file to upload'):
                loader = PyPDFLoader(filepath)
                data = loader.load()

                for doc in data:
                    # Extract the content of the document
                    content = doc.page_content

                    # Translate the content
                    translated_content = translate_large_text(content, translate_to_thai, False)

                    # Replace original content with translated content
                    doc.page_content = translated_content
                    translated_docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(translated_docs)
    
    return docs


@st.cache_data
def read_push_embeddings():
    embeddings = HuggingFaceHubEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2", huggingfacehub_api_token=hgface_token)
    if os.path.exists("db.pickle"):
        with open("db.pickle",'rb') as file_name:
            db = pickle.load(file_name)
    else:     
        db = FAISS.from_documents(docs, embeddings)
        with open('db.pickle','wb') as file_name  :
             pickle.dump(db,file_name)
        st.write("\n")
    return db

# show user input
if user_question := st.text_input(
    "Ask a question about your Policy Document:"
):  
    users_language = language_detector(user_question)
    translated_user_input = translate_to_thai(user_question, False)
    docs = read_pdf(uploaded_files)
    db = read_push_embeddings()
    docs = db.similarity_search(translated_user_input)
    print('docs'+"*"*5)
    print(docs)
    print("*"*5)
    # model_llm = LangChainInterface(model=meta-llama/llama-2-70b-chat, credentials=creds, params=params, project_id=project_id)
    model_llm = LangChainInterface(model="meta-llama/llama-2-13b-chat", credentials=creds, params=params, project_id=project_id)
    custom_prompt = generate_prompt(question=translated_user_input, context=docs, model_type="llama-2")
    print("\nCUSTOM PROMPT:\n", custom_prompt)
    response = model_llm(custom_prompt)

    # Response
    if users_language == "th":
        translated_response = translate_to_thai(response, True)
    else:
        translated_response = response
    translated_response = translated_response.replace("<|endoftext|>", "")
    st.text_area(label="Model Response", value=translated_response, height=500)
    st.write()

