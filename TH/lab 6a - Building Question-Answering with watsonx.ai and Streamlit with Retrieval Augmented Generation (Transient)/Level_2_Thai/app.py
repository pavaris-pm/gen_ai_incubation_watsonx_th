import logging
import os
import pickle
import tempfile

import streamlit as st
from dotenv import load_dotenv
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import \
    GenTextReturnOptMetaNames as ReturnOptions
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import (HuggingFaceHubEmbeddings,
                                  HuggingFaceInstructEmbeddings)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from PIL import Image

from langChainInterface import LangChainInterface
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy as np
import faiss
from function import vector_search, get_model, generate_prompt_th, generate_prompt_en, format_docs_for_display, format_documents, detect_language

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

if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }

GEN_API_KEY = os.getenv("GENAI_KEY", None)

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
    max_new_tokens= st.number_input('max_new_tokens',1,1024,value=500)
    min_new_tokens= st.number_input('min_new_tokens',0,value=15)
    repetition_penalty = st.number_input('repetition_penalty',1,value=1)
    decoding = st.text_input(
            "Decoding",
            "greedy",
            key="placeholder",
        )
    
uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True)

@st.cache_data
def read_pdf(uploaded_files,chunk_size =1000,chunk_overlap=20):
    for uploaded_file in uploaded_files:
      bytes_data = uploaded_file.read()
      with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
      # Write content to the temporary file
          temp_file.write(bytes_data)
          filepath = temp_file.name
          with st.spinner('Waiting for the file to upload'):
             loader = PyPDFLoader(filepath)
             data = loader.load()
             text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size, chunk_overlap=chunk_overlap)
             docs = text_splitter.split_documents(data)
             return docs

# embeddings_model = get_model(model_name='airesearch/wangchanberta-base-att-spm-uncased', max_seq_length=768)
embeddings_model = get_model(model_name='kornwtp/simcse-model-phayathaibert', max_seq_length=768)
@st.cache_data
def read_push_embeddings():
    # Use your get_model function to get the embeddings model

    global docs  # Ensure that docs is accessible globally or passed as a parameter
    if not docs:
        raise ValueError("Documents are not loaded or empty.")
    
    docs_texts = [doc.page_content for doc in docs]
    # Check if the FAISS index already exists
    if os.path.exists("db.pickle"):
        with open("db.pickle", 'rb') as file_name:
            index = pickle.load(file_name)
    else:
        # Encode the documents using the embeddings model
        embeddings = embeddings_model.encode(docs_texts)

        # Step 1: Change data type
        embeddings = np.array(embeddings).astype("float32")

        # Step 2: Instantiate the index
        index = faiss.IndexFlatL2(embeddings.shape[1])

        # Step 3: Pass the index to IndexIDMap
        index = faiss.IndexIDMap(index)

        # Step 4: Add vectors and their IDs
        doc_ids = np.array(range(len(docs_texts))).astype('int64')

        index.add_with_ids(embeddings, doc_ids)

        # Save the index for future use
        with open('db.pickle', 'wb') as file_name:
            pickle.dump(index, file_name)

    return index


def fix_encoding(text):
    try:
        # Attempt to decode the text using 'ISO-8859-1' and then re-encode it in 'UTF-8'
        fixed_text = text.encode('ISO-8859-1').decode('UTF-8')
    except UnicodeDecodeError:
        # If there's a decoding error, return the original text
        fixed_text = text
    return fixed_text


stream = True
# show user input
if user_question := st.text_input(
    "Ask a question about your Policy Document:"
):
    users_language = detect_language(user_question)
    docs = read_pdf(uploaded_files)
    print("DOCS HERE", docs)
    index = read_push_embeddings()
    D, I = vector_search(user_question, embeddings_model, index, num_results=4)
    search_results = [docs[i] for i in I[0]]
    # docs = db.similarity_search(user_question)
    # params = {
    #     'decoding_method': "greedy",
    #     'min_new_tokens': 30,
    #     'max_new_tokens': 500,
    #     'temperature': 0.0,
    #     # GenParams.TOP_K: 100,
    #     # GenParams.TOP_P: 1,
    #     'repetition_penalty': 1
    # }
    params = {
        'decoding_method': 'greedy',
        'min_new_tokens': min_new_tokens,
        'max_new_tokens': max_new_tokens,
        'random_seed': 42,
        # 'temperature': 0.7,
        # 'repetition_penalty': 1.03,
        # GenParams.RETURN_OPTIONS: {ReturnOptions.GENERATED_TOKENS: True,
        #                             ReturnOptions.INPUT_TOKENS: True}
    }
    # model_llm = LangChainInterface(model='ibm-mistralai/mixtral-8x7b-instruct-v01-q', credentials=creds, params=params, project_id=project_id)
    model_llm = Model('ibm-mistralai/mixtral-8x7b-instruct-v01-q',
                    params=params, credentials=creds,
                    project_id=project_id)
    formated_search = format_documents(search_results)
    # 
    formatted_text_for_display = format_docs_for_display(formated_search)

    if users_language == "th":
        current_prompt = generate_prompt_th(user_question, formated_search)
    elif users_language == "en":
        current_prompt = generate_prompt_en(user_question, formated_search)
    print(current_prompt)
    # response = chain.run(input_documents=docs, question=user_question)
    # Call the function with your formatted_docs
    
    if stream == False:
        joined_result = model_llm.generate_text(current_prompt)
        st.text_area(label="Model Response", value=joined_result, height=300)
    elif stream == True:
        model_response_placeholder = st.empty()
        full_response = []
        for response in model_llm.generate_text_stream(prompt=current_prompt):
                wordstream = str(response)
                print(response)
                if wordstream:
                    wordstream = fix_encoding(wordstream)
                    full_response.append(wordstream)
                    result = "".join(full_response).strip()
                    with model_response_placeholder.container():
                        # if len(result) > curr_len:
                        print(len(result))
                        st.markdown('---')
                        st.markdown('#### Response:')
                        st.markdown(result)
                        # st.markdown(translate_large_text(result,translate_to_thai, True))
                        st.markdown('---')
        joined_result = "".join(full_response).strip()
    st.text_area(label="Reference", value=formatted_text_for_display, height=300)
    st.write()
