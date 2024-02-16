import logging
import os
import pickle
import tempfile

import streamlit as st
from dotenv import load_dotenv
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import (HuggingFaceHubEmbeddings,
                                  HuggingFaceInstructEmbeddings)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from PIL import Image
from sentence_transformers import SentenceTransformer, models
from langChainInterface import LangChainInterface
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy as np
import faiss
import re

def filter_text(input_text):
    # Regular expression to match Thai and English characters and numbers
    pattern = re.compile(r'[0-9A-Za-zก-๙\s]')
    
    # Use findall to extract all matching characters
    filtered_characters = re.findall(pattern, input_text)
    
    # Join the characters back into a string
    filtered_text = ''.join(filtered_characters)

    return filtered_text

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
    max_new_tokens= st.number_input('max_new_tokens',1,1024,value=300)
    min_new_tokens= st.number_input('min_new_tokens',0,value=15)
    repetition_penalty = st.number_input('repetition_penalty',1,2,value=2)
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

def vector_search(query, model, index, num_results=10):
    """
    Transforms query to vector using a pretrained model and finds similar vectors using FAISS.
    
    Args:
        query (str): User query that should be more than a sentence long.
        model: Model used for generating embeddings.
        index (faiss.IndexIDMap): FAISS index.
        num_results (int): Number of results to return.
    
    Returns:
        D (:obj:`numpy.array` of `float`): Distance between results and query.
        I (:obj:`numpy.array` of `int`): Document IDs of the results.
    """
    vector = model.encode([query])
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I


def get_model(model_name='airesearch/wangchanberta-base-att-spm-uncased', max_seq_length=768, condition=True):
    if condition:
        # model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        # model_name = "hkunlp/instructor-large"
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),pooling_mode='cls') # We use a [CLS] token as representation
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def generate_prompt(question, context, model_type="llama-2"):
    if model_type =="llama-2":
        output = f"""[INST] <<SYS>>
You are a helpful, respectful Thai assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

You will receive HR Policy on user queries HR POLICY DETAILS, and QUESTION from user in the ''' below. Answer the question in Thai.
'''
HR POLICY DETAILS:
{context}

QUESTION: {question}
'''
Answer the QUESTION use details about HR Policy from HR POLICY DETAILS, explain your reasonings if the question is not related to REFERENCE please Answer
“I don’t know the answer, it is not part of the provided HR Policy”.
<</SYS>>

คำถาม: {question} [/INST]
คำตอบของถามเป็นภาษาไทย:
    """
    else:
        return "Only llama-2 format at the moment, fill here to add other prompt templates for other model types."
    return output

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

def format_documents(docs):
    formatted_docs = []
    for i, doc in enumerate(docs):
        formatted_doc = {
            "reference_number": i + 1,  # Assuming you want to start numbering from 1
            "page_content": doc.page_content,
            "source": "page " + str(doc.metadata.get('page', 'Unknown'))
        }
        formatted_docs.append(formatted_doc)
    return formatted_docs

def format_docs_for_display(docs):
    formatted_text = ""
    for doc in docs:
        formatted_text += f"Reference Number: {doc['reference_number']}\n"
        formatted_text += f"Source: {doc['source']}\n"
        formatted_text += f"Content:\n{doc['page_content']}\n"
        formatted_text += "-" * 40 + "\n"  # Separator
    return formatted_text



# show user input
if user_question := st.text_input(
    "Ask a question about your Policy Document:"
):
    docs = read_pdf(uploaded_files)
    print("DOCS HERE", docs)
    index = read_push_embeddings()
    D, I = vector_search(user_question, embeddings_model, index, num_results=4)
    search_results = [docs[i] for i in I[0]]
    # docs = db.similarity_search(user_question)
    params = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MIN_NEW_TOKENS: 30,
        GenParams.MAX_NEW_TOKENS: 500,
        GenParams.TEMPERATURE: 0.0,
        # GenParams.TOP_K: 100,
        # GenParams.TOP_P: 1,
        GenParams.REPETITION_PENALTY: 1
    }
    model_llm = LangChainInterface(model='ibm-mistralai/mixtral-8x7b-instruct-v01-q', credentials=creds, params=params, project_id=project_id)
    formated_search = format_documents(search_results)
    response = model_llm(generate_prompt(user_question, formated_search))
    print(generate_prompt(user_question, formated_search))
    # response = chain.run(input_documents=docs, question=user_question)
    # Call the function with your formatted_docs
    formatted_text_for_display = format_docs_for_display(formated_search)

    st.text_area(label="Model Response", value=filter_text(response), height=300)
    st.text_area(label="Reference", value=formatted_text_for_display, height=300)
    st.write()
