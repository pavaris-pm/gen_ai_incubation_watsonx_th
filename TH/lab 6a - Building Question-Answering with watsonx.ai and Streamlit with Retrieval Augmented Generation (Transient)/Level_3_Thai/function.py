import numpy as np
from sentence_transformers import SentenceTransformer, models

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

def detect_language(text):
    thai_chars = set("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮะัาำิีึืุูเแโใไ็่้๊๋์")
    if any(char in thai_chars for char in text):
        return "th"
    else:
        return "en"


# def generate_prompt_th(question, context, model_type="llama-2"):
#     if model_type =="llama-2":
#         output = f"""[INST] <<SYS>>
# You are a helpful, respectful Thai assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

# You will receive HR Policy on user queries HR POLICY DETAILS, and QUESTION from user in the ''' below. Answer the question in Thai.
# '''
# HR POLICY DETAILS:
# {context}

# QUESTION: {question}
# '''
# Answer the QUESTION use details about HR Policy from HR POLICY DETAILS, explain your reasonings if the question is not related to REFERENCE please Answer
# “I don’t know the answer, it is not part of the provided HR Policy”.
# <</SYS>>

# คำถาม: {question} [/INST]
# คำตอบของถามเป็นภาษาไทย:
#     """
#     else:
#         return "Only llama-2 format at the moment, fill here to add other prompt templates for other model types."
#     return output 

def generate_prompt_th(question, context, model_type="llama-2"):
    if model_type =="llama-2":
        output = f"""[INST] <<SYS>>
คุณเป็นผู้ช่วยที่ใจดี โปรดตอบคำถามอย่างมีใจดีและประโยชน์ที่สุดเสมอ พร้อมกับรักษาความปลอดภัย คำตอบของคุณไม่ควรมีเนื้อหาที่เป็นอันตราย ไม่ธรรมดา แบ่งแยกทางเชื้อชาติ ลำเอียงทางเพศ มีพิษ อันตราย หรือผิดกฎหมาย โปรดให้แน่ใจว่าคำตอบของคุณไม่มีอคติทางสังคมและเป็นบวกในธรรมชาติ ถ้าคำถามไม่มีเหตุผล หรือไม่สอดคล้องกับความเป็นจริง โปรดอธิบายเหตุผลแทนที่จะตอบคำถามที่ไม่ถูกต้อง ถ้าคุณไม่ทราบคำตอบของคำถาม โปรดอย่าแชร์ข้อมูลที่ผิด

คุณจะได้รับนโยบายทรัพยากรบุคคลเกี่ยวกับคำถามจากผู้ใช้ "รายละเอียดนโยบายทรัพยากรบุคคล" และ"คำถาม"จากผู้ใช้ใน ''' ด้านล่าง ตอบคำถามเป็นภาษาไทย
'''
รายละเอียดนโยบายทรัพยากรบุคคล:
{context}

คำถาม: {question}
'''
ตอบคำถามโดยใช้รายละเอียดเกี่ยวกับนโยบายทรัพยากรบุคคลจาก "รายละเอียดนโยบายทรัพยากรบุคคล" อธิบายเหตุผลของคุณหากคำถามไม่เกี่ยวข้องกับข้อมูลอ้างอิง โปรดตอบว่า “ฉันไม่ทราบคำตอบ, มันไม่ใช่ส่วนหนึ่งของนโยบายทรัพยากรบุคคลที่ได้รับ”
<</SYS>>

คำถาม: {question} [/INST]
คำตอบ:
    """
    else:
        return "Only llama-2 format at the moment, fill here to add other prompt templates for other model types."
    return output 

def generate_prompt_en(question, context, model_type="llama-2"):
    if model_type =="llama-2":
        output = f"""<s>[INST] <<SYS>> \n
    You are communicating with Lendy Bank, a knowledgeable, respectful, and precise assistant. My purpose is to provide accurate answers based on the information found in the document you provided.

        Please adhere to the following guidelines:
        1. Feel free to ask questions related to the document and I'll strive to deliver precise and relevant information.
        2. If your question doesn't directly pertain to the document's content, I'll search our conversation history for a relevant response. If none is found, I'll say "Please to ask me more detail".
        3. If a question's answer is irrelevant or unclear, I'll respond with "I don't know" or request clarification.
        4. For quick yes/no inquiries related to permissions, I'll provide concise responses along with document references.
        5. To maintain context in follow-up questions, I will explicitly reference the previous question, using clear language such as "it" to ensure coherent responses.
        6. When answering questions about numbered or ordered lists, I'll use line breaks for clarity and follow any specific ordering requests.
        7. To avoid redundancy, I'll keep track of your questions and provide concise, relevant answers. Feel free to ask follow-up questions for additional information.
        8. If you wish to engage in casual conversation, simply say "Hi" and I will respond your name with a friendly greeting.

        Please ensure that your responses are clear and respectful. If you have any issues or need clarification, don't hesitate to ask.

    <</SYS>>

    documents: 
    {context}

    Input: {question}

    Ouput:
    [/INST]
    """
    else:
        return "Only llama-2 format at the moment, fill here to add other prompt templates for other model types."
    return output       
        
def format_docs_for_display(docs):
    formatted_text = ""
    for doc in docs:
        formatted_text += f"Reference Number: {doc['reference_number']}\n"
        formatted_text += f"Source: {doc['source']}\n"
        formatted_text += f"Content:\n{doc['page_content']}\n"
        formatted_text += "-" * 40 + "\n"  # Separator
    return formatted_text


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