from googletrans import Translator
import textwrap
import fitz
from PIL import Image

def generate_prompt(question, context, model_type="llama-2"):
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

def language_detector(text):
    translator = Translator()
    lang = translator.detect(text).lang
    return lang

def translate_to_thai(sentence: str, choice: bool) -> str:
    """
    Translate the text between English and Thai based on the 'choice' flag.
    
    Args:
        sentence (str): The text to translate.
        choice (bool): If True, translates text to Thai. If False, translates to English.

    Returns:
        str: The translated text.
    """
    translator = Translator()
    try:
        if choice:
            # Translate to Thai
            translate = translator.translate(sentence, dest='th')
        else:
            # Translate to English
            translate = translator.translate(sentence, dest='en')
        return translate.text
    except Exception as e:
        # Handle translation-related issues (e.g., network error, unexpected API response)
        raise ValueError(f"Translation failed: {str(e)}") from e



# def translate_to_thai(sentence, choice):
#     url = neural_seek_url
#     headers = {
#         "accept": "application/json",
#         "apikey": neural_seek_api_key,  # Replace with your actual API key
#         "Content-Type": "application/json"
#     }
#     if choice == True:
#         target = "th"
#     else:
#         target = "en"
#     data = {
#         "text": [
#             sentence
#         ],
#         "target": target
#     }
#     response = requests.post(url, headers=headers, data=json.dumps(data))
#     return json.loads(response.text)['translations'][0]
    
def translate_large_text(text, translate_function, choice, max_length=500):
    """
    Break down large text, translate each part, and merge the results.

    :param text: str, The large body of text to translate.
    :param translate_function: function, The translation function to use.
    :param max_length: int, The maximum character length each split of text should have.
    :return: str, The translated text.
    """
    
    # Split the text into parts of maximum allowed character length.
    text_parts = textwrap.wrap(text, max_length, break_long_words=True, replace_whitespace=False)

    translated_text_parts = []

    for part in text_parts:
        # Translate each part of the text.
        translated_part = translate_function(part, choice)  # Assuming 'False' is a necessary argument in the actual function.
        translated_text_parts.append(translated_part)

    # Combine the translated parts.
    full_translated_text = ' '.join(translated_text_parts)

    return full_translated_text



def formatting_docs(translated_documents, width=100):
    split_text = translated_documents.split("ข้อมูลเมตา=")
    formatted_text = "ข้อมูลเมตา=" + split_text[1]  # "ข้อมูลเมตา=" is added to the second part
    return textwrap.fill(split_text[0], width=width) + "\n" + formatted_text

def open_pdf(pdf_path, page_num):
    # Opening the PDF file and creating a handle for it
    file_handle = fitz.open(pdf_path)
    
    # The page no. denoted by the index would be loaded
    page = file_handle[page_num]
    
    # Set the desired DPI (e.g., 200)
    zoom_x = 2.0  # horizontal zoom
    zoom_y = 2.0  # vertical zoom
    mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension
    
    # Obtaining the pixelmap of the page
    page_img = page.get_pixmap(matrix=mat)
    
    # Saving the pixelmap into a png image file
    page_img.save('PDF_page_high_res.png')
    
    # Reading the PNG image file using pillow
    img = Image.open('PDF_page_high_res.png')
    
    # Displaying the png image file using an image viewer
    img.show()