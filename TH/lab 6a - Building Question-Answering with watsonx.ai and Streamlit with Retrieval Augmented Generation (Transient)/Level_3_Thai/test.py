import os
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import \
    GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.metanames import \
    GenTextReturnOptMetaNames as ReturnOptions


load_dotenv()
api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)
environment = os.getenv("ENVIRONMENT", "local")
log_level = os.getenv("LOG_LEVEL", "DEBUG")
script_dir = os.path.dirname(__file__)


if api_key is None or ibm_cloud_url is None or project_id is None:
    print(
        "Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key
    }

model_params = {
    GenParams.DECODING_METHOD: 'greedy',
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 400,
    # GenParams.RANDOM_SEED: 42,
    # GenParams.TEMPERATURE: 0.7,
    GenParams.REPETITION_PENALTY: 1,
    GenParams.RETURN_OPTIONS: {ReturnOptions.GENERATED_TOKENS: True,
                                ReturnOptions.GENERATED_TOKENS: True,
                                ReturnOptions.INPUT_TOKENS: True}
}

model_llm = Model("meta-llama/llama-2-13b-chat",
                    params=model_params, credentials=creds,
                    project_id=project_id)

def fix_encoding(text):
    try:
        # Attempt to decode the text using 'ISO-8859-1' and then re-encode it in 'UTF-8'
        fixed_text = text.encode('ISO-8859-1').decode('UTF-8')
    except UnicodeDecodeError:
        # If there's a decoding error, return the original text
        fixed_text = text
    return fixed_text

# Example usage:
broken_text = "Your broken Thai text here"
fixed_text = fix_encoding(broken_text)
print(fixed_text)


# Loop through the chunks streamed back from the API call
for response in model_llm.generate_text_stream("[INST]Please answer in thai to this question: เมืองหลวงไทยคือที่ไหน แล้วมีอะไรน่ากินบ้าง[\INST] คำตอบ:"):
    wordstream = str(response)
    print("wordstream", fix_encoding(wordstream))