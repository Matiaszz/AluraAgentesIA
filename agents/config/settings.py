
import os
from dotenv import load_dotenv


def load_google_api_key():
    load_dotenv()
    GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
    return GOOGLE_API_KEY
