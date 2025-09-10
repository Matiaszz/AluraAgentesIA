# type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .config.settings import load_google_api_key

GOOGLE_API_KEY = load_google_api_key()
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)
