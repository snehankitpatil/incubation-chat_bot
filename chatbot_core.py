import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Force load .env correctly (Flask-safe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ---------------- CONFIG ----------------
API_KEY = os.getenv("GEMINI_API_KEY")
STORE_DISPLAY_NAME = "incubation_portal_base_v2"
FILE_PATH = os.path.join(BASE_DIR, "input/SPPU_RPF_Qs&As.pdf")
MODEL_ID = "gemini-2.5-flash"

if not API_KEY:
    raise ValueError("API key not found. Set GEMINI_API_KEY in .env file.")

client = genai.Client(api_key=API_KEY)

# ---------------- STORE SETUP ----------------
def get_or_create_store():
    for store in client.file_search_stores.list():
        if store.display_name == STORE_DISPLAY_NAME:
            return store

    store = client.file_search_stores.create(
        config={"display_name": STORE_DISPLAY_NAME}
    )

    operation = client.file_search_stores.upload_to_file_search_store(
        file=FILE_PATH,
        file_search_store_name=store.name,
        config={"display_name": os.path.basename(FILE_PATH)}
    )

    while not operation.done:
        time.sleep(2)
        operation = client.operations.get(operation)

    return store

# Initialize once (important)
store = get_or_create_store()

chat = client.chats.create(
    model=MODEL_ID,
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[store.name]
                )
            )
        ],
        system_instruction=(
            "You are an Incubation Portal Consultation Chatbot.\n"
            "Answer strictly and only from the provided document.\n"
            "Give ONE clear, concise answer only.\n"
            "If information is not available, say: "
            "'I don't know based on the document.'"
        )
    )
)

# ---------------- PUBLIC FUNCTION (FOR UI) ----------------
def ask_bot(question: str) -> str:
    response = chat.send_message(question)
    return response.text
