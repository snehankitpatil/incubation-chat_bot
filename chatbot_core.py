import os
import time
from db import chat_collection, user_collection
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
from db import escalation_collection 
import urllib.parse
    

# 🔹 MongoDB (ADDED)
from db import knowledge_collection, chat_collection

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

# 🔹 Lazy initialization - only create when needed
_store = None
_chat = None

def initialize_gemini():
    """Initialize Gemini store and chat (lazy loading)"""
    global _store, _chat
    
    if _chat is not None:
        return _chat
    
    try:
        # Initialize store
        _store = get_or_create_store()
        
        # Save PDF metadata to MongoDB
        knowledge_collection.update_one(
            {"store_name": _store.name},
            {
                "$set": {
                    "store_name": _store.name,
                    "display_name": STORE_DISPLAY_NAME,
                    "file_name": os.path.basename(FILE_PATH),
                    "file_path": FILE_PATH,
                    "model": MODEL_ID,
                    "source_type": "pdf",
                    "created_at": datetime.utcnow()
                }
            },
            upsert=True
        )
        
        # Create chat instance
        _chat = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[_store.name]
                        )
                    )
                ],
                system_instruction=(
                    "You are an Incubation Portal Consultation Chatbot.\n"
                    "Answer strictly and only from the provided document.\n"
                    "Do NOT add external knowledge.\n"
                    "Do NOT merge multiple questions.\n"
                    "Answer ONLY the first question if multiple are asked.\n\n"

                    "VERY IMPORTANT FORMATTING RULES (FOLLOW STRICTLY):\n"
                    "- Use a CLEAR TITLE as the first line.\n"
                    "- Give a ONE-LINE definition below the title.\n"
                    "- Then provide sections with bullet points.\n"
                    "- Use short, crisp bullets (AWS documentation style).\n"
                    "- Do NOT write long paragraphs.\n"
                    "- Each bullet must contain only ONE idea.\n\n"

                    "MANDATORY ANSWER STRUCTURE:\n"
                    "Title\n"
                    "Short description (1–2 lines)\n\n"
                    "Key Points:\n"
                    "•Bullet point\n"
                    "•Bullet point\n\n"
                    "Facilities / Support / Details (if applicable):\n"
                    "•Bullet point\n"
                    "•Bullet point\n\n"
                )
            )
        )
        
        print("✅ Gemini initialized successfully!")
        return _chat
        
    except Exception as e:
        print(f"❌ Error initializing Gemini: {e}")
        print("⚠️  Please check:")
        print("   1. Your internet connection")
        print("   2. GEMINI_API_KEY in .env file")
        print("   3. Gemini API service status")
        raise

# ---------------- CONVERSATION MEMORY ----------------
def get_conversation_history(session_id: str, limit: int = 10):
    """Get last N messages for context"""
    messages = chat_collection.find(
        {"session_id": session_id}
    ).sort("timestamp", -1).limit(limit)
    
    # Reverse to get chronological order
    history = []
    for msg in reversed(list(messages)):
        history.append(f"User: {msg['question']}")
        history.append(f"Assistant: {msg['answer']}")
    
    return "\n".join(history)

def maintain_message_limit(session_id: str, max_messages: int = 10):
    """Keep only the last N messages per session"""
    count = chat_collection.count_documents({"session_id": session_id})
    
    if count > max_messages:
        # Get oldest messages to delete
        to_delete = count - max_messages
        oldest = chat_collection.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).limit(to_delete)
        
        ids_to_delete = [doc["_id"] for doc in oldest]
        chat_collection.delete_many({"_id": {"$in": ids_to_delete}})

def get_contextual_greeting(user_name: str = None) -> str:
    """Generate time-appropriate greeting"""
    hour = datetime.now().hour
    
    if 5 <= hour < 12:
        greeting = "Good morning"
    elif 12 <= hour < 17:
        greeting = "Good afternoon"
    elif 17 <= hour < 21:
        greeting = "Good evening"
    else:
        greeting = "Hello"
    
    if user_name:
        return f"{greeting}, {user_name}! 👋"
    return f"{greeting}! 👋"

# ---------------- TOPIC EXTRACTION ----------------
def extract_topic(question: str) -> str:
    """Extract topic/category from question for analytics"""
    question_lower = question.lower()
    
    keywords = {
        "incubation": ["incubation", "incubator", "startup facility"],
        "consultancy": ["consultancy", "consulting", "advisory"],
        "co-location": ["co-location", "colocation", "space", "office"],
        "research": ["research", "collaboration", "project"],
        "sppu-rpf": ["sppu", "rpf", "research park", "foundation"],
        "pipeline": ["pipeline", "building"],
        "funding": ["funding", "investment", "money", "capital"],
        "mentorship": ["mentor", "mentorship", "guidance"],
        "facilities": ["facility", "facilities", "infrastructure"]
    }
    
    for topic, terms in keywords.items():
        if any(term in question_lower for term in terms):
            return topic
    
    return "general"

# ---------------- PUBLIC FUNCTION (FOR UI) ----------------
def ask_bot(question: str, session_id: str) -> str:
    
    try:
        # Initialize Gemini on first use
        chat = initialize_gemini()
        
        # 🔹 TRACK USER FROM FIRST MESSAGE
        user = user_collection.find_one({"session_id": session_id})
        
        if not user:
            # Create new user record on first interaction
            user_collection.insert_one({
                "session_id": session_id,
                "name": None,
                "name_asked": False,
                "first_seen": datetime.utcnow(),
                "last_seen": datetime.utcnow(),
                "total_messages": 0,
                "first_question": question[:200],  # Store user's first question
                "topics_asked": []  # Will track unique topics
            })
            user = user_collection.find_one({"session_id": session_id})
        else:
            # Update last seen time
            user_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {"last_seen": datetime.utcnow()},
                    "$inc": {"total_messages": 1}
                }
            )
        
        # Count ONLY non-greeting messages for name-asking trigger
        non_greeting_count = chat_collection.count_documents({
            "session_id": session_id,
            "topic": {"$ne": "greeting"}  # Exclude greetings
        })

        # 1️⃣ Waiting for name (user already asked for name)
        if user.get("name_asked") is True and not user.get("name"):
            if is_greeting(question):
                return "😊 That sounds like a greeting. May I know your name?"

            # Save the name
            user_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "name": question.strip().title(),
                        "name_asked": False,
                        "name_provided_at": datetime.utcnow()
                    }
                }
            )
            return f"Nice to meet you, {question.strip().title()} 👋 How can I help you today?"

        # 2️⃣ Handle greetings (but don't count them)
        if is_greeting(question):
            user_name = user.get("name") if user else None
            # Store greeting with special topic marker
            chat_collection.insert_one({
                "session_id": session_id,
                "user_name": user.get("name"),
                "question": question,
                "answer": get_contextual_greeting(user_name),
                "topic": "greeting",  # Special marker
                "timestamp": datetime.utcnow()
            })
            return get_contextual_greeting(user_name)

        # 3️⃣ Ask for name AFTER 3 non-greeting messages (ONLY ONCE)
        if not user.get("name") and not user.get("name_asked") and non_greeting_count >= 3:
            user_collection.update_one(
                {"session_id": session_id},
                {"$set": {"name_asked": True, "name_asked_at": datetime.utcnow()}}
            )
            return "Before we continue, may I know your name? 🙂"

        # 4️⃣ Get conversation history for context
        conversation_context = get_conversation_history(session_id, limit=10)
        
        # 5️⃣ Build prompt with context
        if conversation_context:
            contextualized_question = f"""Previous conversation:
{conversation_context}

Current question: {question}

Answer the current question considering the conversation history."""
        else:
            contextualized_question = question

        # 6️⃣ GEMINI RESPONSE with context
        response = chat.send_message(contextualized_question)

        # 🔎 Detect if document does not contain answer
        text_lower = response.text.lower()

        no_info_phrases = [
            "does not contain information",
            "does not specify",
            "not specified",
            "not available in the document",
            "no information provided"
        ]

        if any(phrase in text_lower for phrase in no_info_phrases):
            return trigger_escalation(question, session_id)



        # 7️⃣ Extract topic/category from question (for analytics)
        topic = extract_topic(question)

        # 8️⃣ STORE CHAT with metadata
        chat_collection.insert_one({
            "session_id": session_id,
            "user_name": user.get("name"),  # Track who asked (if name provided)
            "question": question,
            "answer": response.text,
            "topic": topic,  # Categorize the question
            "timestamp": datetime.utcnow()
        })
        
        # 9️⃣ Update user's analytics
        user_collection.update_one(
            {"session_id": session_id},
            {
                "$inc": {"total_messages": 1},
                "$addToSet": {"topics_asked": topic}  # Track unique topics asked
            }
        )

        # 🔟 Maintain 10-message limit
        maintain_message_limit(session_id, max_messages=10)

        return response.text
        
    except Exception as e:
        error_msg = str(e)
        
        # Specific error messages
        if "ConnectError" in error_msg or "nodename" in error_msg:
            return "❌ Cannot connect to Gemini API. Please check your internet connection and try again."
        elif "API key" in error_msg:
            return "❌ Invalid API key. Please check your GEMINI_API_KEY in the .env file."
        else:
            print(f"Error in ask_bot: {e}")
            return f"❌ An error occurred: {error_msg[:100]}"

def trigger_escalation(question: str, session_id: str):

    user = user_collection.find_one({"session_id": session_id})
    user_name = user.get("name") if user and user.get("name") else "User"

    subject = "Assistance Required – RPF Query"

    body = f"""Dear SPPU-RPF Support Team,

My name is {user_name}.

I have the following query:

"{question}"

I request guidance from the relevant stream expert or consultant.

Regards,
{user_name}
"""

    # Save escalation
    escalation_collection.insert_one({
        "session_id": session_id,
        "user_name": user_name,
        "question": question,
        "status": "pending",
        "created_at": datetime.utcnow()
    })

    # Encode for URL
    encoded_subject = urllib.parse.quote(subject)
    encoded_body = urllib.parse.quote(body)

    mailto_link = f"mailto:rajashree.rpf@gmail.com?subject={encoded_subject}&body={encoded_body}"

    return f"""
⚠️ This query requires expert assistance.<br><br>

You can directly contact our support team:<br><br>

📧 <a href="{mailto_link}" style="color:#4da6ff; font-weight:bold;">
rajashree.rpf@gmail.com
</a><br><br>

📞 Contact No: <b>+91 98765 43210</b>
"""



def is_greeting(text: str) -> bool:
    greetings = [
        "hi", "hello", "hey", "hii", "hiii",
        "good morning", "good afternoon", "good evening"
    ]
    return text.lower().strip() in greetings
