import os
import time
import urllib.parse
import logging
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

from db import (
    chat_collection,
    user_collection,
    knowledge_collection,
    escalation_collection
)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# ENV + CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY            = os.getenv("GEMINI_API_KEY")
STORE_DISPLAY_NAME = os.getenv("STORE_DISPLAY_NAME", "incubation_portal_base_v2")
FILE_PATH          = os.path.join(BASE_DIR, "input/SPPU_RPF_Qs&As.pdf")
MODEL_ID           = "gemini-2.5-flash"
SUPPORT_EMAIL      = os.getenv("SUPPORT_EMAIL", "rajashree.rpf@gmail.com")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found.\n"
        "Add it to your .env file: GEMINI_API_KEY=your_key_here"
    )

client = genai.Client(api_key=API_KEY)


# ─────────────────────────────────────────────
# FILE STORE
# ─────────────────────────────────────────────
def get_or_create_store():
    """Return existing Gemini file store or create + upload PDF."""
    for store in client.file_search_stores.list():
        if store.display_name == STORE_DISPLAY_NAME:
            logger.info(f"Reusing existing store: {store.name}")
            return store

    logger.info("Creating new file store and uploading PDF...")
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

    logger.info(f"PDF uploaded. Store: {store.name}")
    return store


# ─────────────────────────────────────────────
# GEMINI LAZY INIT
# ─────────────────────────────────────────────
_store = None
_chat  = None

def initialize_gemini():
    """Initialize Gemini store + chat session once per process lifetime."""
    global _store, _chat

    if _chat is not None:
        return _chat

    try:
        _store = get_or_create_store()

        # Persist store metadata in MongoDB
        knowledge_collection.update_one(
            {"store_name": _store.name},
            {"$set": {
                "store_name":   _store.name,
                "display_name": STORE_DISPLAY_NAME,
                "file_name":    os.path.basename(FILE_PATH),
                "file_path":    FILE_PATH,
                "model":        MODEL_ID,
                "source_type":  "pdf",
                "updated_at":   datetime.utcnow()
            }},
            upsert=True
        )

        _chat = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[_store.name]
                    )
                )],
                system_instruction=(
                    # ── IDENTITY ─────────────────────────────────────────
                    "You are a knowledgeable, human-like advisor for the "
                    "SPPU-RPF Incubation Portal. You speak like a supportive "
                    "senior colleague — warm, direct, genuinely helpful. "
                    "NOT a customer-service bot.\n\n"

                    # ── CONTENT RULES ────────────────────────────────────
                    "CONTENT RULES (NON-NEGOTIABLE):\n"
                    "- Answer STRICTLY from the provided document only.\n"
                    "- NEVER use external knowledge or assumptions.\n"
                    "- Answer ONLY the first question if multiple are asked.\n"
                    "- If the exact phrase is absent, search for synonyms and "
                    "related concepts before saying not found.\n"
                    "- NEVER hedge. NEVER say 'while the document doesn't "
                    "detail' — if partial info exists, answer confidently.\n"
                    "- NEVER use filler phrases like 'it's important to note', "
                    "'please note', or 'it should be mentioned'.\n\n"

                    # ── REAL EMPATHY ─────────────────────────────────────
                    "REAL EMPATHY — MOST IMPORTANT RULE:\n"
                    "Before answering, think: WHO is this person and what are "
                    "they actually worried about right now?\n\n"
                    "User mindsets by topic:\n"
                    "- Incubation  -> nervous founder, unsure if they qualify\n"
                    "- Funding     -> anxious about survival, is there real support?\n"
                    "- Co-location -> tired of working alone, needs real workspace\n"
                    "- Research    -> academic trying to make work matter\n"
                    "- Application -> decided to apply, needs a clear path\n"
                    "- Mentorship  -> stuck founder needs someone who has been there\n"
                    "- Consultancy -> professional seeking structured expert support\n"
                    "- General     -> first-time explorer, unsure what is possible\n\n"

                    "EMPATHY RULES:\n"
                    "1. The empathy opener MUST be prefixed with '> ' "
                    "(blockquote marker). This is mandatory.\n"
                    "   Example: > Taking that first step into incubation can "
                    "feel overwhelming — here is what RPF offers.\n"
                    "2. BANNED openers — NEVER use: 'Great question!', "
                    "'Happy to help!', 'Absolutely!', 'Of course!', 'Sure!', "
                    "'Certainly!', 'It is completely normal to feel', "
                    "'It is great that you are'.\n"
                    "3. The opener must reflect the user's specific situation "
                    "— personal and specific, never generic.\n"
                    "4. Good empathy examples (adapt, do NOT copy exactly):\n"
                    "   > Taking that first step into incubation can feel "
                    "overwhelming — here is what RPF offers to make it easier.\n"
                    "   > Early-stage funding anxiety is real — here is how "
                    "RPF addresses that directly.\n"
                    "   > Having a real workspace changes everything — here is "
                    "what co-location at RPF looks like.\n"
                    "   > Turning research into real-world impact is exactly "
                    "what RPF is designed to support.\n"
                    "5. End EVERY response with ONE natural follow-up question "
                    "prefixed with '>> ' marker.\n"
                    "   Example: >> Are you currently at the idea stage, or do "
                    "you have a working prototype?\n"
                    "   Must feel like a curious colleague, NOT a script.\n\n"

                    # ── BULLET RULES ─────────────────────────────────────
                    "BULLET RULES (CRITICAL):\n"
                    "- Each bullet must be ONE complete sentence, ONE idea.\n"
                    "- NEVER start a bullet with a bold label like "
                    "'**Purpose:**' — write the full sentence instead.\n"
                    "- WRONG: '• **Purpose:** It was established to...'\n"
                    "- RIGHT:  '• It was established to promote applied "
                    "research and entrepreneurship.'\n"
                    "- Keep bullets short and scannable — max 2 lines each.\n"
                    "- NO bold text anywhere inside bullets.\n\n"

                    # ── FORMATTING ───────────────────────────────────────
                    "FORMATTING RULES:\n"
                    "- Line 1: ### Title (specific, not generic)\n"
                    "- Line 2: > [empathy opener — 1 sentence, personal]\n"
                    "- Line 3-4: Short plain description (1-2 lines, direct)\n"
                    "- Sections with clean bullet points\n"
                    "- NO long paragraphs\n"
                    "- Last line: >> [natural follow-up question]\n\n"

                    "MANDATORY OUTPUT STRUCTURE:\n"
                    "### [Specific Title]\n"
                    "> [Empathy opener — 1 sentence]\n"
                    "[Short description — 1-2 lines]\n\n"
                    "Key Points:\n"
                    "• [Complete sentence, one idea]\n"
                    "• [Complete sentence, one idea]\n\n"
                    "Facilities / Support / Details (only if relevant):\n"
                    "• [Complete sentence, one idea]\n"
                    "• [Complete sentence, one idea]\n\n"
                    ">> [Natural follow-up question]"
                )
            )
        )

        logger.info("✅ Gemini initialized successfully.")
        return _chat

    except Exception as e:
        logger.error(f"❌ Gemini init failed: {e}")
        raise


# ─────────────────────────────────────────────
# SCOPE CHECK
# ─────────────────────────────────────────────
RPF_KEYWORDS = [
    # Core entity
    "sppu", "rpf", "research park", "savitribai phule", "pune university",
    "incubation portal", "foundation",
    # Programs
    "incubation", "incubator", "incubate", "co-location", "colocation",
    "co location", "consultancy", "consulting", "fellowship", "pipeline",
    # Startup ecosystem
    "startup", "founder", "venture", "entrepreneur", "seed fund", "funding",
    "investment", "equity", "valuation", "grant", "capital",
    # Support
    "mentorship", "mentor", "guidance", "coach", "training", "skill",
    # Process
    "apply", "application", "eligibility", "selection", "process",
    "register", "graduation", "graduate", "fee", "onboard",
    # Facilities
    "facility", "facilities", "lab", "laboratory", "workspace", "office",
    "center", "centre", "c4i4", "samarth", "infrastructure", "building",
    # Domain
    "research", "collaboration", "academia", "industry", "msme",
    "innovation", "technology", "r&d", "development", "program",
    "programme", "service", "support", "portal", "connect", "contact",
    "reach", "join",
]

def is_rpf_related(question: str) -> bool:
    """Returns True if the question is within the RPF/incubation domain."""
    q = question.lower().strip()
    return any(keyword in q for keyword in RPF_KEYWORDS)


# ─────────────────────────────────────────────
# FOLLOW-UP DETECTION
# ─────────────────────────────────────────────
FOLLOWUP_PHRASES = [
    "can you share more", "tell me more", "more about", "explain further",
    "elaborate", "what do you mean", "give me more details", "more details",
    "more info", "can you explain", "what about", "and what about",
    "how about", "please explain", "go on", "continue", "expand on",
    "can you clarify", "clarify", "what else", "anything else about",
    "more on that", "tell me about that", "share more", "explain more",
    "give more", "tell more", "i want to know more", "more information",
    "can you elaborate", "detail", "details about that",
]

def is_followup(question: str) -> bool:
    """Returns True if the message is clearly a follow-up."""
    q = question.lower().strip()
    # Short questions ending with ? are almost always follow-ups
    if len(q.split()) <= 5 and q.endswith("?"):
        return True
    return any(phrase in q for phrase in FOLLOWUP_PHRASES)


# ─────────────────────────────────────────────
# NO-INFO DETECTION
# ─────────────────────────────────────────────
NO_INFO_PHRASES = [
    # Direct denials
    "does not contain information", "does not specify", "not specified",
    "not available in the document", "no information provided",
    "no information is available", "cannot provide information",
    "i cannot find", "not found in the document",
    # Document-level denials
    "is not mentioned in the document", "not mentioned in the documents",
    "document does not mention", "the document does not provide",
    "the document does not include", "the provided document does not",
    "not covered in the document", "does not cover",
    "the document does not state", "the document does not indicate",
    # Soft / polite denials (Gemini's usual style)
    "no specific mention", "no specific information", "no explicit mention",
    "no explicit section", "no explicit statement",
    "there is no mention", "there is no specific", "there is no explicit",
    "not explicitly mentioned", "not explicitly stated",
    "no detail is provided", "no details are provided",
    "based on the provided document, there is no", "no record of",
]

def is_no_info_response(text: str) -> bool:
    """Returns True if Gemini's response is a genuine 'not found' answer."""
    t = text.lower().strip()
    return any(phrase in t for phrase in NO_INFO_PHRASES)


# ─────────────────────────────────────────────
# TOPIC EXTRACTION  (for analytics)
# ─────────────────────────────────────────────
TOPIC_MAP = {
    "incubation":  ["incubation", "incubator", "incubate", "onboard"],
    "co-location": ["co-location", "colocation", "co location"],
    "consultancy": ["consultancy", "consulting", "advisory", "consult"],
    "funding":     ["funding", "investment", "seed fund", "capital", "grant", "equity"],
    "mentorship":  ["mentor", "mentorship", "guidance", "coach"],
    "research":    ["research", "collaboration", "r&d", "project"],
    "application": ["apply", "application", "eligibility", "selection", "process"],
    "facilities":  ["facility", "facilities", "lab", "laboratory", "infrastructure"],
    "sppu-rpf":    ["sppu", "rpf", "research park", "foundation"],
    "training":    ["training", "skill", "development", "fellowship"],
    "startup":     ["startup", "founder", "venture", "entrepreneur"],
    "connect":     ["connect", "contact", "reach", "join"],
}

def extract_topic(question: str) -> str:
    q = question.lower()
    for topic, keywords in TOPIC_MAP.items():
        if any(kw in q for kw in keywords):
            return topic
    return "general"


# ─────────────────────────────────────────────
# CONVERSATION MEMORY
# ─────────────────────────────────────────────
def get_conversation_history(session_id: str, limit: int = 4) -> str:
    """Return last N chat turns as formatted string for Gemini context."""
    messages = list(
        chat_collection.find({"session_id": session_id})
        .sort("timestamp", -1)
        .limit(limit)
    )
    history = []
    for msg in reversed(messages):
        history.append(f"User: {msg['question']}")
        history.append(f"Assistant: {msg['answer']}")
    return "\n".join(history)


def maintain_message_limit(session_id: str, max_messages: int = 10):
    """Delete oldest messages when a session exceeds the limit."""
    count = chat_collection.count_documents({"session_id": session_id})
    if count > max_messages:
        to_delete = count - max_messages
        oldest = list(
            chat_collection.find({"session_id": session_id})
            .sort("timestamp", 1)
            .limit(to_delete)
        )
        ids = [doc["_id"] for doc in oldest]
        chat_collection.delete_many({"_id": {"$in": ids}})


# ─────────────────────────────────────────────
# GREETING
# ─────────────────────────────────────────────
GREETINGS = {
    "hi", "hello", "hey", "hii", "hiii", "hiiii",
    "good morning", "good afternoon", "good evening", "good night",
    "howdy", "what's up", "whats up", "sup",
}

def is_greeting(text: str) -> bool:
    return text.lower().strip() in GREETINGS


def get_contextual_greeting(user_name: str = None) -> str:
    hour = datetime.now().hour
    if   5  <= hour < 12: g = "Good morning"
    elif 12 <= hour < 17: g = "Good afternoon"
    elif 17 <= hour < 21: g = "Good evening"
    else:                 g = "Hello"
    if user_name:
        return f"{g}, {user_name}! What would you like to know about RPF today?"
    return f"{g}! What can I help you with today?"


# ─────────────────────────────────────────────
# OUT-OF-SCOPE RESPONSE
# ─────────────────────────────────────────────
def out_of_scope_response() -> str:
    return (
        "I'm built specifically to answer questions about SPPU-RPF — "
        "incubation, co-location, funding, research programs, and more.<br><br>"
        "That topic is outside what I can help with here. "
        "Try asking something like:<br>"
        "<div style='margin-top:8px; margin-left:4px; line-height:2.2;'>"
        "• <em>How do I apply for incubation?</em><br>"
        "• <em>What facilities does RPF offer?</em><br>"
        "• <em>What is the seed funding support?</em><br>"
        "• <em>What is co-location at RPF?</em>"
        "</div>"
    )


# ─────────────────────────────────────────────
# ESCALATION
# ─────────────────────────────────────────────
def trigger_escalation(question: str, session_id: str) -> str:
    """Log escalation in DB and return an email CTA to the user."""
    user      = user_collection.find_one({"session_id": session_id})
    user_name = user.get("name") if user and user.get("name") else "there"

    subject = "Assistance Required - RPF Query"
    body = (
        f"Dear SPPU-RPF Support Team,\n\n"
        f"My name is {user_name}.\n\n"
        f"I have the following query:\n\n"
        f'"{question}"\n\n'
        f"I request guidance from the relevant expert.\n\n"
        f"Regards,\n{user_name}"
    )

    escalation_collection.insert_one({
        "session_id": session_id,
        "user_name":  user_name,
        "question":   question,
        "status":     "pending",
        "created_at": datetime.utcnow()
    })
    logger.info(f"[{session_id[:8]}] Escalation created for: {question[:60]}")

    mailto_link = (
        f"mailto:{SUPPORT_EMAIL}"
        f"?subject={urllib.parse.quote(subject)}"
        f"&body={urllib.parse.quote(body)}"
    )

    return (
        "<div class='aws-empathy'>"
        "That question needs a human expert — the document doesn't cover it, "
        "but our team will get you the right answer personally."
        "</div><br>"
        "This query needs expert guidance.<br><br>"
        "Click below — your question will be pre-filled in the email:<br><br>"
        f"<a href='{mailto_link}' class='escalation-link'>"
        f"<i class='fa-regular fa-envelope'></i> "
        f"Send to {SUPPORT_EMAIL}</a><br><br>"
        "<em style='color:#6B8E9A; font-size:13px;'>"
        "In the meantime, feel free to ask anything else about RPF."
        "</em>"
    )


# ─────────────────────────────────────────────
# MAIN BOT FUNCTION
# ─────────────────────────────────────────────
def ask_bot(question: str, session_id: str) -> str:
    """
    Main entry point.

    Flow:
      1. Lazy-init Gemini
      2. Track / create user record
      3. Name collection (waiting for name)
      4. Greeting handler
      5. Trigger name-ask after 3 substantive messages
      6. Scope check — allow follow-ups, block truly off-topic
      7. Build context prompt + first Gemini call
      8. No-info detection → retry → escalate if still missing
      9. Store chat + update analytics
    """
    try:
        chat = initialize_gemini()

        # ── USER TRACKING ────────────────────────────────────────
        user = user_collection.find_one({"session_id": session_id})

        if not user:
            user_collection.insert_one({
                "session_id":     session_id,
                "name":           None,
                "name_asked":     False,
                "first_seen":     datetime.utcnow(),
                "last_seen":      datetime.utcnow(),
                "total_messages": 0,
                "first_question": question[:200],
                "topics_asked":   []
            })
            user = user_collection.find_one({"session_id": session_id})
        else:
            user_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {"last_seen": datetime.utcnow()},
                    "$inc": {"total_messages": 1}
                }
            )

        # Count only substantive (non-greeting) messages
        non_greeting_count = chat_collection.count_documents({
            "session_id": session_id,
            "topic":      {"$ne": "greeting"}
        })

        # ── STEP 1: NAME COLLECTION ──────────────────────────────
        if user.get("name_asked") is True and not user.get("name"):
            if is_greeting(question):
                return "That sounds like a greeting! Before we continue — may I know your name?"
            name = question.strip().title()
            user_collection.update_one(
                {"session_id": session_id},
                {"$set": {
                    "name":             name,
                    "name_asked":       False,
                    "name_provided_at": datetime.utcnow()
                }}
            )
            return (
                f"Nice to meet you, {name}! "
                "I'm here to help you with everything about SPPU-RPF. "
                "What would you like to know?"
            )

        # ── STEP 2: GREETING HANDLER ─────────────────────────────
        if is_greeting(question):
            user_name = user.get("name") if user else None
            reply = get_contextual_greeting(user_name)
            chat_collection.insert_one({
                "session_id": session_id,
                "user_name":  user.get("name"),
                "question":   question,
                "answer":     reply,
                "topic":      "greeting",
                "timestamp":  datetime.utcnow()
            })
            return reply

        # ── STEP 3: TRIGGER NAME ASK after 3 messages ────────────
        if (
            not user.get("name")
            and not user.get("name_asked")
            and non_greeting_count >= 3
        ):
            user_collection.update_one(
                {"session_id": session_id},
                {"$set": {
                    "name_asked":    True,
                    "name_asked_at": datetime.utcnow()
                }}
            )
            return (
                "You have been asking some really interesting things! "
                "Before we go further — may I know your name?"
            )

        # ── STEP 4: SCOPE CHECK ──────────────────────────────────
        if not is_rpf_related(question) and not is_followup(question):
            return out_of_scope_response()

        # ── STEP 5: BUILD CONTEXT + FIRST GEMINI CALL ────────────
        conversation_context = get_conversation_history(session_id, limit=4)

        if conversation_context:
            prompt = (
                f"Previous conversation:\n{conversation_context}\n\n"
                f"Current question: {question}\n\n"
                "Answer the current question from the document. "
                "If this is a follow-up, use the conversation history for context."
            )
        else:
            prompt = question

        response = chat.send_message(prompt)

        # ── STEP 6: NO-INFO DETECTION → RETRY → ESCALATE ─────────
        if is_no_info_response(response.text):
            retry_prompt = (
                f"Search the entire document thoroughly for: '{question}'.\n"
                "Look for this concept under any related terms, synonyms, or "
                "nearby topics. If the concept exists anywhere in the document, "
                "explain it from that context. Only say 'not found' after "
                "exhausting all related terms."
            )
            retry_response = chat.send_message(retry_prompt)

            if is_no_info_response(retry_response.text):
                # Both attempts failed — escalate to human support
                return trigger_escalation(question, session_id)
            else:
                response = retry_response

        # ── STEP 7: STORE CHAT + UPDATE ANALYTICS ────────────────
        topic = extract_topic(question)

        chat_collection.insert_one({
            "session_id": session_id,
            "user_name":  user.get("name"),
            "question":   question,
            "answer":     response.text,
            "topic":      topic,
            "timestamp":  datetime.utcnow()
        })

        user_collection.update_one(
            {"session_id": session_id},
            {
                "$inc":      {"total_messages": 1},
                "$addToSet": {"topics_asked": topic}
            }
        )

        maintain_message_limit(session_id, max_messages=10)

        logger.info(f"[{session_id[:8]}] Answered: {question[:60]}")
        return response.text

    except Exception as e:
        err = str(e)
        logger.error(f"[ask_bot ERROR] {err}")

        if "ConnectError" in err or "nodename" in err:
            return "Cannot reach the Gemini API. Please check your internet connection."
        if "API key" in err or "api_key" in err.lower():
            return "Invalid API key. Please check your GEMINI_API_KEY in the .env file."
        return f"Something went wrong. Please try again. ({err[:100]})"