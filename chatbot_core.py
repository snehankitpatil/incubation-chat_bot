import os
import time
import urllib.parse
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
# ENV + CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY            = os.getenv("GEMINI_API_KEY")
STORE_DISPLAY_NAME = "incubation_portal_base_v2"
FILE_PATH          = os.path.join(BASE_DIR, "input/SPPU_RPF_Qs&As.pdf")
MODEL_ID           = "gemini-2.5-flash"
SUPPORT_EMAIL      = "rajashree.rpf@gmail.com"

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=API_KEY)


# ─────────────────────────────────────────────
# FILE STORE — create once, reuse always
# ─────────────────────────────────────────────
def get_or_create_store():
    """Return existing store or create and upload PDF."""
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


# ─────────────────────────────────────────────
# GEMINI LAZY INIT
# ─────────────────────────────────────────────
_store = None
_chat  = None

def initialize_gemini():
    """Initialize Gemini store + chat session once, then reuse."""
    global _store, _chat

    if _chat is not None:
        return _chat

    try:
        _store = get_or_create_store()

        knowledge_collection.update_one(
            {"store_name": _store.name},
            {"$set": {
                "store_name":   _store.name,
                "display_name": STORE_DISPLAY_NAME,
                "file_name":    os.path.basename(FILE_PATH),
                "file_path":    FILE_PATH,
                "model":        MODEL_ID,
                "source_type":  "pdf",
                "created_at":   datetime.utcnow()
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
                    # ── WHO YOU ARE ──────────────────────────────────────────
                    "You are a knowledgeable, human-like advisor for the SPPU-RPF "
                    "Incubation Portal. You speak like a supportive senior colleague "
                    "— warm, direct, and genuinely helpful. NOT a customer-service bot.\n\n"

                    # ── CONTENT RULES ─────────────────────────────────────────
                    "CONTENT RULES (NON-NEGOTIABLE):\n"
                    "- Answer STRICTLY from the provided document only.\n"
                    "- NEVER use external knowledge or assumptions.\n"
                    "- Answer ONLY the first question if multiple are asked.\n"
                    "- If the exact phrase is absent, search for synonyms and "
                    "related concepts before saying 'not found'.\n"
                    "- If the concept exists in the document under any related term, "
                    "answer from that context.\n\n"

                    # ── REAL EMPATHY ──────────────────────────────────────────
                    "REAL EMPATHY — THE MOST IMPORTANT RULE:\n"
                    "Before answering, think about WHO is asking and WHY.\n"
                    "Put yourself in their shoes — what are they worried about? "
                    "What decision are they trying to make?\n\n"
                    "User mindsets by topic:\n"
                    "- Incubation  → A founder nervous about their first step, "
                    "unsure if they're ready or eligible.\n"
                    "- Funding     → A startup anxious about survival, wondering "
                    "if there's real financial support.\n"
                    "- Co-location → Someone tired of working alone, looking for "
                    "a real workspace and a community.\n"
                    "- Research    → An academic trying to turn their work into "
                    "something that matters.\n"
                    "- Application → Someone who has decided to apply and just "
                    "needs a clear path forward.\n"
                    "- Mentorship  → A founder who feels stuck and needs guidance "
                    "from someone who has been there.\n"
                    "- General     → Someone exploring RPF for the first time, "
                    "not sure what is possible.\n\n"

                    "EMPATHY RULES:\n"
                    "1. Start with ONE short sentence that reflects what the user "
                    "is likely feeling — personal, specific, never generic.\n"
                    "2. BANNED openers — never use these: 'Great question!', "
                    "'Happy to help!', 'Absolutely!', 'Of course!', 'Sure!', "
                    "'Certainly!', 'That is a great question!'.\n"
                    "3. Good empathy examples — adapt to context, do NOT copy:\n"
                    "   - 'Taking that first step into incubation can feel "
                    "overwhelming — here is what RPF offers to make it easier.'\n"
                    "   - 'Worrying about early-stage funding is completely normal "
                    "— here is how RPF addresses that.'\n"
                    "   - 'Having a real space to work and collaborate changes "
                    "everything — here is what co-location at RPF looks like.'\n"
                    "   - 'Turning research into real-world impact is exactly what "
                    "RPF is designed to support.'\n"
                    "   - 'Figuring out the application process can feel confusing "
                    "— let me walk you through it clearly.'\n"
                    "4. End EVERY response with ONE natural follow-up question "
                    "that feels genuinely curious — like a colleague asking what "
                    "is next for you. NOT a scripted prompt.\n"
                    "   Good follow-up examples:\n"
                    "   - 'Are you currently at the idea stage, or do you have "
                    "a working prototype?'\n"
                    "   - 'Would it help to know more about the selection criteria?'\n"
                    "   - 'Is funding or mentorship more important to you right now?'\n\n"

                    # ── FORMATTING ────────────────────────────────────────────
                    "FORMATTING RULES (STRICTLY FOLLOW):\n"
                    "- Line 1: ### Title (specific, not generic)\n"
                    "- Line 2: Empathy opener (1 sentence, context-aware, no cliches)\n"
                    "- Line 3-4: Short plain-language description (1-2 lines max)\n"
                    "- Then: sections with clean bullet points\n"
                    "- One idea per bullet — short and scannable\n"
                    "- NO bold text inside bullets\n"
                    "- NO long paragraphs\n\n"

                    "MANDATORY OUTPUT STRUCTURE:\n"
                    "### [Specific Title]\n"
                    "[Empathy opener — 1 sentence]\n"
                    "[Short description — 1-2 lines]\n\n"
                    "Key Points:\n"
                    "• [Point]\n"
                    "• [Point]\n\n"
                    "Facilities / Support / Details (include only if relevant):\n"
                    "• [Point]\n"
                    "• [Point]\n\n"
                    "[Natural follow-up — 1 sentence]"
                )
            )
        )

        print("Gemini initialized successfully!")
        return _chat

    except Exception as e:
        print(f"Gemini init failed: {e}")
        raise


# ─────────────────────────────────────────────
# SCOPE CHECK — is question RPF-related?
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
    "apply", "application", "eligibility", "selection", "process", "register",
    "graduation", "graduate", "fee", "onboard",
    # Facilities
    "facility", "facilities", "lab", "laboratory", "workspace", "office",
    "center", "centre", "c4i4", "samarth", "infrastructure", "building",
    # Domain
    "research", "collaboration", "academia", "industry", "msme", "innovation",
    "technology", "r&d", "development", "program", "programme", "service",
    "support", "portal"
]

def is_rpf_related(question: str) -> bool:
    """Returns True if the question is within the RPF/incubation domain."""
    q = question.lower().strip()
    return any(keyword in q for keyword in RPF_KEYWORDS)


# ─────────────────────────────────────────────
# NO-INFO DETECTION
# ─────────────────────────────────────────────
NO_INFO_PHRASES = [
    # Direct denials
    "does not contain information",
    "does not specify",
    "not specified",
    "not available in the document",
    "no information provided",
    "no information is available",
    "cannot provide information",
    "i cannot find",
    "not found in the document",
    # Document-level denials
    "is not mentioned in the document",
    "not mentioned in the documents",
    "document does not mention",
    "the document does not provide",
    "the document does not include",
    "the provided document does not",
    "not covered in the document",
    "does not cover",
    "the document does not state",
    "the document does not indicate",
    # Soft / polite denials (Gemini's usual style)
    "no specific mention",
    "no specific information",
    "no explicit mention",
    "no explicit section",
    "no explicit statement",
    "there is no mention",
    "there is no specific",
    "there is no explicit",
    "not explicitly mentioned",
    "not explicitly stated",
    "no detail is provided",
    "no details are provided",
    "based on the provided document, there is no",
    "no record of",
]

def is_no_info_response(text: str) -> bool:
    """
    Returns True if Gemini's response is a genuine not-found answer.
    No length check — Gemini often writes long soft-refusals too.
    """
    text_lower = text.lower().strip()
    return any(phrase in text_lower for phrase in NO_INFO_PHRASES)


# ─────────────────────────────────────────────
# TOPIC EXTRACTION (for analytics)
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
}

def extract_topic(question: str) -> str:
    """Return analytics category for a question."""
    q = question.lower()
    for topic, keywords in TOPIC_MAP.items():
        if any(kw in q for kw in keywords):
            return topic
    return "general"


# ─────────────────────────────────────────────
# CONVERSATION MEMORY
# ─────────────────────────────────────────────
def get_conversation_history(session_id: str, limit: int = 4) -> str:
    """Return last N chat turns as a formatted string for Gemini context."""
    messages = chat_collection.find(
        {"session_id": session_id}
    ).sort("timestamp", -1).limit(limit)

    history = []
    for msg in reversed(list(messages)):
        history.append(f"User: {msg['question']}")
        history.append(f"Assistant: {msg['answer']}")

    return "\n".join(history)


def maintain_message_limit(session_id: str, max_messages: int = 10):
    """Delete oldest messages when session exceeds the limit."""
    count = chat_collection.count_documents({"session_id": session_id})
    if count > max_messages:
        to_delete = count - max_messages
        oldest = chat_collection.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).limit(to_delete)
        ids = [doc["_id"] for doc in oldest]
        chat_collection.delete_many({"_id": {"$in": ids}})


# ─────────────────────────────────────────────
# GREETING
# ─────────────────────────────────────────────
GREETINGS = {
    "hi", "hello", "hey", "hii", "hiii", "hiiii",
    "good morning", "good afternoon", "good evening", "good night",
    "howdy", "what's up", "whats up", "sup"
}

def is_greeting(text: str) -> bool:
    return text.lower().strip() in GREETINGS


def get_contextual_greeting(user_name: str = None) -> str:
    hour = datetime.now().hour
    if   5  <= hour < 12: time_greeting = "Good morning"
    elif 12 <= hour < 17: time_greeting = "Good afternoon"
    elif 17 <= hour < 21: time_greeting = "Good evening"
    else:                 time_greeting = "Hello"

    if user_name:
        return (
            f"{time_greeting}, {user_name}! What would you like "
            "to know about RPF today?"
        )
    return f"{time_greeting}! What can I help you with today?"


# ─────────────────────────────────────────────
# OUT-OF-SCOPE RESPONSE
# ─────────────────────────────────────────────
def out_of_scope_response() -> str:
    return (
        "I'm built specifically to answer questions about SPPU-RPF — "
        "incubation, co-location, funding, research programs, and more.<br><br>"
        "That topic is outside what I can help with here. Try something like:<br>"
        "<div style='margin-top:8px; margin-left:6px; line-height:2;'>"
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
    """Save escalation record and return an email CTA to the user."""
    user      = user_collection.find_one({"session_id": session_id})
    user_name = user.get("name") if user and user.get("name") else "there"

    subject = "Assistance Required - RPF Query"
    body = (
        f"Dear SPPU-RPF Support Team,\n\n"
        f"My name is {user_name}.\n\n"
        f"I have the following query:\n\n"
        f'"{question}"\n\n'
        f"I request guidance from the relevant expert or consultant.\n\n"
        f"Regards,\n{user_name}"
    )

    escalation_collection.insert_one({
        "session_id": session_id,
        "user_name":  user_name,
        "question":   question,
        "status":     "pending",
        "created_at": datetime.utcnow()
    })

    mailto_link = (
        f"mailto:{SUPPORT_EMAIL}"
        f"?subject={urllib.parse.quote(subject)}"
        f"&body={urllib.parse.quote(body)}"
    )

    return (
        "<div class='aws-empathy'>"
        "That is a question worth getting right — and it needs a human expert, "
        "not just a document search. Our team will personally help you."
        "</div>"
        "<br>"
        "This question needs expert guidance.<br><br>"
        "Click below — your question will be pre-filled in the email:<br><br>"
        f"<a href='{mailto_link}' "
        f"style='color:#2E86AB; font-weight:600; text-decoration:underline;'>"
        f"Send to {SUPPORT_EMAIL}</a><br><br>"
        "<em style='color:#6B8E9A;'>"
        "In the meantime, feel free to ask me anything else about RPF."
        "</em>"
    )


# ─────────────────────────────────────────────
# MAIN BOT FUNCTION
# ─────────────────────────────────────────────
def ask_bot(question: str, session_id: str) -> str:
    """
    Main entry point for the chatbot.

    Flow:
      1. Lazy-init Gemini
      2. Track / create user record
      3. Name collection flow (waiting for name)
      4. Greeting handler
      5. Trigger name-ask after 3 substantive messages
      6. Scope check — politely decline non-RPF questions
      7. Build context prompt + first Gemini call
      8. No-info detection → retry with clean prompt → escalate if still missing
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

        # Only count substantive (non-greeting) messages for name trigger
        non_greeting_count = chat_collection.count_documents({
            "session_id": session_id,
            "topic":      {"$ne": "greeting"}
        })

        # ── STEP 1: NAME COLLECTION — already asked, waiting for name ──
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

        # ── STEP 3: TRIGGER NAME ASK after 3 non-greeting messages ─
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

        # ── STEP 4: SCOPE CHECK ───────────────────────────────────
        if not is_rpf_related(question):
            return out_of_scope_response()

        # ── STEP 5: BUILD CONTEXT + FIRST GEMINI CALL ────────────
        conversation_context = get_conversation_history(session_id, limit=4)

        if conversation_context:
            prompt = (
                f"Previous conversation:\n{conversation_context}\n\n"
                f"Current question: {question}\n\n"
                "Answer the current question from the document, "
                "keeping the conversation history in mind."
            )
        else:
            prompt = question

        response = chat.send_message(prompt)

        # ── STEP 6: NO-INFO DETECTION → RETRY → ESCALATE ─────────
        if is_no_info_response(response.text):

            # Retry once with a clean, context-free, rephrased prompt
            retry_prompt = (
                f"Search the entire document thoroughly for: '{question}'.\n"
                "Look for this concept under any related terms, synonyms, or "
                "nearby topics. If the exact phrase is absent but the concept "
                "is discussed anywhere in the document, explain it from that "
                "context. Only say 'not found' after exhausting all related terms."
            )
            retry_response = chat.send_message(retry_prompt)

            if is_no_info_response(retry_response.text):
                # Both attempts failed — escalate to human support
                return trigger_escalation(question, session_id)
            else:
                # Retry found an answer — use it
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

        return response.text

    # ── ERROR HANDLING ────────────────────────────────────────────
    except Exception as e:
        err = str(e)
        if "ConnectError" in err or "nodename" in err:
            return (
                "Cannot reach the Gemini API. "
                "Please check your internet connection and try again."
            )
        if "API key" in err or "api_key" in err.lower():
            return (
                "Invalid API key. "
                "Please check your GEMINI_API_KEY in the .env file."
            )
        print(f"[ask_bot ERROR] {e}")
        return f"Something went wrong. Please try again. ({err[:100]})"
