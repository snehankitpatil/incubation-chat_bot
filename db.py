import os
import logging
from pathlib import Path

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load .env ─────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

logger.info(f"[db.py] .env path: {_env_path}")
logger.info(f"[db.py] .env exists: {_env_path.exists()}")

# ── Read Mongo URI ───────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI not found in .env file")

logger.info(f"[db.py] MONGO_URI loaded: {repr(MONGO_URI)}")

# ── Connect to MongoDB ───────────────────────────────────────
try:
    client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsAllowInvalidCertificates=False,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=20000
)

    client.admin.command("ping")
    logger.info("✅ MongoDB connected successfully.")

except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    raise RuntimeError(
        f"Cannot connect to MongoDB at: {MONGO_URI}\n"
        "Check your MONGO_URI in the .env file."
    )

# ── Database + Collections ───────────────────────────────────
DB_NAME = os.getenv("MONGO_DB_NAME", "startup_support_bot")

db = client[DB_NAME]

knowledge_collection = db["knowledge_sources"]
chat_collection = db["chat_history"]
user_collection = db["users"]
escalation_collection = db["escalations"]

# ── Indexes ──────────────────────────────────────────────────
try:
    chat_collection.create_index(
        [("session_id", ASCENDING), ("timestamp", ASCENDING)],
        background=True,
        name="idx_chat_session_time",
    )

    user_collection.create_index(
        [("session_id", ASCENDING)],
        unique=True,
        background=True,
        name="idx_user_session",
    )

    escalation_collection.create_index(
        [("status", ASCENDING), ("created_at", ASCENDING)],
        background=True,
        name="idx_escalation_status",
    )

    logger.info("✅ MongoDB indexes verified.")

except Exception as e:
    logger.warning(f"⚠️ Index creation warning: {e}")