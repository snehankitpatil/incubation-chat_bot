import os
import logging
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# READ URI FROM ENVIRONMENT
# Local  → MONGO_URI=mongodb://localhost:27017
# Atlas  → MONGO_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/<dbname>?retryWrites=true&w=majority
# ─────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError(
        "MONGO_URI environment variable is not set.\n"
        "  Local  : MONGO_URI=mongodb://localhost:27017\n"
        "  Atlas  : MONGO_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
    )

# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────
try:
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=8000,   # wait up to 8s for Atlas cold-start
        connectTimeoutMS=8000,
        socketTimeoutMS=30000,
        tls=True if "mongodb+srv" in MONGO_URI else False,
        retryWrites=True,
    )
    # Verify connection immediately at startup
    client.admin.command("ping")
    logger.info("✅ MongoDB connected successfully.")

except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    raise RuntimeError(
        "Cannot connect to MongoDB.\n"
        "Check your MONGO_URI in the .env file and ensure the cluster is reachable."
    )

# ─────────────────────────────────────────────
# DATABASE + COLLECTIONS
# ─────────────────────────────────────────────
DB_NAME = os.getenv("MONGO_DB_NAME", "startup_support_bot")
db = client[DB_NAME]

knowledge_collection  = db["knowledge_sources"]
chat_collection       = db["chat_history"]
user_collection       = db["users"]
escalation_collection = db["escalations"]

# ─────────────────────────────────────────────
# INDEXES  (idempotent — safe to run repeatedly)
# ─────────────────────────────────────────────
try:
    # chat_history: session lookup + chronological sort
    chat_collection.create_index(
        [("session_id", ASCENDING), ("timestamp", ASCENDING)],
        background=True,
        name="idx_chat_session_time"
    )
    # users: fast single-session lookup, enforce uniqueness
    user_collection.create_index(
        [("session_id", ASCENDING)],
        unique=True,
        background=True,
        name="idx_user_session"
    )
    # escalations: admin dashboard — filter by status + date
    escalation_collection.create_index(
        [("status", ASCENDING), ("created_at", ASCENDING)],
        background=True,
        name="idx_escalation_status"
    )
    logger.info("✅ MongoDB indexes verified.")

except Exception as e:
    # Non-fatal — app still works without indexes
    logger.warning(f"⚠️  Index creation warning: {e}")