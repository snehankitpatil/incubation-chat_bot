import os
import logging
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

try:
    # ✅ Set a connection timeout so app fails fast if DB is down
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,  # 5 second timeout
        connectTimeoutMS=5000
    )
    # Force connection check at startup
    client.admin.command("ping")
    logger.info("MongoDB connected successfully.")

except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    logger.error(f"MongoDB connection failed: {e}")
    raise RuntimeError(
        "Cannot connect to MongoDB. "
        "Check your MONGO_URI environment variable and ensure MongoDB is running."
    )

# ─────────────────────────────────────────────
# DATABASE + COLLECTIONS
# ─────────────────────────────────────────────
db = client["startup_support_bot"]

knowledge_collection  = db["knowledge_sources"]
chat_collection       = db["chat_history"]
user_collection       = db["users"]
escalation_collection = db["escalations"]

# ─────────────────────────────────────────────
# INDEXES — created once, speed up queries
# ─────────────────────────────────────────────
try:
    # chat_history: fast lookup + sorting by session and time
    chat_collection.create_index(
        [("session_id", ASCENDING), ("timestamp", ASCENDING)],
        background=True
    )
    # users: fast session lookup
    user_collection.create_index(
        [("session_id", ASCENDING)],
        unique=True,
        background=True
    )
    # escalations: lookup by status for admin dashboard
    escalation_collection.create_index(
        [("status", ASCENDING), ("created_at", ASCENDING)],
        background=True
    )
    logger.info("MongoDB indexes ensured.")

except Exception as e:
    # Non-fatal — app still works without indexes, just slower
    logger.warning(f"Index creation warning: {e}")