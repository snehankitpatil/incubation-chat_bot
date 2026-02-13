from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

client = MongoClient(MONGO_URI)

db = client["startup_support_bot"]

knowledge_collection = db["knowledge_sources"]
chat_collection = db["chat_history"]
user_collection = db["users"]
escalation_collection = db["escalations"]