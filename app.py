from flask import Flask, render_template, request, jsonify, session
import uuid
from chatbot_core import ask_bot
from db import chat_collection


app = Flask(__name__)
app.secret_key = "super_secret_key"  # required for sessions

@app.before_request
def set_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question")
    session_id = session["session_id"]

    answer = ask_bot(user_question, session_id)
    return jsonify({"answer": answer})

@app.route("/reset", methods=["POST"])
def reset_chat():
    session_id = session.get("session_id")

    # Clear chat history only (keep user name)
    chat_collection.delete_many({"session_id": session_id})

    return jsonify({"message": "Chat cleared! 👋 Hello! Ask me anything."})

if __name__ == "__main__":
    app.run(debug=True)
