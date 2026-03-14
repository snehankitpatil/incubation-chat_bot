import os
import uuid
import logging
from flask import Flask, render_template, request, jsonify, session
from chatbot_core import ask_bot
from db import chat_collection

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# APP CONFIG
# ─────────────────────────────────────────────
app = Flask(__name__)

# ✅ Production: read secret key from environment variable
# Never hardcode secrets in production code
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(32))

# ✅ Secure session cookie settings for production
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,   # JS cannot access cookie
    SESSION_COOKIE_SAMESITE="Lax",  # CSRF protection
    SESSION_COOKIE_SECURE=os.getenv("FLASK_ENV") == "production",  # HTTPS only in prod
    PERMANENT_SESSION_LIFETIME=86400  # 24 hours
)


# ─────────────────────────────────────────────
# BEFORE EACH REQUEST — ensure session exists
# ─────────────────────────────────────────────
@app.before_request
def set_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session.permanent = True
        logger.info(f"New session created: {session['session_id']}")


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """Receive question, return bot answer."""
    try:
        data = request.get_json(silent=True)

        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' field."}), 400

        user_question = data["question"].strip()

        if not user_question:
            return jsonify({"error": "Question cannot be empty."}), 400

        # ✅ Limit question length to prevent abuse
        if len(user_question) > 1000:
            return jsonify({
                "answer": "Your question is too long. Please keep it under 1000 characters."
            })

        session_id = session["session_id"]
        logger.info(f"[{session_id}] Q: {user_question[:80]}")

        answer = ask_bot(user_question, session_id)

        logger.info(f"[{session_id}] A: {str(answer)[:80]}")
        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"Error in /ask: {e}", exc_info=True)
        return jsonify({
            "answer": "Something went wrong on our end. Please try again."
        }), 500


@app.route("/reset", methods=["POST"])
def reset_chat():
    """Clear this session's chat history (keep user name)."""
    try:
        session_id = session.get("session_id")
        if session_id:
            deleted = chat_collection.delete_many({"session_id": session_id})
            logger.info(f"[{session_id}] Chat cleared: {deleted.deleted_count} messages removed.")

        return jsonify({"message": "Chat cleared successfully."})

    except Exception as e:
        logger.error(f"Error in /reset: {e}", exc_info=True)
        return jsonify({"error": "Could not clear chat."}), 500


# ─────────────────────────────────────────────
# HEALTH CHECK — for deployment / uptime monitoring
# ─────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# ─────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Unhandled 500 error: {e}")
    return jsonify({"error": "Internal server error."}), 500


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    is_production = os.getenv("FLASK_ENV") == "production"

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5001)),
        debug=not is_production   # debug=False in production
    )