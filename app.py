import os
import uuid
import logging
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from chatbot_core import ask_bot
from db import chat_collection

# ─────────────────────────────────────────────
# LOAD ENV
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)

# Secret key: must come from environment in production
_secret = os.getenv("FLASK_SECRET_KEY")
if not _secret:
    if os.getenv("FLASK_ENV") == "production":
        raise RuntimeError(
            "FLASK_SECRET_KEY must be set in production. "
            "Generate one with: python3 -c \"import secrets; print(secrets.token_hex(32))\""
        )
    # Development fallback — new key each restart (sessions won't persist across restarts)
    _secret = os.urandom(32)
    logger.warning("FLASK_SECRET_KEY not set — using random key (dev only).")

app.secret_key = _secret

IS_PRODUCTION = os.getenv("FLASK_ENV") == "production"

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,       # JS cannot steal the cookie
    SESSION_COOKIE_SAMESITE="Lax",      # Basic CSRF protection
    SESSION_COOKIE_SECURE=IS_PRODUCTION,# HTTPS-only cookies in production
    PERMANENT_SESSION_LIFETIME=86400,   # Sessions last 24 hours
    MAX_CONTENT_LENGTH=1 * 1024 * 1024, # Max request body: 1 MB
)


# ─────────────────────────────────────────────
# BEFORE EACH REQUEST
# ─────────────────────────────────────────────
@app.before_request
def ensure_session():
    """Create a unique session ID for every new visitor."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session.permanent = True
        logger.info(f"New session: {session['session_id']}")


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """Receive a user question and return the bot's answer."""
    try:
        data = request.get_json(silent=True)

        # Validate request
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' in request body."}), 400

        user_question = str(data["question"]).strip()

        if not user_question:
            return jsonify({"error": "Question cannot be empty."}), 400

        if len(user_question) > 1000:
            return jsonify({
                "answer": "Your question is too long. "
                          "Please keep it under 1000 characters."
            })

        session_id = session["session_id"]
        logger.info(f"[{session_id[:8]}] Q: {user_question[:80]}")

        answer = ask_bot(user_question, session_id)

        logger.info(f"[{session_id[:8]}] A: {str(answer)[:80]}")
        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"/ask error: {e}", exc_info=True)
        return jsonify({
            "answer": "Something went wrong on our end. Please try again."
        }), 500


@app.route("/reset", methods=["POST"])
def reset_chat():
    """Clear the current session's chat history (keep user name)."""
    try:
        session_id = session.get("session_id")
        if session_id:
            result = chat_collection.delete_many({"session_id": session_id})
            logger.info(
                f"[{session_id[:8]}] Chat cleared: "
                f"{result.deleted_count} messages removed."
            )
        return jsonify({"message": "Chat cleared."})

    except Exception as e:
        logger.error(f"/reset error: {e}", exc_info=True)
        return jsonify({"error": "Could not clear chat."}), 500


@app.route("/health")
def health():
    """Health check endpoint for deployment / uptime monitoring."""
    return jsonify({"status": "ok"}), 200


# ─────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request."}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405

@app.errorhandler(413)
def request_too_large(e):
    return jsonify({"error": "Request body too large."}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Unhandled 500: {e}")
    return jsonify({"error": "Internal server error."}), 500


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    logger.info(
        f"Starting Flask on port {port} "
        f"({'production' if IS_PRODUCTION else 'development'} mode)"
    )
    app.run(
        host="0.0.0.0",
        port=port,
        debug=not IS_PRODUCTION   # NEVER run debug=True in production
    )