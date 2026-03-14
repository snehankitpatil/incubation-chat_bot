from app import app

# Vercel looks for a WSGI app named `app` or `application`.
# We expose the existing Flask app instance here.

application = app
