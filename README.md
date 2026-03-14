Chatbot for SPPU RPF to implement inside website
Members:
  1. Snehankit Patil
  2. Atharva Raut
  3. Vishal Gadekar

## Deployment on Vercel

### Prerequisites
- Vercel account
- MongoDB Atlas account (for database)
- Google Gemini API key

### Steps
1. Create a MongoDB Atlas cluster and get the connection string (MONGO_URI).
2. Get your Gemini API key.
3. Push this code to a GitHub repository.
4. Connect the repo to Vercel.
5. In Vercel dashboard, set environment variables:
   - GEMINI_API_KEY: your Gemini API key
   - MONGO_URI: your MongoDB Atlas connection string
   - FLASK_SECRET_KEY: a random secret key (e.g., openssl rand -hex 32)
6. Deploy.

### WordPress Integration
To embed the chatbot on your WordPress website:
1. Copy the content of `templates/index.html`.
2. Replace `BACKEND_URL` with your Vercel deployment URL (e.g., 'https://your-app.vercel.app').
3. Embed the HTML code in your WordPress page using a custom HTML block or a plugin like "Insert Headers and Footers".

The chatbot will appear as a floating widget on your site.
