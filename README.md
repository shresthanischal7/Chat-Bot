# Company Support Chatbot

A FastAPI chat application with a browser interface, Groq-powered responses, retrieval from a local Qdrant knowledge base, and chat history backed by PostgreSQL and Redis.

The included `about.txt` describes a fictional demonstration company and uses placeholder contact details. Replace it with approved content before using this application for a real organization.

## Requirements

- Python 3.10 or newer
- PostgreSQL and Redis services
- A Groq API key

## Setup

1. Create and activate a virtual environment, then install dependencies:

   ```sh
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set `GROQ_API_KEY`, `SECRET_KEY`, and the PostgreSQL connection values. Keep `.env` private.

3. Start PostgreSQL and Redis. Configure the database tables expected by `company_chatbot.py` (`users`, `chat_sessions`, and `messages`) before starting the app. The application does not currently create those tables automatically.

4. Start the API from the project directory:

   ```sh
   uvicorn company_chatbot:app --reload
   ```

5. Open [http://127.0.0.1:8000](http://127.0.0.1:8000). On startup, the application builds its local Qdrant index from `about.txt`; the sentence-transformer model may need to download on first run.

## Project files

- `company_chatbot.py` — FastAPI application and chat endpoints
- `index.html` — browser chat interface
- `static/` — styles and browser-side JavaScript
- `about.txt` — source text for the retrieval knowledge base
- `.env.example` — template for local configuration

Local credentials, virtual environments, Python caches, and generated Qdrant storage are excluded from Git.
