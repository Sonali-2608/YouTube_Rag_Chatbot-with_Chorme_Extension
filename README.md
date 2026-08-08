# YouTube RAG Chatbot (Using LangChain)

This repository contains a simple YouTube Retrieval-Augmented Generation (RAG) chatbot backend and a Chrome extension UI.

## Project structure

- `backend/`
  - `main.py` - FastAPI server exposing the chatbot API
  - `rag_utils.py` - transcript extraction, embedding, retrieval, and LLM answer generation
- `chrome_extension/`
  - `manifest.json` - Chrome extension configuration
  - `sidebar.html` - extension UI page
  - `popup.js` - frontend code that sends requests to the backend
  - `background.js` - side panel opening handler
- `requirements.txt` - Python package dependencies
- `.gitignore` - ignore rules for virtual environments and temp files

## How it works

1. The Chrome extension reads the current YouTube video ID from the active tab.
2. It sends the video ID and a user question to the backend API.
3. The backend fetches the YouTube transcript, converts it to text, and builds embeddings.
4. The backend retrieves the most relevant transcript chunks for the question.
5. It passes that context to a language model and returns the answer.

## Run the backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

## Load the Chrome extension

1. Open Chrome and go to `chrome://extensions`
2. Enable `Developer mode`
3. Click `Load unpacked`
4. Select the `chrome_extension` folder

## Use the extension

- Open a YouTube video page
- Open the extension side panel
- Ask a question about the video
- The extension will display the backend answer

## Notes

- The backend must be running at `http://localhost:8000`
- The extension currently calls the `/chat` endpoint
- You may need API credentials or environment variables for the Google GenAI integration
