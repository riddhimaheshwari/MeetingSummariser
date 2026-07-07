# Meeting Summariser

An AI-powered tool that turns a recorded meeting into a transcript, a speaker-attributed transcript (via real diarization), a structured summary with agenda items/decisions/action items, a downloadable PDF of the minutes, and a retrieval-augmented Q&A chat scoped to that specific meeting.

The project exists in two forms:
- **`backend/app.py`** — the original Streamlit prototype. Single-file, all four features in tabs.
- **`backend/main.py`** + **`frontend/`** — a FastAPI backend with a React frontend, built as an expansion of the Streamlit version to support a proper API layer and async request handling.

Both versions share the same underlying pipeline (Whisper → diarization → GPT-4 → FAISS/RAG).

---

## Features

- **Transcription** — OpenAI Whisper (`base` model), run locally.
- **Speaker diarization** — `pyannote.audio`'s pretrained speaker-diarization pipeline analyzes the raw audio and detects who is actually speaking; its speaker turns are aligned with Whisper's per-segment timestamps to produce a real speaker-attributed transcript.
- **Summarization** — GPT-4 generates a concise summary, agenda items, decisions made, and action items in a structured format.
- **Retrieval-augmented Q&A** — the transcript is chunked, embedded, and stored in a per-meeting FAISS index; questions are answered using retrieved context via LangChain, not the model guessing from memory.
- **PDF export** — a downloadable PDF of the generated minutes, built with ReportLab.
- **Persistent chat history** — chat turns are saved to a JSON file per meeting (`backend/db/chat_history/`), so a conversation survives a restart.

---

## Project structure

```
backend/
  app.py                 # Streamlit prototype (all 4 features in tabs)
  main.py                # FastAPI backend (async, REST API for the React frontend)
  requirements.txt
  utils/
    whisper_transcriber.py   # Whisper transcription (+ segment timestamps)
    diarization.py           # pyannote diarization + alignment with Whisper segments
    gpt_summarizer.py        # GPT-4 summarization prompt + call
    vector_rag.py            # Chunking, embeddings, FAISS, RAG chat (LangChain)
    pdf_generator.py         # ReportLab PDF generation
    memory_handler.py        # JSON-file chat history persistence
  db/chat_history/        # Saved chat history per meeting (JSON)
  transcripts/             # Uploaded audio + saved transcripts
  faiss_indexes/           # Saved FAISS vector indexes, one per meeting
  pdfs/                    # Generated PDF minutes

frontend/
  src/
    api.js                        # API client matching main.py's endpoints exactly
    App.jsx
    components/
      Sidebar.jsx                 # Upload + meeting list
      MeetingDetail.jsx           # Tab container
      SummaryTab.jsx
      SpeakerTranscriptTab.jsx
      ChatTab.jsx
```

---

## Setup

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in `backend/` (never commit this file — see Security below):

```
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

The Hugging Face token is required for diarization. Before it will work, you must accept the user agreement (free, just requires being logged in) on both of these model pages:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

**Run the Streamlit version:**
```bash
streamlit run app.py
```

**Run the FastAPI version:**
```bash
uvicorn main:app --reload --port 8000
```

### 2. Frontend (only needed for the FastAPI version)

```bash
cd frontend
npm install
npm run dev
```

Opens on `http://localhost:3000`, which is already whitelisted in `main.py`'s CORS config. The frontend expects the API at `http://localhost:8000` (see `BASE_URL` in `src/api.js`).

---

## How it works, briefly

1. Audio is uploaded and saved to disk.
2. Whisper transcribes it, returning both full text and per-segment timestamps.
3. Pyannote diarizes the same audio file independently, returning speaker turns based on voice characteristics.
4. Whisper segments are matched to the pyannote speaker turn they overlap most with, producing a real speaker-attributed transcript.
5. The transcript is sent to GPT-4 with a structured prompt to produce a summary, agenda, decisions, and action items.
6. The transcript is chunked (~1000 characters, 200-character overlap) and embedded into a FAISS index, scoped to that one meeting.
7. Questions are answered by retrieving the most relevant chunk(s) from that meeting's FAISS index and passing them to GPT-4 alongside the conversation history (LangChain `create_retrieval_chain` + `create_stuff_documents_chain`).
8. A PDF of the minutes is generated and made available for download.

---

## Known limitations / honest notes

- **Diarization accuracy depends on audio quality.** Crosstalk, background noise, or multiple people sharing a single microphone can reduce accuracy. It hasn't been evaluated against a labeled ground-truth dataset — accuracy claims should be described qualitatively (e.g. "consistently reasonable on tested recordings"), not as a measured percentage.
- **The FastAPI backend stores meeting state in memory** (`in_memory_meetings = {}`), not a database — state is lost on server restart. `memory_handler.py` (JSON persistence) exists and works, but isn't currently wired into `main.py`; it's used by the Streamlit version only.
- **This is a single FastAPI application, not microservices** — transcription, diarization, summarization, and RAG all run in one process, organized into separate modules for clarity. It would be straightforward to split into real independent services later, but it isn't one today.
- **The summary parser (`parse_summary_string` in `main.py`) is a manual string parser** keyed on emoji headers, not a structured-output/JSON-mode call — it's a bit brittle if GPT's formatting drifts.
- **`/meetings/` processes everything synchronously** before responding (transcription + diarization + summarization + embedding), so uploads can take a minute or two depending on file length. A natural next step would be returning immediately with a "processing" status and polling instead.

---

## Security

Never commit `.env` — it contains real API keys/tokens. Make sure `.env` is listed in `.gitignore` before your first commit. If a key is ever accidentally committed and pushed, rotate it immediately in the relevant provider's dashboard (OpenAI, Hugging Face) and scrub it from git history (`git filter-repo` or the BFG Repo-Cleaner) — a plain `git rm` does not remove it from history.

---

## Tech stack

| Component | Tool |
|---|---|
| Transcription | OpenAI Whisper (local) |
| Speaker diarization | pyannote.audio (pretrained) |
| Summarization | GPT-4 (OpenAI API) |
| Embeddings | OpenAI Embeddings (via LangChain) |
| Vector store | FAISS (per-meeting index, saved to disk) |
| RAG orchestration | LangChain |
| Backend | FastAPI (async) / Streamlit (prototype) |
| Frontend | React (Vite) |
| PDF generation | ReportLab |
| Chat persistence | JSON files (per meeting) |
