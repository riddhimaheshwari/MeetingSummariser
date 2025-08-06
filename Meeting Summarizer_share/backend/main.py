# backend/main.py (FINAL BACKEND CODE FOR PHASE 1)

import os
import shutil
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import your existing utilities (these should be in backend/utils folder)
from utils.whisper_transcriber import transcribe_audio
from utils.gpt_summarizer import summarize_and_format
from utils.pdf_generator import create_mom_pdf
from utils.vector_rag import MeetingRAG

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost:3000",
"http://localhost:3001",
"http://localhost:3002",
"http://localhost:3003",
"http://localhost:3004"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- DIRECTORY DEFINITIONS ---
FAISS_DIR = "faiss_indexes"
os.makedirs(FAISS_DIR, exist_ok=True)
TRANSCRIPT_DIR = "transcripts"
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
PDF_DIR = "pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

# --- IN-MEMORY STORAGE ---
in_memory_meetings: Dict[str, Any] = {}
active_rag_instances: Dict[str, MeetingRAG] = {}

# --- Pydantic Models for Meeting Data ---
class MeetingSummary(BaseModel):
    concise_summary: str
    agenda_items: List[str]
    decisions_made: List[str]
    action_items: List[Dict[str, str]]

class ChatMessage(BaseModel):
    question: str
    answer: str

class ChatRequest(BaseModel):
    question: str

class MeetingBase(BaseModel):
    meeting_id: str
    meeting_name: Optional[str] = None
    upload_date: str
    raw_transcript: str
    speaker_transcript: str
    summary: Optional[MeetingSummary] = None
    pdf_path_on_server: Optional[str] = None
    chat_history: List[ChatMessage] = []
    faiss_index_path: Optional[str] = None

# --- HELPER FUNCTION FOR SUMMARIZATION PARSING ---
def parse_summary_string(summary_text: str) -> MeetingSummary:
    summary_parts = {}
    lines = summary_text.split('\n')
    current_section = None

    for line in lines:
        line = line.strip()
        if line.startswith("📋 Summary:"):
            summary_parts["concise_summary"] = line.replace("📋 Summary:", "").strip()
            current_section = "summary"
        elif line.startswith("🗂️ Agenda Items:"):
            summary_parts["agenda_items"] = []
            current_section = "agenda"
        elif line.startswith("✅ Decisions Made:"):
            summary_parts["decisions_made"] = []
            current_section = "decisions"
        elif line.startswith("📌 Action Items:"):
            summary_parts["action_items"] = []
            current_section = "actions"
        elif line.startswith("-") and current_section:
            item = line[1:].strip()
            if current_section == "agenda":
                summary_parts["agenda_items"].append(item)
            elif current_section == "decisions":
                summary_parts["decisions_made"].append(item)
            elif current_section == "actions":
                parts = item.split(":", 1)
                if len(parts) == 2:
                    person = parts[0].strip()
                    task_date = parts[1].strip()
                    summary_parts["action_items"].append({"responsible_person": person, "task_date": task_date})
                else:
                    summary_parts["action_items"].append({"responsible_person": "Unknown", "task_date": item})

    return MeetingSummary(
        concise_summary=summary_parts.get("concise_summary", "No summary generated."),
        agenda_items=summary_parts.get("agenda_items", []),
        decisions_made=summary_parts.get("decisions_made", []),
        action_items=summary_parts.get("action_items", [])
    )


# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "AI Meeting Assistant Backend is running!"}

@app.post("/meetings/")
async def upload_meeting(audio_file: UploadFile = File(...)):
    meeting_id = str(uuid.uuid4())
    audio_save_path = os.path.join(TRANSCRIPT_DIR, f"{meeting_id}_{audio_file.filename}")
    try:
        with open(audio_save_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        transcript = transcribe_audio(audio_save_path)
        speaker_transcript = "\n".join([f"Speaker {i % 2 + 1}: {line.strip()}" for i, line in enumerate(transcript.split(". "))])
        summary_raw_text = summarize_and_format(transcript)
        summary_parsed_object = parse_summary_string(summary_raw_text)
        pdf_filename = f"{meeting_id}_MOM.pdf"
        pdf_save_path = os.path.join(PDF_DIR, pdf_filename)
        create_mom_pdf(pdf_save_path, summary_raw_text)
        faiss_index_file = os.path.join(FAISS_DIR, f"{meeting_id}.faiss")
        current_rag_instance = MeetingRAG(meeting_id)
        current_rag_instance.add_document(speaker_transcript)
        current_rag_instance.vectorstore.save_local(faiss_index_file)
        active_rag_instances[meeting_id] = current_rag_instance
        meeting_data_object = MeetingBase(
            meeting_id=meeting_id,
            meeting_name=audio_file.filename,
            upload_date=str(datetime.now()),
            raw_transcript=transcript,
            speaker_transcript=speaker_transcript,
            summary=summary_parsed_object,
            pdf_path_on_server=pdf_save_path,
            faiss_index_path=faiss_index_file,
            chat_history=[]
        )
        in_memory_meetings[meeting_id] = meeting_data_object
        return {"message": "Meeting processed successfully", "meeting_id": meeting_id, "meeting_name": audio_file.filename}
    except Exception as e:
        print(f"Error processing meeting: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process meeting: {str(e)}")

@app.get("/meetings/", response_model=List[MeetingBase])
async def get_all_meetings():
    return list(in_memory_meetings.values())

@app.get("/meetings/{meeting_id}", response_model=MeetingBase)
async def get_meeting_details(meeting_id: str):
    meeting = in_memory_meetings.get(meeting_id)
    if meeting:
        return meeting
    raise HTTPException(status_code=404, detail="Meeting not found")

@app.post("/meetings/{meeting_id}/chat")
async def chat_with_meeting(meeting_id: str, chat_request: ChatRequest):
    meeting_doc = in_memory_meetings.get(meeting_id)
    if not meeting_doc:
        raise HTTPException(status_code=404, detail="Meeting not found for chat.")
    rag_instance = active_rag_instances.get(meeting_id)
    if not rag_instance:
        rag_instance = MeetingRAG(meeting_id)
        faiss_path = meeting_doc.faiss_index_path
        if not faiss_path or not os.path.exists(faiss_path):
            raise HTTPException(status_code=500, detail="FAISS index file not found for this meeting.")
        rag_instance.add_document(meeting_doc.speaker_transcript, faiss_path=faiss_path)
        active_rag_instances[meeting_id] = rag_instance
    db_chat_history = meeting_doc.chat_history
    rag_instance.memory.chat_memory.messages = rag_instance._convert_history_to_lc_messages(db_chat_history)
    answer = await rag_instance.ask_question(chat_request.question)
    new_chat_entry = ChatMessage(question=chat_request.question, answer=answer)
    in_memory_meetings[meeting_id].chat_history.append(new_chat_entry)
    return {"question": chat_request.question, "answer": answer}


# backend/main.py (Continue after your chat_with_meeting endpoint)

@app.get("/meetings/{meeting_id}/pdf")
async def get_meeting_pdf(meeting_id: str):
    """Serves the generated PDF for a specific meeting."""
    # 1. Check if the meeting exists in our in-memory storage
    meeting = in_memory_meetings.get(meeting_id)
    if not meeting or not meeting.pdf_path_on_server:
        # If the meeting doesn't exist or has no PDF path, return an error
        raise HTTPException(status_code=404, detail="PDF not found for this meeting.")

    file_path = meeting.pdf_path_on_server

    # 2. Check if the PDF file actually exists on the server's disk
    if not os.path.exists(file_path):
        # If the path exists in our storage but the file is gone, return an error
        raise HTTPException(status_code=404, detail="PDF file not found on server.")

    # 3. Use FastAPI's FileResponse to send the file to the browser
    return FileResponse(file_path, media_type="application/pdf", filename=os.path.basename(file_path))