import os
import streamlit as st
from dotenv import load_dotenv
from streamlit.web.server.server_util import allowlisted_origins

from utils.whisper_transcriber import transcribe_audio_with_segments
from utils.diarization import diarize_audio, build_speaker_transcript
from utils.gpt_summarizer import summarize_and_format
from utils.pdf_generator import create_mom_pdf
from utils.vector_rag import MeetingRAG
from utils.memory_handler import load_chat_history, save_chat_history
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
load_dotenv()
st.set_page_config(page_title="Meeting Summarizer", layout="wide")
st.title("📊 AI Meeting Assistant with Memory")

# Constants
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Session Management
if "meeting_transcripts" not in st.session_state:
    st.session_state.meeting_transcripts = {}

if "current_meeting" not in st.session_state:
    st.session_state.current_meeting = None

# Sidebar: Upload + Meeting Selector
st.sidebar.header("📁 Meetings")

uploaded_files = list(st.session_state.meeting_transcripts.keys())
selected_file = st.sidebar.selectbox("🔄 Choose a Meeting", uploaded_files if uploaded_files else ["None"])

if selected_file != "None":
    st.session_state.current_meeting = selected_file

audio_file = st.sidebar.file_uploader("📤 Upload New Audio", type=["mp3", "wav", "m4a"])
if audio_file:
    save_path = os.path.join(DATA_DIR, audio_file.name)
    with open(save_path, "wb") as f:
        f.write(audio_file.read())


with st.spinner("📝 Transcribing audio..."):
    whisper_result = transcribe_audio_with_segments(save_path)
    transcript = whisper_result["text"]

with st.spinner("🗣️ Identifying speakers..."):
    diarization_turns = diarize_audio(save_path)
    speaker_transcript = build_speaker_transcript(whisper_result["segments"], diarization_turns)

st.session_state.meeting_transcripts[audio_file.name] = {
    "path": save_path,
    "transcript": transcript,
    "summary": None,
    "speaker_transcript": speaker_transcript,
    "pdf_path": None
}
    st.session_state.current_meeting = audio_file.name
    selected_file = audio_file.name

# Proceed only if a meeting is selected
if selected_file and selected_file != "None":
    meeting_data = st.session_state.meeting_transcripts[selected_file]
    transcript = meeting_data["transcript"]
    print(type(transcript))
    st.subheader(f"🎧 Selected Meeting: {selected_file}")

    # Tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🗣️ Transcript", "💬 Chat", "📄 Download"])

    # Tab 1: Summary
    with tab1:
        st.header("📋 Meeting Summary and MOM")
        if not meeting_data["summary"]:
            with st.spinner("Summarizing..."):
                summary = summarize_and_format(transcript)
                meeting_data["summary"] = summary
        st.markdown(meeting_data["summary"])
        st.download_button("📥 Download Summary", meeting_data["summary"], file_name="summary.txt")

    # Tab 2: Speaker Transcript
    with tab2:
        st.header("🗣️ Transcript with Speaker Tags")

    # Tab 3: Chat with Memory
    with tab3:
        st.header("💬 Ask Questions (Persistent Memory)")
        rag = MeetingRAG()
        rag.add_document(meeting_data["speaker_transcript"])
        chat_history = load_chat_history(selected_file)

        for entry in chat_history:
            st.markdown(f"**You:** {entry['question']}")
            st.markdown(f"**Bot:** {entry['answer']}")

        question = st.text_input("Ask something:")
        if question:
            with st.spinner("🤖 Thinking..."):
                answer = rag.ask_question(question)
            chat_history.append({"question": question, "answer": answer})
            save_chat_history(selected_file, chat_history)
            st.success(f"💡 {answer}")

    # Tab 4: Download PDF
    with tab4:
        st.header("📄 Generate and Download MOM PDF")
        if not meeting_data["pdf_path"]:
            pdf_file = os.path.join(DATA_DIR, f"{selected_file}_MOM.pdf")
            create_mom_pdf(pdf_file, meeting_data["summary"])
            meeting_data["pdf_path"] = pdf_file

        with open(meeting_data["pdf_path"], "rb") as f:
            st.download_button("📥 Download PDF", data=f, file_name="Meeting_MOM.pdf")

class Transcribe(BaseModel):
    raw_text:str
app=FastAPI()
origins=[
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
@app.get("/")
async def read_root():
    return {"message": "AI Meeting Assistant Backend is running!"}

