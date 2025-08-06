import whisper

# Load Whisper model once globally
model = whisper.load_model("base")

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file using Whisper (local).
    Returns raw plain-text transcript (no speaker tags).
    """
    result = model.transcribe(audio_path)
    return result["text"]
