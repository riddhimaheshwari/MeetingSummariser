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


def transcribe_audio_with_segments(audio_path: str) -> dict:
    """
    Transcribe an audio file using Whisper, returning both the full text
    and per-segment start/end timestamps. The timestamps are required to
    align this transcript with pyannote's speaker-diarization output.
    """
    result = model.transcribe(audio_path)
    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        for seg in result["segments"]
    ]
    return {"text": result["text"], "segments": segments}
