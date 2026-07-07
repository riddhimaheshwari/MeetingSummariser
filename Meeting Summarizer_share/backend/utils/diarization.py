import os
import torch
from pyannote.audio import Pipeline

_pipeline = None


def get_pipeline():
    """
    Loads the pretrained pyannote speaker-diarization pipeline once and reuses it.
    Requires a Hugging Face access token with read access, and requires you to have
    accepted the user agreement for these two gated models on huggingface.co:
      - pyannote/speaker-diarization-3.1
      - pyannote/segmentation-3.0
    """
    global _pipeline
    if _pipeline is None:
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HUGGINGFACE_TOKEN not set. Add it to your .env file. "
                "Get one at https://huggingface.co/settings/tokens, and make sure you've "
                "accepted the terms for pyannote/speaker-diarization-3.1 and "
                "pyannote/segmentation-3.0 on huggingface.co first."
            )
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        if torch.cuda.is_available():
            _pipeline.to(torch.device("cuda"))
    return _pipeline


def diarize_audio(audio_path: str) -> list[dict]:
    """
    Runs speaker diarization on an audio file.
    Returns a list of {"start": float, "end": float, "speaker": str} turns,
    e.g. {"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00"}.
    """
    pipeline = get_pipeline()
    diarization = pipeline(audio_path)

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })
    return turns


def _assign_speaker(seg_start: float, seg_end: float, turns: list[dict]) -> str:
    """Finds which diarized speaker turn overlaps most with a given Whisper segment."""
    best_speaker = "Unknown"
    best_overlap = 0.0
    for t in turns:
        overlap = min(seg_end, t["end"]) - max(seg_start, t["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = t["speaker"]
    return best_speaker


def build_speaker_transcript(whisper_segments: list[dict], diarization_turns: list[dict]) -> str:
    """
    Combines Whisper's per-segment timestamps with pyannote's speaker turns to produce
    a real speaker-attributed transcript — replaces the old alternating-label heuristic.

    whisper_segments: [{"start": float, "end": float, "text": str}, ...]  (from Whisper)
    diarization_turns: [{"start": float, "end": float, "speaker": str}, ...]  (from pyannote)
    """
    # Map raw pyannote labels (SPEAKER_00, SPEAKER_01, ...) to readable
    # "Speaker 1", "Speaker 2", ... in order of first appearance.
    label_map: dict[str, str] = {}
    next_speaker_num = 1

    lines = []
    for seg in whisper_segments:
        text = seg["text"].strip()
        if not text:
            continue
        raw_speaker = _assign_speaker(seg["start"], seg["end"], diarization_turns)
        if raw_speaker not in label_map:
            if raw_speaker == "Unknown":
                label_map[raw_speaker] = "Unknown Speaker"
            else:
                label_map[raw_speaker] = f"Speaker {next_speaker_num}"
                next_speaker_num += 1
        lines.append(f"{label_map[raw_speaker]}: {text}")

    return "\n".join(lines)
