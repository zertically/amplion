import os
import subprocess
import tempfile
from pathlib import Path
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from env


def _probe_duration_seconds(media_path: str) -> float:
    """Return media duration using ffprobe, or 0.0 on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                media_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return 0.0
        return float(result.stdout.strip() or 0.0)
    except Exception:
        return 0.0


def extract_audio(video_path: str, output_wav: str) -> None:
    """Extract mono 16kHz WAV from a video file using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                  # no video
        "-acodec", "pcm_s16le", # uncompressed WAV
        "-ar", "16000",         # 16kHz sample rate (Whisper's sweet spot)
        "-ac", "1",             # mono
        output_wav
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{result.stderr}")


def transcribe_video(video_path: str, *, mock: bool = False) -> dict:
    """
    Transcribe a video file using OpenAI Whisper.

    Returns a dict with:
        - 'words':    list of {word, start, end} — word-level timestamps
        - 'segments': list of {text, start, end} — sentence-level segments
        - 'text':     full transcript as a single string
        - 'duration': total audio duration in seconds
    """
    video_path = str(Path(video_path).resolve())
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if mock:
        # Minimal transcript sufficient for planning + rendering tests.
        # We approximate a short duration and a few words spread across time.
        words = [
            {"word": "hello", "start": 0.20, "end": 0.60},
            {"word": "this", "start": 0.60, "end": 0.85},
            {"word": "is", "start": 0.85, "end": 1.00},
            {"word": "amplion", "start": 1.00, "end": 1.60},
            {"word": "test", "start": 1.60, "end": 2.00},
        ]
        segments = [{"text": "hello this is amplion test", "start": 0.20, "end": 2.00}]
        return {
            "words": words,
            "segments": segments,
            "text": "hello this is amplion test",
            "duration": 2.0,
        }

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = os.path.join(tmp_dir, "audio.wav")

        print(f"  [1/2] Extracting audio from {Path(video_path).name}...")
        extract_audio(video_path, wav_path)

        print(f"  [2/2] Sending to Whisper API...")
        with open(wav_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"]
            )

    # Parse word-level timestamps
    words = []
    if hasattr(response, "words") and response.words:
        for w in response.words:
            words.append({
                "word":  w.word.strip(),
                "start": round(w.start, 3),
                "end":   round(w.end, 3),
            })

    # Parse segment-level timestamps
    segments = []
    if hasattr(response, "segments") and response.segments:
        for s in response.segments:
            segments.append({
                "text":  s.text.strip(),
                "start": round(s.start, 3),
                "end":   round(s.end, 3),
            })

    # Total duration = end of last word (or last segment as fallback)
    duration = 0.0
    if words:
        duration = words[-1]["end"]
    elif segments:
        duration = segments[-1]["end"]
    else:
        # Videos with no detected speech can yield no words/segments. Fall back to ffprobe.
        duration = _probe_duration_seconds(video_path)

    return {
        "words":    words,
        "segments": segments,
        "text":     response.text.strip(),
        "duration": duration,
    }


def print_transcript(transcript: dict) -> None:
    """Pretty-print a transcript for manual inspection."""
    print(f"\n{'='*60}")
    print(f"FULL TEXT:\n{transcript['text']}")
    print(f"\nDURATION: {transcript['duration']:.1f}s")
    print(f"\nWORD-LEVEL TIMESTAMPS ({len(transcript['words'])} words):")
    for w in transcript["words"]:
        print(f"  {w['start']:6.2f}s → {w['end']:6.2f}s  |  {w['word']}")
    print(f"\nSEGMENTS ({len(transcript['segments'])} segments):")
    for s in transcript["segments"]:
        print(f"  {s['start']:6.2f}s → {s['end']:6.2f}s  |  {s['text']}")
    print("="*60)


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <video_file>")
        print("Example: python transcribe.py raw.mp4")
        sys.exit(1)

    video = sys.argv[1]
    print(f"\nTranscribing: {video}")
    transcript = transcribe_video(video)
    print_transcript(transcript)