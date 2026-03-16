"""
transcribe.py — Extract audio from a video file and transcribe it with OpenAI Whisper.

Returns word-level timestamps so downstream modules can slice the video precisely.
Requires: openai, ffmpeg on PATH, OPENAI_API_KEY env var.
"""

import os
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI


def transcribe(video_path: str) -> list[dict]:
    """
    Transcribe a video file using OpenAI Whisper.

    Extracts mono 16kHz WAV audio from the video with FFmpeg, sends it to
    the Whisper API with word-level timestamps, and returns every word with
    its start and end time in seconds.

    Args:
        video_path: Path to the input video file (any format FFmpeg supports).

    Returns:
        List of dicts: [{"word": str, "start": float, "end": float}, ...]

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If FFmpeg fails to extract audio.
        openai.OpenAIError: If the Whisper API call fails.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "audio.wav"

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(audio_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed (exit {result.returncode}):\n"
                + result.stderr.decode(errors="replace")
            )

        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )

    # audio_path is cleaned up automatically when the tempdir context exits

    return [
        {"word": w.word, "start": w.start, "end": w.end}
        for w in (response.words or [])
    ]


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python transcribe.py <video.mp4>")
        sys.exit(1)

    video = sys.argv[1]
    print(f"Transcribing: {video}")

    try:
        words = transcribe(video)
    except (FileNotFoundError, EnvironmentError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\nFound {len(words)} words:\n")
    for w in words:
        print(f"  [{w['start']:6.2f}s – {w['end']:6.2f}s]  {w['word']}")
