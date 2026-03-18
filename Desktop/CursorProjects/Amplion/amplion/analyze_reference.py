import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI
from .transcribe import transcribe_video

client = OpenAI()

ANALYZE_PROMPT = """\
You are a professional short-form video editor. Analyze the provided video frames \
and transcript and return a JSON style profile describing exactly how this video \
was edited.

Extract the following — be specific and concrete, not vague:

{
  "structure": {
    "hook_duration_seconds": float,          // how long is the opening hook?
    "has_text_hook_overlay": bool,           // is there on-screen text in the hook?
    "section_count": int,                    // how many distinct sections/chapters?
    "has_cta_at_end": bool,                  // does it end with a call-to-action?
    "total_duration_seconds": float
  },
  "pacing": {
    "avg_cut_interval_seconds": float,       // average time between cuts
    "silence_removed": bool,                 // is dead air cut out?
    "energy": "low" | "medium" | "high"     // overall energy/intensity
  },
  "captions": {
    "present": bool,
    "position": "top" | "center" | "bottom" | null,
    "style": "simple" | "bold" | "animated" | null,
    "notes": string                          // any notable caption details
  },
  "visual_treatment": {
    "aspect_ratio": string,                  // e.g. "9:16", "16:9", "1:1"
    "has_zoom_punch_ins": bool,
    "has_broll": bool,
    "notes": string
  },
  "text_overlays": [
    {
      "text": string,
      "timing": "hook" | "throughout" | "end" | "other",
      "position": "top" | "center" | "bottom"
    }
  ]
}

Return ONLY valid JSON. No explanation outside the JSON.
"""


def _extract_frames(video_path: str, tmp_dir: str, fps: float = 0.5) -> list[str]:
    """Extract frames at `fps` frames/sec. Returns list of JPEG file paths."""
    pattern = os.path.join(tmp_dir, "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "3",        # JPEG quality (lower = better)
        pattern
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg frame extraction failed:\n{result.stderr}")

    frames = sorted(Path(tmp_dir).glob("frame_*.jpg"))
    return [str(f) for f in frames]


def _encode_frames(frame_paths: list[str], max_frames: int = 30) -> list[dict]:
    """Base64-encode frames and return as GPT-4o vision content blocks."""
    # If there are more frames than the cap, sample evenly
    if len(frame_paths) > max_frames:
        step = len(frame_paths) / max_frames
        frame_paths = [frame_paths[int(i * step)] for i in range(max_frames)]

    blocks = []
    for path in frame_paths:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{data}", "detail": "low"},
        })
    return blocks


def analyze_single(video_path: str, *, mock: bool = False) -> dict:
    """
    Analyze one reference video and return a style profile dict.
    """
    video_path = str(Path(video_path).resolve())
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if mock:
        # Simple default profile that exercises rendering.
        return {
            "structure": {
                "hook_duration_seconds": 1.5,
                "has_text_hook_overlay": True,
                "section_count": 2,
                "has_cta_at_end": False,
                "total_duration_seconds": 10.0,
            },
            "pacing": {
                "avg_cut_interval_seconds": 1.5,
                "silence_removed": True,
                "energy": "high",
            },
            "captions": {
                "present": True,
                "position": "bottom",
                "style": "bold",
                "notes": "High-contrast captions.",
            },
            "visual_treatment": {
                "aspect_ratio": "9:16",
                "has_zoom_punch_ins": False,
                "has_broll": False,
                "notes": "Simple talking head.",
            },
            "text_overlays": [
                {"text": "HOOK", "timing": "hook", "position": "top"},
            ],
        }

    name = Path(video_path).name
    print(f"  [1/3] Extracting frames from {name}...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_paths = _extract_frames(video_path, tmp_dir)
        print(f"        {len(frame_paths)} frames extracted.")

        print(f"  [2/3] Transcribing {name}...")
        transcript = transcribe_video(video_path, mock=mock)

        print(f"  [3/3] Sending to GPT-4o for style analysis...")
        image_blocks = _encode_frames(frame_paths)

    user_content = [
        {
            "type": "text",
            "text": (
                f"VIDEO TRANSCRIPT:\n{transcript['text']}\n\n"
                f"VIDEO FRAMES (one every 2 seconds):"
            ),
        },
        *image_blocks,
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": ANALYZE_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    )

    return json.loads(response.choices[0].message.content)


def _merge_profiles(profiles: list[dict]) -> dict:
    """
    Blend multiple style profiles into one by averaging numbers
    and taking the most common values for categorical fields.
    """
    if len(profiles) == 1:
        return profiles[0]

    from collections import Counter

    def avg(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else None

    def majority(vals):
        vals = [v for v in vals if v is not None]
        return Counter(vals).most_common(1)[0][0] if vals else None

    merged = {
        "structure": {
            "hook_duration_seconds":  avg([p["structure"]["hook_duration_seconds"] for p in profiles]),
            "has_text_hook_overlay":  majority([p["structure"]["has_text_hook_overlay"] for p in profiles]),
            "section_count":          avg([p["structure"]["section_count"] for p in profiles]),
            "has_cta_at_end":         majority([p["structure"]["has_cta_at_end"] for p in profiles]),
            "total_duration_seconds": avg([p["structure"]["total_duration_seconds"] for p in profiles]),
        },
        "pacing": {
            "avg_cut_interval_seconds": avg([p["pacing"]["avg_cut_interval_seconds"] for p in profiles]),
            "silence_removed":          majority([p["pacing"]["silence_removed"] for p in profiles]),
            "energy":                   majority([p["pacing"]["energy"] for p in profiles]),
        },
        "captions": {
            "present":  majority([p["captions"]["present"] for p in profiles]),
            "position": majority([p["captions"]["position"] for p in profiles]),
            "style":    majority([p["captions"]["style"] for p in profiles]),
            "notes":    profiles[0]["captions"].get("notes", ""),
        },
        "visual_treatment": {
            "aspect_ratio":     majority([p["visual_treatment"]["aspect_ratio"] for p in profiles]),
            "has_zoom_punch_ins": majority([p["visual_treatment"]["has_zoom_punch_ins"] for p in profiles]),
            "has_broll":        majority([p["visual_treatment"]["has_broll"] for p in profiles]),
            "notes":            profiles[0]["visual_treatment"].get("notes", ""),
        },
        "text_overlays": profiles[0].get("text_overlays", []),  # use first as representative
    }
    return merged


def analyze_reference(video_paths: list[str], *, mock: bool = False) -> dict:
    """
    Analyze one or more reference videos and return a single blended style profile.
    """
    profiles = []
    for i, path in enumerate(video_paths, 1):
        print(f"\nAnalyzing reference {i}/{len(video_paths)}: {Path(path).name}")
        profile = analyze_single(path, mock=mock)
        profiles.append(profile)

    return _merge_profiles(profiles)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_reference.py <ref1.mp4> [ref2.mp4 ...]")
        sys.exit(1)

    profile = analyze_reference(sys.argv[1:])
    print("\nSTYLE PROFILE:")
    print(json.dumps(profile, indent=2))
