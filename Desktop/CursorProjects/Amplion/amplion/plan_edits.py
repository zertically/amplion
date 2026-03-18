import json
from openai import OpenAI
from .schemas import EditPlan, EditPlans, Segment, CaptionConfig, TextOverlay

client = OpenAI()

PLAN_PROMPT = """\
You are a professional short-form video editor. You will be given:
1. A raw video transcript with word-level timestamps (JSON)
2. A style profile describing how the target editing style looks (JSON)
3. How many edit variants to produce (N)

Your job is to produce N distinct edit plans that cut and shape the raw transcript \
to match the target style.

Rules:
- Only use timestamps that exist in the transcript. Never invent timestamps.
- Keep segments in chronological order.
- Remove dead air, filler words, and silence — keep energy high.
- Each variant must have a meaningfully different hook, segment selection, or length.
- Segments shorter than 0.3 seconds are not allowed.

Return ONLY valid JSON matching this schema exactly:
{
  "plans": [
    {
      "segments": [{"start": float, "end": float}, ...],
      "captions": {
        "enabled": true,
        "style": "simple" | "bold" | "animated",
        "position": "top" | "center" | "bottom"
      },
      "overlays": [
        {"text": "...", "start": float, "end": float, "position": "top" | "center" | "bottom"},
        ...
      ]
    },
    ...
  ]
}
"""


def _format_transcript_for_prompt(transcript: dict) -> str:
    if transcript.get("words"):
        return "\n".join(
            f"{w['start']:.3f}-{w['end']:.3f}: {w['word']}" for w in transcript["words"]
        )
    if transcript.get("segments"):
        return "\n".join(
            f"{s['start']:.3f}-{s['end']:.3f}: {s['text']}" for s in transcript["segments"]
        )
    return ""


def _clamp_plan(plan: EditPlan, duration: float) -> EditPlan:
    """Clamp all timestamps to [0, duration] and drop invalid segments."""
    clean_segments = []
    for seg in plan.segments:
        start = max(0.0, min(seg.start, duration))
        end   = max(0.0, min(seg.end,   duration))
        if end - start >= 0.3:
            clean_segments.append(Segment(start=start, end=end))

    clean_overlays = []
    for ov in plan.overlays:
        start = max(0.0, min(ov.start, duration))
        end   = max(0.0, min(ov.end,   duration))
        if end - start > 0.0:
            clean_overlays.append(TextOverlay(
                text=ov.text, start=start, end=end, position=ov.position
            ))

    # Ensure deterministic ordering for rendering + caption time offsets.
    clean_segments = sorted(clean_segments, key=lambda s: (s.start, s.end))
    clean_overlays = sorted(clean_overlays, key=lambda o: (o.start, o.end))

    return EditPlan(
        segments=clean_segments,
        captions=plan.captions,
        overlays=clean_overlays,
    )


def plan_edits(
    transcript: dict,
    style_profile: dict,
    n_variants: int = 3,
    *,
    mock: bool = False,
) -> list[EditPlan]:
    """
    Generate N edit plans from a raw transcript + style profile.

    Args:
        transcript:    Output from transcribe_video() — must have 'words' and 'duration'.
        style_profile: Output from analyze_reference() — style JSON dict.
        n_variants:    How many distinct edit plans to generate.

    Returns:
        List of validated EditPlan objects, clamped to the video's actual duration.
    """
    duration = transcript["duration"]

    if mock:
        # Deterministic plans that are guaranteed to be in-bounds.
        # Useful for local rendering tests without API calls.
        seg1_end = min(duration, max(0.3, duration * 0.45))
        seg2_start = min(duration, seg1_end + 0.1)
        seg2_end = min(duration, max(seg2_start + 0.3, duration))

        plans = []
        for i in range(n_variants):
            segments = []
            if seg1_end - 0.0 >= 0.3:
                segments.append(Segment(start=0.0, end=seg1_end))
            if seg2_end - seg2_start >= 0.3:
                segments.append(Segment(start=seg2_start, end=seg2_end))

            captions = CaptionConfig(
                enabled=True,
                style="bold" if i % 2 == 0 else "simple",
                position="bottom",
            )
            overlays = [
                TextOverlay(text=f"Variant {i+1}", start=0.0, end=min(duration, 1.5), position="top")
            ]
            plans.append(EditPlan(segments=segments, captions=captions, overlays=overlays))

        valid = [_clamp_plan(p, duration) for p in plans if p.segments]
        if not valid:
            raise ValueError("Mock planning produced no usable segments.")
        return valid

    user_message = (
        f"RAW TRANSCRIPT (word-level timestamps):\n"
        f"{_format_transcript_for_prompt(transcript)}\n\n"
        f"STYLE PROFILE:\n"
        f"{json.dumps(style_profile, indent=2)}\n\n"
        f"Produce {n_variants} edit plan variants."
    )
    if not _format_transcript_for_prompt(transcript).strip():
        raise ValueError(
            "Transcript has no words/segments with timestamps; cannot generate edit plans."
        )

    print(f"  [plan_edits] Asking GPT-4o for {n_variants} edit plans...")
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PLAN_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    raw_json = json.loads(response.choices[0].message.content)
    plans_obj = EditPlans.model_validate(raw_json)

    clamped = [_clamp_plan(p, duration) for p in plans_obj.plans]

    # Drop plans that have no usable segments after clamping
    valid = [p for p in clamped if p.segments]
    if not valid:
        raise ValueError("GPT-4o returned no usable edit plans after timestamp validation.")

    print(f"  [plan_edits] Got {len(valid)} valid plan(s).")
    return valid


def print_plans(plans: list[EditPlan]) -> None:
    """Pretty-print edit plans for manual inspection."""
    for i, plan in enumerate(plans, 1):
        total = sum(s.end - s.start for s in plan.segments)
        print(f"\n{'='*60}")
        print(f"PLAN {i}  ({len(plan.segments)} segments, ~{total:.1f}s total)")
        print(f"  Captions: {plan.captions.enabled} | {plan.captions.style} | {plan.captions.position}")
        for seg in plan.segments:
            print(f"  {seg.start:7.3f}s -> {seg.end:7.3f}s  ({seg.end - seg.start:.2f}s)")
        if plan.overlays:
            print(f"  Overlays:")
            for ov in plan.overlays:
                print(f"    [{ov.start:.2f}s-{ov.end:.2f}s] {ov.position}: \"{ov.text}\"")
    print("="*60)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json as _json
    from .transcribe import transcribe_video

    if len(sys.argv) < 3:
        print("Usage: python plan_edits.py <raw_video> <style_profile.json> [variants]")
        print("Example: python plan_edits.py raw.mp4 style.json 3")
        sys.exit(1)

    raw_video    = sys.argv[1]
    profile_path = sys.argv[2]
    variants     = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    with open(profile_path) as f:
        style = _json.load(f)

    print(f"\nTranscribing: {raw_video}")
    transcript = transcribe_video(raw_video)

    print(f"\nGenerating {variants} edit plan(s)...")
    plans = plan_edits(transcript, style, n_variants=variants)
    print_plans(plans)
