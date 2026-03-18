import os
import subprocess
import tempfile
from pathlib import Path

from .schemas import EditPlan


def _run(cmd: list[str], label: str, *, cwd: str | None = None) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed ({label}):\n{result.stderr}")


def _cut_segments(raw_video: str, plan: EditPlan, tmp_dir: str) -> list[str]:
    """Extract each segment from the raw video as a separate clip."""
    clips = []
    for i, seg in enumerate(plan.segments):
        duration = seg.end - seg.start
        out = os.path.join(tmp_dir, f"clip_{i:03d}.mp4")
        _run([
            "ffmpeg", "-y",
            "-ss", str(seg.start),
            "-i", raw_video,
            "-t", str(duration),
            "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,"
                   "pad=1080:1920:-1:-1:color=black",
            "-c:v", "libx264", "-c:a", "aac",
            "-avoid_negative_ts", "1",
            out
        ], f"cut segment {i}")
        clips.append(out)
    return clips


def _concat_clips(clips: list[str], tmp_dir: str) -> str:
    """Concatenate clips into one video using the concat demuxer."""
    concat_list = os.path.join(tmp_dir, "concat.txt")
    with open(concat_list, "w") as f:
        for clip in clips:
            # Concat demuxer is happier with forward slashes on Windows.
            safe = Path(clip).resolve().as_posix()
            f.write(f"file '{safe}'\n")

    joined = os.path.join(tmp_dir, "joined.mp4")
    _run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c:v", "libx264", "-c:a", "aac",
        joined
    ], "concat")
    return joined


def _make_ass(plan: EditPlan, transcript_words: list[dict], tmp_dir: str) -> str:
    """
    Build an ASS subtitle file from transcript words that fall within selected
    segments, recalculating timestamps relative to the output timeline.
    """
    cap = plan.captions

    # Map from source time → output time using a running offset
    # Build a lookup: for each word, find which segment it belongs to and offset it.
    offset_map = []  # list of (src_start, src_end, offset)
    running = 0.0
    for seg in plan.segments:
        offset_map.append((seg.start, seg.end, running - seg.start))
        running += seg.end - seg.start

    def to_output_time(src_t: float):
        for src_start, src_end, offset in offset_map:
            if src_start <= src_t <= src_end:
                return src_t + offset
        return None

    # ASS style based on caption config
    style_map = {
        "simple":   ("Arial", 48, "&H00FFFFFF", "&H00000000", 0),   # white, no outline
        "bold":     ("Arial", 56, "&H00FFFFFF", "&H00000000", 1),   # white, black outline
        "animated": ("Arial", 56, "&H00FFFF00", "&H00000000", 1),   # yellow, black outline
    }
    font, size, primary, outline_color, bold = style_map.get(cap.style, style_map["bold"])

    valign_map = {"top": 8, "center": 5, "bottom": 2}
    alignment = valign_map.get(cap.position, 2)

    ass_path = os.path.join(tmp_dir, "captions.ass")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, Bold, "
                "Alignment, MarginL, MarginR, MarginV, Outline, Shadow\n")
        f.write(f"Style: Default,{font},{size},{primary},{outline_color},"
                f"{bold},{alignment},60,60,80,2,1\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Text\n")

        def fmt_time(t: float) -> str:
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = t % 60
            return f"{h}:{m:02d}:{s:05.2f}"

        # Group words into caption lines (~5 words each)
        chunk = []
        for w in transcript_words:
            out_start = to_output_time(w["start"])
            out_end   = to_output_time(w["end"])
            if out_start is None or out_end is None:
                if chunk:
                    # flush pending chunk
                    chunk = []
                continue
            chunk.append((w["word"], out_start, out_end))
            if len(chunk) >= 5:
                line_start = chunk[0][1]
                line_end   = chunk[-1][2]
                text = " ".join(c[0] for c in chunk)
                f.write(f"Dialogue: 0,{fmt_time(line_start)},{fmt_time(line_end)},Default,{text}\n")
                chunk = []

        # flush remaining words
        if chunk:
            line_start = chunk[0][1]
            line_end   = chunk[-1][2]
            text = " ".join(c[0] for c in chunk)
            f.write(f"Dialogue: 0,{fmt_time(line_start)},{fmt_time(line_end)},Default,{text}\n")

    return ass_path


def _build_vf_filters(plan: EditPlan, ass_path: str | None) -> str:
    """Compose the -vf filter chain for captions + drawtext overlays."""
    filters = []

    if plan.captions.enabled and ass_path:
        safe = ass_path.replace("\\", "/")
        filters.append(f"ass={safe}")

    for ov in plan.overlays:
        # Basic drawtext — font path works on most systems; falls back gracefully
        # FFmpeg filtergraphs use commas/quotes/backslashes as syntax. Escape conservatively.
        text = ov.text.replace("\r", " ").replace("\n", " ")
        text_escaped = (
            text.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace(":", "\\:")
            .replace(",", "\\,")
        )
        y_map = {"top": "50", "center": "(h-text_h)/2", "bottom": "h-text_h-80"}
        y = y_map.get(ov.position, "50")
        filters.append(
            f"drawtext=font='Arial':text='{text_escaped}':fontsize=64:fontcolor=white:"
            f"x=(w-text_w)/2:y={y}:"
            f"box=1:boxcolor=black@0.5:boxborderw=10:"
            f"enable='between(t,{ov.start},{ov.end})'"
        )

    return ",".join(filters) if filters else "null"


def render(
    raw_video: str,
    plan: EditPlan,
    transcript_words: list[dict],
    output_path: str,
) -> str:
    """
    Render one edit plan to an output MP4.

    Args:
        raw_video:        Path to the original unedited video.
        plan:             EditPlan from plan_edits().
        transcript_words: The 'words' list from transcribe_video().
        output_path:      Where to save the final MP4.

    Returns:
        Absolute path to the rendered file.
    """
    raw_video   = str(Path(raw_video).resolve())
    output_path = str(Path(output_path).resolve())
    os.makedirs(Path(output_path).parent, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"  [1/4] Cutting {len(plan.segments)} segment(s)...")
        clips = _cut_segments(raw_video, plan, tmp_dir)

        print(f"  [2/4] Concatenating clips...")
        joined = _concat_clips(clips, tmp_dir)

        ass_path = None
        if plan.captions.enabled:
            print(f"  [3/4] Generating captions ({plan.captions.style}, {plan.captions.position})...")
            ass_path = _make_ass(plan, transcript_words, tmp_dir)
        else:
            print(f"  [3/4] Captions disabled, skipping.")

        # Use paths relative to tmp_dir to avoid Windows drive-letter issues in -vf.
        joined_rel = os.path.basename(joined)
        ass_rel = os.path.basename(ass_path) if ass_path else None
        vf = _build_vf_filters(plan, ass_rel)

        print(f"  [4/4] Burning captions & overlays -> {Path(output_path).name}")
        _run([
            "ffmpeg", "-y",
            "-i", joined_rel,
            "-vf", vf,
            "-c:v", "libx264", "-c:a", "aac",
            output_path
        ], "burn captions/overlays", cwd=tmp_dir)

    return output_path
