import argparse
import json
import os
import sys
from pathlib import Path

from .transcribe import transcribe_video
from .analyze_reference import analyze_reference
from .plan_edits import plan_edits, print_plans
from .render import render


def main():
    parser = argparse.ArgumentParser(
        description="Amplion — auto-edit a raw video to match example styles."
    )
    parser.add_argument("--plain",    required=True,       help="Path to the raw (unedited) video")
    parser.add_argument("--examples", required=True, nargs="+", help="One or more reference/example videos")
    parser.add_argument("--variants", type=int, default=3, help="Number of edit variants to produce (default: 3)")
    parser.add_argument("--output",   default="output",    help="Output directory (default: output/)")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run without API calls (generates simple plans for local testing).",
    )
    args = parser.parse_args()

    raw_video    = args.plain
    ref_videos   = args.examples
    n_variants   = args.variants
    output_dir   = args.output
    mock         = args.mock

    # Validate inputs
    for path in [raw_video] + ref_videos:
        if not os.path.exists(path):
            print(f"Error: file not found — {path}")
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    raw_name = Path(raw_video).stem

    print(f"\n{'='*60}")
    print(f"Amplion")
    print(f"  Raw video : {raw_video}")
    print(f"  References: {', '.join(ref_videos)}")
    print(f"  Variants  : {n_variants}")
    print(f"  Output dir: {output_dir}/")
    print(f"{'='*60}\n")

    # ── Job 1: Transcribe ────────────────────────────────────────────────────
    print("[ Job 1 / 4 ] Transcribing raw video...")
    transcript = transcribe_video(raw_video, mock=mock)
    print(f"  Done. {len(transcript['words'])} words, {transcript['duration']:.1f}s\n")

    # ── Job 2: Analyze references ────────────────────────────────────────────
    print("[ Job 2 / 4 ] Analyzing reference video(s)...")
    style_profile = analyze_reference(ref_videos, mock=mock)
    profile_path  = os.path.join(output_dir, f"{raw_name}_style_profile.json")
    with open(profile_path, "w") as f:
        json.dump(style_profile, f, indent=2)
    print(f"  Done. Style profile saved -> {profile_path}\n")

    # ── Job 3: Generate edit plans ───────────────────────────────────────────
    print("[ Job 3 / 4 ] Generating edit plans...")
    plans = plan_edits(transcript, style_profile, n_variants=n_variants, mock=mock)
    print_plans(plans)
    plans_path = os.path.join(output_dir, f"{raw_name}_plans.json")
    with open(plans_path, "w") as f:
        json.dump([p.model_dump() for p in plans], f, indent=2)
    print(f"  Plans saved -> {plans_path}\n")

    # ── Job 4: Render ────────────────────────────────────────────────────────
    print("[ Job 4 / 4 ] Rendering variants...")
    rendered = []
    for i, plan in enumerate(plans, 1):
        out = os.path.join(output_dir, f"{raw_name}_variant_{i:02d}.mp4")
        print(f"\n  Rendering variant {i}/{len(plans)} -> {out}")
        render(raw_video, plan, transcript["words"], out)
        rendered.append(out)

    print(f"\n{'='*60}")
    print(f"Done. {len(rendered)} video(s) saved to {output_dir}/")
    for path in rendered:
        print(f"  {path}")
    print("="*60)


if __name__ == "__main__":
    main()
