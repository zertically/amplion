# Amplion — How to Get Started

## What you're building

A Python CLI script. You give it a raw video and example videos. It gives you back multiple edited videos styled like the examples.

```
python amplion.py --plain raw.mp4 --examples ref1.mp4 ref2.mp4 --variants 3
```

No frontend. No server. No database. A script that takes files in and puts files out.

---

## Install these things first

```bash
pip install openai pydantic
```

Install FFmpeg if you don't have it:
- Mac: `brew install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`
- Windows: download from ffmpeg.org, add to PATH

Get an OpenAI API key from platform.openai.com. You need `whisper-1` and `gpt-4o`. Set it:

```bash
export OPENAI_API_KEY="sk-..."
```

That's all the setup.

---

## How the script works (the whole thing)

There are 4 jobs. Each one feeds into the next.

**Job 1 — Transcribe the raw video.** Send its audio to Whisper API. You get back every word with its start and end timestamp in seconds. This is how the system knows what was said and when.

**Job 2 — Analyze the example videos.** Extract a screenshot every 2 seconds from each example video using FFmpeg. Also transcribe them. Send the screenshots + transcript to GPT-4o with a prompt that says: "You're a video editor. Look at this video and tell me its editing style — pacing, captions, structure, overlays, energy." GPT-4o returns a JSON style profile describing how the example was edited.

**Job 3 — Generate edit plans.** Send the raw video's transcript + the style profile to GPT-4o with a prompt that says: "Here's a raw transcript with timestamps. Here's the editing style to match. Create N different edit plans. Each plan is a list of segments to keep (with start/end timestamps), caption settings, and any text overlays." GPT-4o returns a JSON with multiple edit plans, each a different creative interpretation.

**Job 4 — Render.** For each edit plan, use FFmpeg to: trim the selected segments out of the raw video, concatenate them together, burn in captions using an ASS subtitle file you generate from the transcript, and add any text overlays (hooks, CTAs) with FFmpeg's drawtext filter. Save each variant as a separate MP4.

That's the entire product. Four jobs. Two API calls (Whisper + GPT-4o), one rendering tool (FFmpeg).

---

## Where to start writing code

Start with the smallest piece that proves something works, then add the next piece.

### First: Make transcription work

Write a function that takes a video file path, extracts the audio with FFmpeg (`ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav`), sends that WAV to the Whisper API with `response_format="verbose_json"` and `timestamp_granularities=["word", "segment"]`, and returns the words with timestamps.

Test it. Print the words. Confirm the timestamps make sense against the actual video. When this works, you have the foundation everything else builds on.

### Second: Make reference analysis work

Write a function that takes a video file path, uses FFmpeg to extract one frame every 2 seconds (`ffmpeg -i video.mp4 -vf fps=0.5 frame_%04d.jpg`), base64-encodes those frames, and sends them to GPT-4o along with the video's transcript.

The prompt you send to GPT-4o is the most important thing in Amplion. It needs to ask for specific, actionable style information structured as JSON. The things you want to extract are:

- **Structure:** How long is the hook? Is there a text hook overlay? How many distinct sections? Is there a CTA at the end?
- **Pacing:** How fast are cuts? Is silence removed or left in? What's the energy like?
- **Captions:** Are there captions? Where are they positioned? What do they look like (simple, bold, animated)?
- **Visual treatment:** Aspect ratio? Zoom/punch-in effects? B-roll?
- **Text overlays:** Any on-screen text? What does it say and when does it appear?

Use `response_format={"type": "json_object"}` so GPT-4o returns parseable JSON. Use low temperature (0.2) for consistent results.

Test it on 3-4 very different example videos. Read the JSON it returns. Does it accurately describe what you see when you watch those videos? Tweak the prompt until it does. This step is where you'll spend the most iteration time, and it's worth it — the quality of this analysis determines the quality of everything downstream.

### Third: Make edit planning work

Write a function that takes the raw video's transcript and a style profile, and asks GPT-4o to produce edit plans.

The edit plan is a JSON object. Each plan contains:
- A list of segments to keep, where each segment has a `start` and `end` timestamp from the raw video
- Caption configuration (on/off, style, position)
- Text overlays (hook text, CTA) with their timing

The prompt needs to tell GPT-4o:
- Here's the raw transcript with word-level timestamps
- Here's the editing style to match
- Create N variants that are meaningfully different (different hooks, different segment selections, different lengths)
- Only select segments using timestamps that exist in the transcript
- Remove dead air and silence
- Keep segments in chronological order

Define a Pydantic model for the edit plan schema so you can validate what comes back. Add a sanity check that clamps any timestamp to within the raw video's actual duration.

Use higher temperature (0.7) here so the variants are actually different from each other.

Test it by printing the plans and manually checking: do the timestamps make sense? Are the segments from real parts of the transcript? Are the variants actually different?

### Fourth: Make rendering work

Write a function that takes a raw video path and an edit plan, and produces an output MP4.

The rendering has 3 sub-steps:

1. **Cut segments.** For each segment in the plan, use FFmpeg to extract that time range from the raw video: `ffmpeg -ss {start} -i raw.mp4 -t {duration} -c:v libx264 -c:a aac clip_001.mp4`

2. **Concatenate.** Write a concat list file and use FFmpeg's concat demuxer to join all clips into one video: `ffmpeg -f concat -safe 0 -i concat.txt -c:v libx264 -c:a aac joined.mp4`

3. **Add captions and overlays.** Generate an ASS subtitle file from the transcript words that fall within the selected segments (recalculating timestamps relative to the output timeline, not the source timeline). Burn it in with `ffmpeg -i joined.mp4 -vf "ass=captions.ass" output.mp4`. Layer on any text overlays using FFmpeg's drawtext filter with the `enable='between(t,start,end)'` parameter.

Test it by playing the output video. Do the cuts happen where expected? Are captions roughly in sync? Does text appear and disappear at the right times?

### Fifth: Wire it together

Write the CLI entry point that chains all four jobs together: transcribe → analyze → plan → render. Use argparse for the `--plain`, `--examples`, and `--variants` arguments. Loop through each plan and render it to a separate output file.

---

## The file structure

```
amplion/
├── amplion.py              # CLI entry point, wires everything together
├── transcribe.py           # Whisper API: video → transcript with timestamps
├── analyze_reference.py    # GPT-4o vision: example video → style profile JSON
├── plan_edits.py           # GPT-4o: transcript + style → edit plan JSON
├── render.py               # FFmpeg: raw video + edit plan → output MP4
├── schemas.py              # Pydantic models for the edit plan structure
├── prompts/
│   ├── analyze.txt         # System prompt for reference analysis
│   └── plan.txt            # System prompt for edit planning
├── output/                 # Rendered videos land here
└── temp/                   # Working files (frames, clips, subtitles)
```

---

## What will break and how to deal with it

**GPT-4o returns timestamps outside the video's duration.** Clamp every segment's start and end to `[0, video_duration]` after parsing. Reject segments shorter than 0.3 seconds.

**FFmpeg concat fails because clips have different resolutions.** Add a scale+pad filter to the clip extraction step that forces all clips to the same resolution: `-vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:-1:-1:color=black"`

**Caption timing is wrong after concatenation.** This happens when you use the source video timestamps instead of recalculating them relative to the concatenated output. For each segment, track a running offset that accumulates the duration of previous segments.

**The style analysis misses something obvious about the example video.** Improve the prompt. This is the part of the project where prompt engineering matters most. Look at what GPT-4o returns, compare it to what you see in the video, and refine the prompt until they match.

**Text overlays don't render because the font path is wrong.** Check what fonts exist on your system. On Mac: `/System/Library/Fonts/`. On Ubuntu: `/usr/share/fonts/truetype/dejavu/`. Point FFmpeg's `fontfile=` parameter to an actual .ttf file.

---

## Cost per video

Whisper transcription costs $0.006 per minute of audio. GPT-4o with ~20 images costs roughly $0.02-0.08 per call depending on image count and response length. You make 2-3 GPT-4o calls per video (one for analysis, one for planning, occasionally a retry). FFmpeg is free.

A typical 3-minute raw video with one reference and 3 variants costs roughly $0.10-0.25 total.
