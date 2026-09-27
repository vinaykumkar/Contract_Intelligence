"""Assemble the ContractIQ walkthrough video: slide segments + demo recording + narration."""
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
BUILD = Path(r"D:\ContractIQ\reports\deck_assets\video_build")
RENDER = Path(r"D:\ContractIQ\reports\deck_assets\render")
OUT = Path(r"D:\ContractIQ\reports\deck_assets\ContractIQ_Walkthrough.mp4")

durations = json.loads((BUILD / "durations.json").read_text())
TAIL = 0.6  # silence after each segment

# segment id -> still slide (demo segment uses the recorded webm)
STILLS = {
    "01_title": "Slide1.PNG", "02_problem": "Slide3.PNG", "03_insight": "Slide4.PNG",
    "04_dataset": "Slide7.PNG", "05_model": "Slide9.PNG", "06_flow": "Slide11.PNG",
    "07_architecture": "Slide8.PNG", "08_risk": "Slide13.PNG", "09_quality": "Slide18.PNG",
    "11_results": "Slide12.PNG", "12_deploy": "Slide17.PNG", "13_limits": "Slide20.PNG",
    "14_closing": "Slide21.PNG",
}

def run(args):
    r = subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error"] + args,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1200:])
        sys.exit(1)

webm = next(BUILD.glob("*.webm"))
seg_files = []

# --- still segments ---
for seg_id, slide in STILLS.items():
    wav = BUILD / f"{seg_id}.wav"
    dur = durations[seg_id] + TAIL
    mp4 = BUILD / f"seg_{seg_id}.mp4"
    run(["-loop", "1", "-framerate", "30", "-i", str(RENDER / slide),
         "-i", str(wav),
         "-vf", "scale=1920:1080",
         "-af", f"apad=pad_dur={TAIL}",
         "-c:v", "libx264", "-preset", "medium", "-tune", "stillimage",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "22050", "-ac", "1",
         "-t", str(dur), "-shortest", str(mp4)])
    seg_files.append(mp4)
    print("still segment:", seg_id, f"{dur:.1f}s")

# --- demo segment (screen recording + narration) ---
demo_wav = BUILD / "10_demo.wav"
demo_dur = durations["10_demo"] + TAIL
demo_mp4 = BUILD / "seg_10_demo.mp4"
run(["-i", str(webm), "-i", str(demo_wav),
     "-vf", "fps=30,scale=1920:1080,tpad=stop_mode=clone:stop_duration=20",
     "-af", f"apad=pad_dur={TAIL}",
     "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
     "-c:a", "aac", "-ar", "22050", "-ac", "1",
     "-t", str(demo_dur), "-shortest", str(demo_mp4)])
seg_files.insert(9, demo_mp4)  # after 09_quality
print("demo segment:", f"{demo_dur:.1f}s")

# --- concat ---
lst = BUILD / "concat.txt"
lst.write_text("".join(f"file '{f.as_posix()}'\n" for f in seg_files), encoding="utf-8")
run(["-f", "concat", "-safe", "0", "-i", str(lst),
     "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
     "-c:a", "aac", "-ar", "22050", "-ac", "1", "-movflags", "+faststart",
     str(OUT)])

size_mb = OUT.stat().st_size / 1e6
print(f"DONE: {OUT}  ({size_mb:.1f} MB)")
