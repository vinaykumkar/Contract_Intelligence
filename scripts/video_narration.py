"""Generate narration audio segments for the ContractIQ demo video (Windows SAPI TTS)."""
import json
import os
import subprocess
import wave
from pathlib import Path

OUT = Path(r"D:\ContractIQ\reports\deck_assets\video_build")
OUT.mkdir(parents=True, exist_ok=True)

# (id, still slide png, narration text)
SEGMENTS = [
    ("01_title", "Slide1.PNG",
     "Contract IQ. A I Contract Intelligence and Risk Analysis. A local-first contract review tool, "
     "built end to end in Python and React. This video walks through the problem, the machine learning "
     "behind it, and a live demonstration."),
    ("02_problem", "Slide3.PNG",
     "The problem. Commercial contracts bury their most dangerous terms, like auto renewals, non compete "
     "clauses, and liability caps, inside a hundred pages of boilerplate. The median contract in the CUAD "
     "study runs about six thousand seven hundred tokens, and manual review means one expert reading "
     "everything to find the few clauses that matter. Cloud based review tools are often a non starter, "
     "because contracts are among the most sensitive documents a company holds."),
    ("03_insight", "Slide4.PNG",
     "The key insight. Instead of training a separate classifier for every clause type, Contract IQ casts "
     "extraction as extractive question answering. Each clause category becomes a question, such as, is "
     "there a non compete restriction? This gives three things for free. The evidence span is the answer "
     "itself. Absence of a clause is modeled natively through the no answer head. And all forty one CUAD "
     "categories share one small model. The pipeline runs from upload, to parsing, to extraction, to risk "
     "scoring, to review."),
    ("04_dataset", "Slide7.PNG",
     "The data. Contract IQ is fine tuned on CUAD, the Contract Understanding Atticus dataset. Five hundred "
     "and ten real commercial contracts, annotated by lawyers with forty one clause categories and over "
     "twenty thousand nine hundred question answer pairs. The split is three hundred sixty seven contracts "
     "for training, forty one for validation and threshold calibration, and one hundred two held out for "
     "testing. The test set was evaluated exactly once, with frozen thresholds."),
    ("05_model", "Slide9.PNG",
     "Model selection. The engine uses Mini LM, a thirty three million parameter transformer fine tuned "
     "from the deep set squad two checkpoint. It was chosen because it is the fastest small model per "
     "window on CPU in the project benchmark, at four point four three windows per second, and the only "
     "small model trained on no answer data. The whole model is about one hundred twenty seven megabytes, "
     "which is what makes the privacy promise possible. Inference never needs a data center."),
    ("06_flow", "Slide11.PNG",
     "From document to evidence. A contract can reach seventy eight thousand tokens, far beyond the "
     "model's context. So the text is split into blocks, ranked by T F I D F with clause specific keywords "
     "and title boost, and only the top fifteen blocks are analyzed, with four high value clause types "
     "scanning the full document. Gold answer recall at this stage is ninety five point two percent. "
     "Windows of five hundred twelve tokens go through the model, and every decoded span maps back to "
     "exact character offsets in the original document."),
    ("07_architecture", "Slide8.PNG",
     "The architecture is a strict four layer pipeline. A React nineteen single page app talks to a Fast "
     "A P I backend over H T T P. The backend orchestrates the document parser, the machine learning "
     "adapter, and the risk engine, and persists everything through SQLAlchemy into SQLite. The M L layer "
     "sits behind a Python protocol, so the model can be swapped without touching the backend. All state "
     "changes flow in one direction, and the U I never writes to the database directly."),
    ("08_risk", "Slide13.PNG",
     "The risk engine is deliberately not machine learning. C U A D contains no risk labels, so any risk "
     "model would be fake. Instead, fifteen transparent, deterministic rules score the contract from zero "
     "to one hundred, in three bands. For example, a non compete clause adds eighteen points, exclusivity "
     "fifteen, a missing liability cap twelve, and auto renewal language ten. A rule only fires when the "
     "extraction confidence is at least zero point five. Below that, the finding is marked uncertain and "
     "excluded from the score. Every point is auditable, line by line."),
    ("09_quality", "Slide18.PNG",
     "Quality is verified end to end. One hundred ninety nine automated tests pass, one hundred eighty "
     "six in pytest and thirteen in vitest. A twenty two step browser test runs the real model from "
     "upload to analysis with zero page errors. And the defining invariant, that every stored clause span "
     "exactly matches the text at those offsets, was checked one hundred twenty nine times across real "
     "analyses, with zero mismatches."),
    ("10_demo", "__DEMO__",
     "Now, a live demonstration. This is the overview dashboard, with portfolio statistics and a live "
     "engine status. Opening a high risk contract. The risk orb scores it sixty seven, in the high band. "
     "Clicking a clause card highlights the exact evidence in the document text, character for character. "
     "Below, every risk finding shows its rule, its weight, and its own evidence, so the score of sixty "
     "seven is fully auditable. The contract history keeps every past analysis searchable. Here is a "
     "low risk agreement, the same engine, and the orb stays low because almost nothing fires. Finally, "
     "the intelligence page shows the model state, the compute device, and the risk methodology, with the "
     "disclaimer that this is A I assisted analysis, not legal advice."),
    ("11_results", "Slide12.PNG",
     "Results, measured honestly on the official CUAD test set. Sixty point eight percent exact match and "
     "sixty six point two percent token F one overall. No answer accuracy is seventy four point four "
     "percent, meaning the model usually knows when a clause is absent. Clause level F one ranges from "
     "eighty nine percent for document names down to thirteen point seven percent for parties, which "
     "needs multi span decoding, and that weakness is disclosed in the report rather than hidden."),
    ("12_deploy", "Slide17.PNG",
     "Deployment and privacy. The production build serves the React app and the A P I from a single Fast "
     "A P I process on one port. Uploads are validated and stored under generated names, path traversal is "
     "neutralized, and contract text is never written to logs. Nothing ever leaves the machine. The "
     "project is published on GitHub, with the fine tuned model attached as a release asset."),
    ("13_limits", "Slide20.PNG",
     "Limitations, stated openly. It is a single user, local application without authentication. Analysis "
     "is synchronous. Risk rules are heuristics, not learned from outcomes. Scanned P D Fs need O C R, "
     "which is not implemented yet. And parties extraction needs multi span decoding. The roadmap: an "
     "async job queue, O C R support, a learned risk layer, wider clause coverage, and a team mode."),
    ("14_closing", "Slide21.PNG",
     "Contract IQ. Every finding traceable to its source text. Thank you for watching."),
]

VOICE = "Microsoft David Desktop"
RATE = -1

for seg_id, _slide, text in SEGMENTS:
    if text == "__DEMO__":
        continue
    txt_path = OUT / f"{seg_id}.txt"
    wav_path = OUT / f"{seg_id}.wav"
    txt_path.write_text(text, encoding="utf-8")
    ps = (
        f"Add-Type -AssemblyName System.Speech; "
        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SelectVoice('{VOICE}'); $s.Rate = {RATE}; "
        f"$s.SetOutputToWaveFile('{wav_path}'); "
        f"$s.Speak((Get-Content -Raw '{txt_path}')); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True,
                   capture_output=True, timeout=180)

durations = {}
for seg_id, _slide, _text in SEGMENTS:
    wav = OUT / f"{seg_id}.wav"
    if wav.exists():
        with wave.open(str(wav), "rb") as w:
            durations[seg_id] = round(w.getnframes() / w.getframerate(), 2)
(OUT / "durations.json").write_text(json.dumps(durations, indent=2))
total = sum(durations.values())
print(f"generated {len(durations)} narration segments, total speech {total:.0f}s")
for k, v in durations.items():
    print(f"  {k:16} {v:6.1f}s")
