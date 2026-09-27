# ContractIQ — Interview / Examination Guide

A complete playbook for explaining and demonstrating this project to an
interviewer or examiner. Rehearse with this file open next to the running app.

---

## 1. The 30-second elevator pitch (memorize)

> "ContractIQ is a local-first contract review tool. You upload a PDF, DOCX or
> TXT agreement, and a MiniLM question-answering transformer — fine-tuned on
> the CUAD legal dataset — extracts 15 high-value clause types and returns the
> **exact character offsets** of every finding, so the app can highlight the
> evidence in the document itself. A transparent 15-rule engine then scores
> contract risk 0–100 — deterministic rules, never the model, so every point
> is auditable. Everything runs on the user's own machine: no cloud, no API
> keys, contracts never leave the laptop. It's verified end to end — 199
> automated tests, a real-model browser E2E, and an evidence invariant with
> zero mismatches across 129 checks."

Say it slowly. The three phrases that impress: **exact character offsets**,
**rules never the model**, **evidence invariant — zero mismatches**.

---

## 2. The story arc (2–3 minutes, before the demo)

Tell it in five beats:

1. **The problem.** "Contracts bury their most dangerous terms — auto-renewals,
   non-competes, liability caps — inside 100 pages of boilerplate. CUAD's own
   study pegs the median contract at ~6,700 tokens; manual review means one
   expert reading everything to find 15 things that matter. And cloud review
   tools are a non-starter for confidential documents."

2. **The insight (your key decision).** "Instead of training a classifier per
   clause type, I cast extraction as **extractive question answering**: each
   clause category is a question — 'Is there a non-compete restriction?' —
   asked over the contract. This gives three things for free: the **evidence
   span** is the answer itself, **absence** is natively modeled (SQuAD-2
   no-answer head), and all 41 CUAD categories share one small model."

3. **The engineering pillars.** "Three principles shaped the build: evidence
   — every AI claim stores its exact character range; honest uncertainty —
   low-confidence findings are flagged 'uncertain' and excluded from scoring;
   local-first — the 33M-parameter model runs on a plain laptop CPU."

4. **The proof.** "199 automated tests, a 22-step browser E2E with the real
   model, and the deployment is production-shaped: one FastAPI process serves
   both the built React app and the API."

5. **Honesty as a feature.** "The report publishes the weak spots — overall
   60.8% exact match, parties extraction at 13.7% F1 — because a review tool
   that hides its error modes is dangerous. The UI shows uncertainty instead
   of hiding it."  ← *This is what separates you from every other student project.*

---

## 3. Live demo runbook

### Before the meeting (10 minutes)

1. Start the app: double-click `run_contractiq_prod.bat` → http://127.0.0.1:8010
2. **Pre-analyze the three demo files once** (upload + analyze on the
   New Analysis page): `demo_contracts/01_saas_subscription_low_risk.docx`,
   `05_consulting_services_medium_risk.docx`, `09_franchise_agreement_high_risk.docx`.
   CPU analysis takes ~30–90 s each; doing this beforehand means during the
   demo you just **reopen them from History** and results appear instantly.
   It also **pre-warms the model** (first analysis after a cold start loads
   weights for ~30 s — never do that live).
3. Keep the GitHub repo open in a browser tab.

### The demo flow (3–4 minutes)

| Step | What to do | What to say |
|---|---|---|
| 1 | Overview page | "This is the workspace — engine status is live bottom-right, portfolio stats and risk constellation update from the database." |
| 2 | Open the **HIGH** contract from History (09 franchise, or demo_services_agreement 67/HIGH) | "Risk Orb: 67 — HIGH. The orb is animated but never color-only: numeric readout, accessible role." |
| 3 | Click a clause card (e.g. Non-Compete) | "Watch the document: the exact span is highlighted. That offset came from the model and is stored in SQLite — the viewer renders only from stored offsets." |
| 4 | Scroll the Risk Findings panel | "Each finding: rule, weight, severity, its own evidence. 67 = +18 non-compete, +15 exclusivity, +10 auto-renewal… every line contestable." |
| 5 | Open the **LOW** contract (01 SaaS) | "Same engine, friendly terms — the orb stays low, cap-on-liability and insurance are found, nothing risky fires." |
| 6 | Intelligence page | "Model state: fine-tuned on CUAD, compute device shown honestly, and the disclaimer everywhere — AI-assisted, not legal advice." |
| 7 | (If time) New Analysis page | "Drag-and-drop upload; corrupt, encrypted or scanned files are rejected with typed errors — the system never fabricates text." |

### If something goes wrong live

- Analysis feels slow → it's CPU mode by design; say "about a minute on CPU,
  seconds on GPU — the speed numbers are in the report."
- App closed → relaunch `run_contractiq_prod.bat`; analyses are in the DB,
  nothing is lost.
- Asked to analyze a NEW file live → fine, but narrate the wait: "this is the
  cold model load — subsequent analyses are faster."

---

## 4. Numbers cheat sheet (the ones worth memorizing)

| Fact | Number |
|---|---|
| CUAD dataset | 510 contracts · 41 categories · 20,910 QA pairs |
| Splits | 367 train / 41 validation / 102 test (test touched once) |
| Model | MiniLM (deepset/minilm-uncased-squad2) · 33.4M params · ~127 MB |
| Fine-tune | 1 epoch · 20,420 windows · 838 s on an RTX 2050 |
| Test results | EM 60.8% · F1 66.2% · no-answer accuracy 74.4% · answerable recall 83.2% |
| Best / worst clauses | Document name 89.0% F1 · Governing law 84.5% · Parties 13.7% (disclosed) |
| Retrieval | 79.4% → 91.4% → **95.2%** gold-answer recall @ K=15 |
| Speed | 4.43 win/s CPU · 52.8 GPU = **11.9×** · ~80 s CPU / 4–12 s GPU per contract |
| Evidence invariant | **0 mismatches across 129 checks** |
| Tests | 199 passing (186 pytest + 13 vitest) · E2E 22 steps · 0 page errors |
| Frontend | React 19 + Vite · 135 kB gzipped bundle |
| Backend | FastAPI · 10 routes · 4 SQLite tables · 15 domain exceptions |
| Risk engine | 15 rules · 0–100 · bands 0-30 / 31-60 / 61-100 · confidence gate 0.5 |

---

## 5. Question bank — likely questions and strong answers

**Q: Why question answering instead of a clause classifier?**
"A classifier gives a label; QA gives a **span**. The span is the evidence the
UI highlights — no separate mapping step. QA with a SQuAD-2 head also models
*absence* natively, which a classifier can't do honestly. And one QA model
covers all 41 categories through different question templates."

**Q: Why MiniLM and not a bigger model?**
"Local-first is a hard constraint — the privacy promise only holds if
inference runs on the user's machine. MiniLM is 33M parameters, ~127 MB, and
was the fastest per-window on CPU in my benchmark (4.43 win/s vs DistilBERT's
3.41, RoBERTa ~1.2) *and* the only small one trained on no-answer data. On a
GPU the same code is 11.9× faster."

**Q: Why is the risk score not machine learning?**
"CUAD has no risk labels — any 'risk model' would be fake. So risk is a
deterministic, configurable rule set (weights live in clauses.json): the model
extracts facts, rules score them, and a confidence gate (≥ 0.5) excludes weak
extractions. Separation of concerns, and every point is auditable."

**Q: How do you trust the highlighting?**
"The defining invariant: stored spans must exactly match the document text.
I tested it across 129 checks — browser and database, real analyses — zero
mismatches. There's a pytest suite for the offset math too."

**Q: Only 15 of the 41 categories — why?**
"The registry covers all 41; v1 production-enables 15. Three reasons: tail
categories have too few annotations for reliable 1-epoch extraction, CPU
runtime scales with clause count, and the 15 chosen map directly to the risk
engine and the highest-stakes commercial terms. Enabling more is a config
flag — it's in Future Work."

**Q: Biggest limitation?**
"Parties extraction — 13.7% F1, because gold answers span multiple entities
and my decoder is single-span. The fix is multi-span decoding; I'd also add an
async job queue and OCR next. All documented in the phase-8 QA report."

**Q: Security?**
"Uploads are untrusted input: type/size validation, UUID storage names,
traversal-sanitized filenames, content never executed. Contract text is never
logged. No auth by design — it's a single-user local app; LAN mode is opt-in
and documented as trusted-networks-only."

**Q: Why SQLite? Where's this going?**
"Right-sized for single-user local persistence, accessed only through
SQLAlchemy with bound parameters. The DB URL is config — the docs describe
the Postgres migration path for a hosted deployment."

**Q: How is it deployed?**
"Production mode: one FastAPI process serves the built React bundle and the
API on a single port with SPA fallback — deep links survive refresh. Pushed to
GitHub; the 127 MB model ships as a Release asset with a documented baseline
fallback so the repo itself stays light."

---

## 6. Do / Don't

- **Do** open with the honest-limits slide before they ask — owning 13.7% F1
  is more impressive than being caught by it.
- **Do** use the artifacts: the 21-slide deck, the 27-page report with the
  phase-by-phase evidence trail in `reports/`, the GitHub repo.
- **Do** say "extractive QA", "evidence offsets", "confidence gate" — precise
  vocabulary signals ownership.
- **Don't** say it "never misses" a clause (recall is 83.2%) or that it
  replaces lawyers — the disclaimer is part of the design.
- **Don't** memorize the whole report — know §1 (pitch), §4 (numbers), §5
  (answers) cold; for anything else, open the doc live: "the exact figure is
  in the report — let me show you."

---

## 7. One-page rehearsal script

**Pitch (30 s)** → **Architecture in one breath (20 s):** "React 19 SPA →
FastAPI → parser, ML adapter, risk engine → SQLAlchemy/SQLite; the ML layer
sits behind a Protocol so models swap without backend changes." →
**Demo (3–4 min, per §3)** → **Results honestly (30 s):** test metrics +
evidence invariant + 199 tests → **Roadmap (20 s):** async queue, OCR,
multi-span parties decoding, more of the 41 categories, team mode →
**Close:** "The design goal was trust: every finding traceable to its source
text, every limitation documented."
