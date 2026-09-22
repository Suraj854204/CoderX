# CoderX

### Private AI Code Review. Powered by Snapdragon.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![React Native](https://img.shields.io/badge/React%20Native-Expo-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)
![Snapdragon X Elite](https://img.shields.io/badge/Hardware-Snapdragon%20X%20Elite-red.svg)
![Qualcomm AI Hub](https://img.shields.io/badge/Runtime-Qualcomm%20AI%20Hub%20%2F%20GenieX-6f42c1.svg)


Built for the **Snapdragon® Multiverse Hackathon**
## Table of contents

1. [Problem](#1-problem)
2. [Solution](#2-solution)
3. [Why on-device AI](#3-why-on-device-ai)
4. [Why Snapdragon](#4-why-snapdragon)
5. [Architecture](#5-architecture)
6. [AI pipeline](#6-ai-pipeline)
7. [Risk-aware routing](#7-risk-aware-routing)
8. [Incremental review cache](#8-incremental-review-cache)
9. [NPU inference](#9-npu-inference)
10. [Human-in-the-loop reiteration](#10-human-in-the-loop-reiteration)
11. [Privacy](#11-privacy)
12. [Offline capability](#12-offline-capability)
13. [Benchmarks](#13-benchmarks)
14. [Installation](#14-installation)
15. [Snapdragon setup](#15-snapdragon-setup)
16. [Demo](#16-demo)
17. [Screenshots](#17-screenshots)
18. [Technical details](#18-technical-details)
19. [Limitations](#19-limitations)
20. [Future work](#20-future-work)
- [Team](#team) · [References](#references) · [License](#license)

---

## 1. Problem

AI code review tools (GitHub Copilot, CodeRabbit, Cursor, etc.) are great,
but every one of them sends your source code to a cloud API. For a huge
class of developers — fintech, healthtech, defense, government, or just
companies with a strict data-residency policy — that's a non-starter. India
alone has an estimated 5.8M developers, and a meaningful slice of them work
somewhere that simply will not permit proprietary code to leave the
building, let alone the country.

## 2. Solution

CoderX answers a simple question: **can you get a genuinely useful AI code
reviewer without any of the code ever touching the network?**

`git commit` (or a GitHub PR) triggers diff extraction → a local LLM
running **entirely on a Snapdragon NPU** analyzes the diff → findings
stream live to a mobile triage app → the developer approves each finding or
flags it as a false positive → the AI reconsiders flagged findings given
the developer's reasoning → a final PDF/JSON report is generated on-device.

- ✅ Runs entirely on Snapdragon NPUs — no cloud API, no CPU fallback needed
- ✅ Reviews Git commits and GitHub PRs automatically
- ✅ Never uploads a single line of source code
- ✅ Risk-aware routing prioritizes sensitive changes and skips genuinely
  trivial ones before they reach the NPU
- ✅ Incremental cache skips re-reviewing a hunk it has already seen
- ✅ Lets the developer push back — the AI reconsiders any finding flagged
  as a false positive, given the developer's reasoning
- ✅ Generates an offline PDF + structured JSON audit trail

## 3. Why on-device AI

The answer is yes — **if the model runs on the same machine that owns the
code**, and modern Snapdragon NPUs are fast enough to make that practical
rather than theoretical. The alternative (send the diff to a cloud API) is
a non-starter for the exact developers CoderX targets, so on-device
inference isn't a performance optimization here — it's the feature that
makes the product usable at all for its intended audience.

## 4. Why Snapdragon

CoderX runs a 4B-parameter instruction-tuned model (Qwen3-4B-Instruct-2507)
entirely on-NPU via **Qualcomm AI Hub / GenieX**, confirmed end-to-end on
Snapdragon X Elite hardware — not a CPU fallback, not a mock (see
[Benchmarks](#13-benchmarks) for the measured numbers, and
[NPU inference](#9-npu-inference) for exactly how `geniex infer` is
invoked). Three properties of Snapdragon's NPU specifically make this
product possible rather than merely nice-to-have:

- **Enough throughput to be usable interactively.** A 4B-parameter model
  reviewing a diff the moment you commit needs real tokens/second, not
  cloud-GPU-class but also not "leave it running overnight" CPU-only
  speed. See the measured prefill/decode numbers in section 13.
- **Efficient enough to run alongside normal dev work**, not a dedicated
  inference server — the NPU handles this without monopolizing the CPU the
  developer is also using for their editor, browser, and build tools.
- **A supported path to close the loop with GenieX/QAIRT** — Qualcomm AI
  Hub is what let this team benchmark and select Qwen3-4B-Instruct-2507
  in the first place (see [Tech stack](#18-technical-details)), and GenieX
  is the one-shot CLI that makes NPU inference callable from a normal
  Python backend without a bespoke native integration.

## 5. Architecture

```mermaid
flowchart TD
    A["git commit<br/>(post-commit hook)"] --> C
    B["GitHub PR webhook<br/>(HMAC-verified)"] --> C
    C["Diff extraction<br/>git diff --function-context"] --> D["Split into hunks"]
    D --> E["Risk-Aware Review Scheduler<br/>risk_router.py"]
    E -->|CRITICAL_PATH<br/>always reviewed, sorted first| F["Incremental Review Cache<br/>review_cache.py"]
    E -->|STANDARD| F
    E -->|TRIVIAL &amp; small &amp; safe| G["Fast-path skip<br/>no NPU call, synthetic SUGGESTION finding"]
    F -->|cache hit| H["Reused verdict<br/>inference_path = cache"]
    F -->|cache miss| I["Snapdragon NPU<br/>geniex infer"]
    I --> J["Qwen3-4B-Instruct-2507"]
    J --> K["Structured findings<br/>response_parser.py"]
    G --> L["Mobile triage app<br/>React Native / WebSocket"]
    H --> L
    K --> L
    L -->|Approve| M["Final verdict"]
    L -->|"False positive + developer reasoning"| N["Reiteration<br/>reiteration.py"]
    N -->|MAINTAINED / WITHDRAWN / PARTIALLY_VALID| M
    M --> O["PDF + TXT + JSON audit trail<br/>report_generator.py"]
```

Both entrypoints (the local git hook and the GitHub webhook) converge on the
same router → cache → NPU pipeline — an earlier audit of this codebase
found the webhook path had its own older copy of the review loop that
predated the router/cache entirely, which has since been fixed so a PR
review gets identical treatment to a local commit review.

## 6. AI pipeline

Each hunk is reviewed independently, not the whole diff at once:

1. `diff_extractor.py` runs `git diff --function-context`, so each hunk
   carries its full enclosing function, not just the default 3 lines of
   context.
2. `prompt_builder.py` wraps the hunk in a fixed-format instruction prompt
   (`PROMPT_TEMPLATE`, versioned as `PROMPT_TEMPLATE_VERSION` so a prompt
   edit automatically invalidates any cached verdicts produced under the
   old wording).
3. `llm_client.py` dispatches to `geniex infer` (NPU) or Ollama (CPU
   dev/fallback), selected by `CODERX_BACKEND`.
4. `response_parser.py` parses the model's `[SEVERITY] file:line —
   description. Fix: ...` formatted text into structured `Finding` objects,
   now carrying provenance metadata (`category`, `inference_path`, `model`,
   and an honestly-`None` `confidence` — see section 19).

## 7. Risk-aware routing

`risk_router.py` classifies every hunk **on-device, with zero model
calls**, before anything reaches the NPU:

```mermaid
flowchart TD
    Start["Hunk diff text"] --> Sig{"Matches a risk signal?<br/>auth · authz · crypto · secrets · sql ·<br/>deserialization · shell_exec ·<br/>filesystem_write · network_boundary ·<br/>memory_unsafe · privilege_boundary"}
    Sig -->|Yes| Critical["CRITICAL_PATH<br/>always reviewed, sorted to top of triage queue"]
    Sig -->|No| Trivial{"Every CHANGED line is<br/>comment / blank / plain import only?"}
    Trivial -->|No| Standard["STANDARD<br/>reviewed normally, cache-eligible"]
    Trivial -->|Yes| Size{"Changed lines ><br/>CODERX_TRIVIAL_MAX_LINES (default 25)?"}
    Size -->|Yes, too large to trust the heuristic| Standard
    Size -->|No, small &amp; structurally safe| Skip["TRIVIAL<br/>skip NPU entirely, synthetic SUGGESTION finding"]
```

Two safeguards exist specifically to prevent a heuristic false-negative
from hiding a meaningful change:

- **Risk signals always win**, checked first — a comment line that happens
  to mention `sudo` or `subprocess` still routes `CRITICAL_PATH`.
- **A size cap** (`CODERX_TRIVIAL_MAX_LINES`, default 25) — a large
  diff that structurally *looks* trivial (e.g. a 40-line bulk comment
  edit) is escalated to `STANDARD` and reviewed anyway, never silently
  skipped, on the reasoning that large mechanical-looking diffs are
  exactly where a heuristic is most likely to miss something.

One caveat worth stating plainly: because diff extraction uses
`--function-context`, a hunk's diff text includes unchanged surrounding
lines, and risk-signal matching runs against that full text — so editing a
comment a few lines above an existing SQL query still classifies
`CRITICAL_PATH`. This is intentionally conservative (errs toward review,
not toward missing a sensitive area), and the `TRIVIAL` fast-path is
unaffected by it, since that path separately requires every *changed*
line specifically to be safe.

Every commit prints a routing summary, e.g.:

```
Commit 4c1d4ee
  4 hunk(s)

  CRITICAL_PATH   1
  STANDARD        2
  TRIVIAL         1

  NPU reviews performed: 3
  Served from cache:     0  (hit rate: 0%)
  Skipped (fast-path):   1
```

## 8. Incremental review cache

```mermaid
sequenceDiagram
    participant P as Pipeline
    participant C as review_cache.py (SQLite)
    participant N as Snapdragon NPU

    P->>C: lookup(sha256(normalized diff), model + prompt version)
    alt cache hit
        C-->>P: cached Finding(s)<br/>inference_path re-tagged "cache"
    else cache miss
        P->>N: geniex infer (real NPU call)
        N-->>P: raw model output
        P->>C: store(hash, parsed findings)
    end
```

- **Normalized hashing** — `@@ -N,M +N,M @@` line-number headers are
  stripped before hashing, so a hunk that shifted a few lines because of
  an earlier edit elsewhere in the file still hits the cache if the actual
  changed code is identical.
- **Model + prompt-version namespacing** — the cache key is
  `{backend}:{model}@prompt-{version}`, so switching `CODERX_GENIE_MODEL`
  or editing `PROMPT_TEMPLATE` never serves a stale verdict from a
  different model or instruction set.
- **Deterministic invalidation** — `review_cache.invalidate_model()` /
  `invalidate_all()`, or `python review_cache.py --clear`, rather than
  implicit TTL/expiry logic.
- **Storage** — a single local SQLite file
  (`backend/coderx_cache.db`), no network, no server dependency.

Measured behavior (this repository's own test run, reproducible via
`python demo.py`): reviewing the same commit a second time produced **0
new NPU calls, 100% cache hit rate**, with `review_cache.lookup()`
returning the original findings including their original `category` and
`model` metadata, re-tagged `inference_path: "cache"` for that run.

## 9. NPU inference

`llm_client.py` selects the backend via `CODERX_BACKEND` (`qnn` by
default):

```bash
geniex infer ai-hub-models/Qwen3-4B-Instruct-2507 -c npu --prompt-file ...
```

| Variable | Purpose | Default |
|---|---|---|
| `CODERX_GENIE_MODEL` | Model identifier passed to `geniex infer` | `ai-hub-models/Qwen3-4B-Instruct-2507` |
| `CODERX_GENIE_COMPUTE` | Compute target | `npu` |
| `CODERX_GENIE_THINK` | Model's thinking-mode toggle | `false` |
| `CODERX_GENIE_PROMPT_FILE` | Path to the fixed prompt file `geniex` reads from | `backend/coderx_geniex_prompt.txt` |

Ollama (`CODERX_BACKEND=ollama`) is the CPU dev/fallback path for machines
without a Snapdragon NPU — functionally identical pipeline, just slower
and not NPU-accelerated. `CODERX_MOCK_LLM=1` bypasses the LLM entirely with
a canned response, for testing pipeline mechanics (routing, caching,
report generation) without any model call at all — always clearly labeled
`backend=mock` in logs and benchmark output, never presented as a real
inference result.

## 10. Human-in-the-loop reiteration

```mermaid
flowchart LR
    F["AI finding"] --> D{"Developer decision"}
    D -->|Approve| Final1["Final verdict: Approved"]
    D -->|"False positive<br/>(a real reason is required)"| R["Reiteration<br/>original diff + developer's comment<br/>re-sent to the LLM"]
    R --> V{"Model verdict"}
    V -->|MAINTAINED| Final2["Finding stands"]
    V -->|WITHDRAWN| Final3["Finding withdrawn"]
    V -->|PARTIALLY_VALID| Final4["Partially valid, explained"]
```

A one-word dismissal is rejected before it ever reaches the model — this
is deliberately not an autonomous reviewer that produces warnings and
walks away; every disagreement is argued out, and the outcome (developer's
reasoning + the model's re-evaluation) is recorded, not just the final
severity. This trail is written to all three report formats — see the
JSON example below — so it survives past the mobile session itself once a
report is generated.

```json
{
  "finding_id": "f_a1b2",
  "severity": "MAJOR",
  "category": "auth",
  "decision": "false_positive",
  "developer_comment": "Rate limiting is enforced upstream by the gateway.",
  "llm_verdict": "WITHDRAWN",
  "llm_reiteration": "Confirmed — upstream gateway rate limiting makes this a non-issue here."
}
```

## 11. Privacy

- No source code, diff, or finding is ever sent to a network endpoint for
  the purpose of review — inference happens via `geniex infer` against a
  locally-loaded model.
- The mobile app performs **zero on-device inference** by design: it is
  strictly a triage/display client over a local WebSocket. The privacy
  story is "the machine that owns the code also runs the model," so the
  phone never needs — and never gets — the source code beyond what's shown
  in a finding's own diff snippet.
- The one network call that *does* exist is optional: fetching a real
  GitHub PR's diff via GitHub's API (only relevant to the webhook path,
  not local commits), and only to retrieve the diff itself, never to send
  it anywhere.

## 12. Offline capability

Local commit review (`git commit` → post-commit hook → NPU review) has no
network dependency at any stage and has been run and tested that way in
this codebase's mock/CPU-backend test suite (`tests/test_pipeline.py`) and
manually against a real local git repo. The one path that genuinely
requires network access is fetching a **real** GitHub PR diff through
GitHub's REST API — that specific feature, not the core review pipeline,
is the only thing offline mode doesn't cover.

## 13. Benchmarks

**Model:** Qwen3-4B-Instruct-2507 (non-thinking variant)
**Hardware:** Snapdragon X Elite CRD, on-NPU, via GenieX

| Metric | Value | Status |
|---|---|---|
| Prefill throughput | ~1,301 tok/s | **Measured** (venue hardware) |
| Decode throughput | ~23.1 tok/s | **Measured** (venue hardware) |
| Per-hunk latency | ~11.8s – 34.4s | **Measured** (venue hardware) |
| Context window | 4,096 tokens | **Measured** (venue hardware) |
| Cache hit rate (repeat-commit test) | 100% (0 new NPU calls) | **Measured** (this repo's own test run) |
| Cold-start / model-load time | — | **Not yet measured** |
| CPU-only (Ollama) vs. NPU comparison | — | **Not yet measured** |
| Cache-driven latency saved, real hardware | — | **Not yet measured** — `RoutingSummary` computes this from each run's own real average NPU latency and explicitly declines to show a number when no real NPU call happened in that run to measure against, rather than estimating from a stale or foreign figure |

The prefill/decode/latency numbers come from real per-hunk benchmark runs
on venue hardware (`backend/benchmark.py`), not simulated or CPU-only
figures. The harness itself (`benchmark.py`) has been re-verified to run
correctly end-to-end in this pass, but re-running it requires the actual
Snapdragon NPU + GenieX setup described in section 15 to produce a real
number — running it in `CODERX_MOCK_LLM=1` mode only confirms the harness
mechanics, and its own output self-labels `backend=mock` for exactly that
reason.

## 14. Installation

**Prerequisites**

| Tool | Why | Check |
|---|---|---|
| Python 3.10+ | Backend pipeline, webhook listener | `python3 --version` |
| Node.js (LTS) + npm | Mobile app (Expo/React Native) | `node --version` |
| Git | Version control | `git --version` |
| Expo Go (App Store / Play Store) | Run the mobile app on your phone | — |
| Either: Ollama (dev/fallback) | Local CPU LLM | `ollama --version` |
| Or: Qualcomm AI Hub / GenieX SDK (venue path) | On-NPU LLM inference | `geniex --version` |

```bash
git clone <this-repo-url> coderx
cd coderx

cd backend
pip install -r requirements.txt --break-system-packages   # or use a virtualenv
cd ..

cd mobile
npm install
cd ..

chmod +x setup.sh
./setup.sh
```

`setup.sh` installs `hooks/post-commit` into `.git/hooks/post-commit`,
starts the FastAPI webhook listener on port 8000 (which also starts the
WebSocket server on port 8765), and starts the Expo dev server on a fixed
port, logging its QR code to `logs/coderx_expo.log`. Safe to re-run
anytime.

## 15. Snapdragon setup

**CPU/dev fallback (any machine):**
```bash
ollama pull qwen3:4b-instruct
ollama serve
export CODERX_BACKEND=ollama
```

**Snapdragon X Elite (venue path):**
1. Install the Qualcomm AI Hub / QAIRT SDK, confirm `geniex --version`
   works.
2. Download/compile Qwen3-4B-Instruct-2507 for your device via Qualcomm
   AI Hub.
3. No extra env var needed — `qnn` is the default `CODERX_BACKEND`.

Then point the mobile app at your machine: edit `mobile/App.js` and set
`SERVER_IP` to your laptop's LAN IP (phone and laptop must be on the same
Wi-Fi).

## 16. Demo

`backend/demo.py` is a deterministic, judge-facing demo: it builds a
scratch git repo seeded from this project's own known-vulnerable sample
(`samples/buggy_auth.py`, not invented for the demo), makes commits chosen
to hit all three routing tiers, and runs the **real** pipeline against
them twice — the second run against the same commit demonstrates the
cache-hit behavior directly rather than asserting it.

```bash
cd backend
CODERX_MOCK_LLM=1 python demo.py              # pipeline mechanics, no LLM required
python demo.py                                 # real inference, needs geniex or ollama
python demo.py --via-webhook                   # exercises the webhook path instead of the CLI path
```

Trigger a review three other ways: an automatic real commit
(`git commit`), a simulated GitHub PR (`curl -X POST .../webhook`), or the
manual CLI (`python run_review.py`) — see `backend/run_review.py --help`
for flags.

## 17. Screenshots

<p align="center"><img src="docs/mobile.png" width="320"></p>

*Mobile triage app — the developer reviews findings, approves valid
issues, or marks false positives, then taps Generate Report.*

*Offline report summarizing approved findings and AI re-evaluation
results.*

[Watch the full demo video](https://drive.google.com/file/d/1WPevNqbA86IYdafQGsBEq2PT2Y9U9qpy/view?usp=sharing)

## 18. Technical details

**Tech stack**

| Component | Technology |
|---|---|
| LLM model | Qwen3-4B-Instruct-2507 — 4096-token context, 1,301 tok/s prefill, 23.1 tok/s decode (measured) |
| LLM runtime (production) | `geniex infer` — Qualcomm AI Hub's GenieX CLI, one-shot per hunk, on-NPU |
| LLM runtime (dev/fallback) | Ollama (`qwen3:4b-instruct`), CPU-only |
| Backend | Python 3.10+, FastAPI, `websockets` |
| Git/PR trigger | Shell hook (`hooks/post-commit`) → FastAPI webhook listener (HMAC-verified) |
| Mobile app | React Native (Expo) — display/triage only, zero on-device inference |
| Report generation | fpdf2 (pure Python, no native deps) + plain text + structured JSON |
| Cache | SQLite, `backend/coderx_cache.db` |
| Structured logging | `devlog.py` — console + persistent commit-tagged log file |

**Repository structure**

```
coderx/
├── backend/     Review pipeline, risk router, cache, WebSocket server,
│                webhook listener, PDF/JSON report generation, tests/
├── mobile/      React Native (Expo) triage app
├── hooks/       post-commit git hook
├── samples/     demo buggy files used for live-review testing
├── setup.sh     one-command install + hook install + start everything
├── README.md
└── LICENSE      MIT
```

**Full environment variable reference**

| Variable | Purpose | Default |
|---|---|---|
| `CODERX_BACKEND` | `qnn` (NPU) or `ollama` (CPU dev/fallback) | `qnn` |
| `CODERX_MODEL` | Ollama model tag | `qwen3:4b-instruct` |
| `CODERX_GENIE_MODEL` | GenieX model identifier | `ai-hub-models/Qwen3-4B-Instruct-2507` |
| `CODERX_GENIE_COMPUTE` | GenieX compute target | `npu` |
| `CODERX_GENIE_THINK` | GenieX thinking-mode toggle | `false` |
| `CODERX_MOCK_LLM` | `1` = bypass LLM with a canned response | `0` |
| `CODERX_TIMEOUT` | LLM call timeout, seconds | `420` |
| `CODERX_LOG_LEVEL` | `DEBUG`/`INFO`/`WARNING`/`ERROR` | `INFO` |
| `CODERX_LOG_DIR` | Where `coderx.log` is written | `./logs` |
| `GITHUB_TOKEN` | Avoids GitHub's 60 req/hr unauthenticated rate limit | unset |
| `CODERX_WEBHOOK_SECRET` | GitHub webhook HMAC secret — verifies `X-Hub-Signature-256`. **Unset = verification skipped (logged loudly)** | unset |
| `CODERX_FAST_PATH` | `1` = skip NPU for structurally-trivial hunks | `1` |
| `CODERX_TRIVIAL_MAX_LINES` | Size-cap safeguard before a trivial hunk is skipped | `25` |
| `CODERX_DEBUG_DUMP` | `1` = write full diff + raw LLM output to disk for debugging | `0` |

**Testing:**
```bash
cd backend
python tests/test_pipeline.py -v   
```

## 19. Limitations

- **Confidence is not calibrated.** Qwen3-4B-Instruct-2507 is not prompted
  for a confidence score, and GenieX's one-shot CLI exposes no token
  log-probs to compute one post-hoc — every `Finding.confidence` is
  explicitly `None` with a note explaining why, never a fabricated number.
- **Mobile WebSocket has no per-device auth/pairing** — any client that
  can reach port 8765 can currently connect and see findings.
- **Risk-signal matching scans full hunk text, including unchanged
  context lines** (see section 7) — conservative, but means routing isn't
  literally "changed lines only."
- **No message backlog/replay** — a mobile client that connects after a
  review already ran will miss those findings.
- **Real GitHub PR diffs use GitHub's default 3-line context**, not the
  full-function context local commits get — GitHub's diff API has no
  equivalent option.
- **Single review session at a time, in-memory** — a process restart
  loses in-flight (not-yet-reported) decisions; a completed report (PDF +
  TXT + JSON) does persist to disk.
- **Malicious-diff hardening is untested** — parser robustness against a
  diff specifically crafted to break `response_parser.py` hasn't been
  adversarially tested yet.

## 20. Future work

- Multi-developer review sessions (persisted, not single in-memory session)
- Per-device mobile WebSocket authentication/pairing
- Persistent review history across restarts
- VS Code / JetBrains extension for in-editor triage
- `coderx benchmark` as a single wrapped CLI command around the existing
  `benchmark.py` harness
- Support for larger NPU-optimized models as Snapdragon NPU headroom grows
- Message backlog/replay for mobile clients that connect mid-review
- Adversarial testing against malicious/malformed diffs

## References

- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — model benchmarking and
  compilation for Snapdragon NPUs
- [Qualcomm AI Hub documentation](https://app.aihub.qualcomm.com/docs/)
- [GenieX / QAIRT documentation](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)
- [Qwen3 model family](https://github.com/QwenLM/Qwen3)
- [Qwen3 technical report](https://arxiv.org/abs/2505.09388)
- [Ollama](https://ollama.com)
- [fpdf2](https://github.com/py-pdf/fpdf2)
- [Expo](https://expo.dev) / [React Native](https://reactnative.dev)

## License

Apache-2.0 license — see [LICENSE](LICENSE).
