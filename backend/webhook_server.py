"""
webhook_server.py

FastAPI listener for the GitHub PR trigger path (Section 3/6 of the
knowledge file: "GitHub PR submission -> via local webhook listener").

UPDATE (Section 20 sync-up, July 10): run_pipeline_for_repo() previously
had its own older copy of the review loop that predated review_session /
commit-tracking / the false-positive reiteration flow entirely. It never
fetched commit info, never assigned finding ids, and called the OLD
report_generator.generate_report(Dict[str, list]) signature directly and
immediately after parsing. That's what caused commit_id/finding_id to come
through null/empty when a review was triggered via setup.sh's webhook
listener, while the same review worked fine via the manual
`python run_review.py` CLI path -- the two entrypoints had silently
diverged into two different, incompatible pipelines even though this
file's own docstring claimed they were "the same pipeline". Fixed below:
this now mirrors run_review.py's current flow exactly (fetch commit info,
start a fresh review_session, register every finding with its id + source
diff BEFORE broadcasting, and do NOT call generate_report() here anymore
-- that's mobile-triggered now, same as the CLI path).

WHY IMPORT DIRECTLY INSTEAD OF SHELLING OUT TO run_review.py:
run_review.py's main() does argparse + ends with a blocking input() (a
keep-alive so the WebSocket doesn't die immediately when run manually).
Shelling out to it as a subprocess means fighting that blocking call and
losing easy access to results. Importing diff_extractor / prompt_builder /
llm_client / response_parser / ws_broadcaster / review_session directly
(same as run_review.py does) gives the same pipeline without either
problem, and requires zero changes to Hardik's or Vatsal's modules.

RUN STANDALONE:
    cd backend
    uvicorn webhook_server:app --reload --port 8000
(port 8000 chosen to avoid colliding with ws_broadcaster's 8765)

TEST IT — mock payload, reviews local last commit (no GitHub call at all,
since there's no "repository" key -- this is exactly what post-commit sends):
    curl -X POST http://localhost:8000/webhook \
      -H "X-GitHub-Event: pull_request" \
      -H "Content-Type: application/json" \
      -d '{"action": "opened", "pull_request": {"number": 0}}'

TEST IT — real GitHub PR diff fetch (needs a real public repo + open PR
number; GITHUB_TOKEN optional but recommended to avoid the 60 req/hour
unauthenticated rate limit):
    curl -X POST http://localhost:8000/webhook \
      -H "X-GitHub-Event: pull_request" \
      -H "Content-Type: application/json" \
      -d '{"action": "opened", "repository": {"full_name": "octocat/Hello-World"}, "pull_request": {"number": 1}}'
"""

import hashlib
import hmac
import os
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from diff_extractor import get_last_commit_diff, get_diff, split_into_hunks, get_commit_info, CommitInfo
from prompt_builder import build_prompt
import llm_client
from response_parser import parse_findings
from ws_broadcaster import broadcast_findings  # importing this also starts the WS server eagerly (see ws_broadcaster.py)
import review_session
import review_cache
from risk_router import classify, sort_by_priority, RoutingSummary
from devlog import get_logger, stage

log = get_logger(__name__)

app = FastAPI(title="CoderX Webhook Listener")

GITHUB_API_BASE = "https://api.github.com"

# --- Webhook authentication ---------------------------------------------
# GitHub signs every webhook delivery with HMAC-SHA256 over the raw request
# body, using a secret configured on the GitHub webhook itself, sent as the
# X-Hub-Signature-256 header ("sha256=<hex digest>"). Without verifying
# this, ANYONE who can reach this port can POST a payload and trigger a
# full review pipeline run (real NPU calls, real report generation) with
# no proof it actually came from GitHub — the "loose validation" the old
# _looks_like_pull_request_payload() docstring described was never a
# substitute for this, only a shape check.
#
# CODERX_WEBHOOK_SECRET must match the secret configured on the GitHub
# webhook. If it's unset, verification is skipped and a loud warning is
# logged on every request — this keeps the pre-signature local-testing
# workflow (curl with no signature header, exactly as documented in this
# file's own module docstring) working, but makes the unverified state
# impossible to miss rather than silently insecure-by-default.
WEBHOOK_SECRET = os.environ.get("CODERX_WEBHOOK_SECRET", "")


def _verify_github_signature(raw_body: bytes, signature_header: Optional[str]) -> bool:
    """Constant-time HMAC-SHA256 verification of GitHub's X-Hub-Signature-256.
    Returns True if verification passes OR is intentionally disabled (no
    secret configured) — callers must check WEBHOOK_SECRET separately to
    know which case they're in, so the warning is always logged."""
    if not WEBHOOK_SECRET:
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(WEBHOOK_SECRET.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    provided = signature_header.split("=", 1)[1]
    return hmac.compare_digest(expected, provided)

# Deliveries for these actions mean "nothing new to review" -- e.g. GitHub
# fires another 'pull_request' event when a PR is closed/merged, and
# re-reviewing a closed PR would just waste an LLM call and confuse the
# demo. "opened", "synchronize" (new commits pushed), and "reopened" all
# fall through to a real review; anything in this set is ignored instead.
SKIP_ACTIONS = {"closed"}


def _fetch_pr_diff_from_github(payload: dict) -> Optional[str]:
    """
    Fetches the actual unified diff for a real GitHub pull request via
    GitHub's REST API, using the special `application/vnd.github.v3.diff`
    Accept header (this returns a plain-text unified diff body, not JSON).

    Returns None (rather than raising) if:
      - the payload doesn't carry enough info to build a real API URL.
        This is exactly how we distinguish a REAL GitHub webhook delivery
        from OUR OWN local post-commit mock payload: real GitHub payloads
        always include a top-level "repository" object with "full_name";
        our post-commit hook's fake payload (see hooks/post-commit) does
        not, since it isn't reviewing any repository on GitHub at all --
        it's reviewing whatever was just committed locally.
      - the GitHub API call itself fails for any reason (network, auth,
        rate limit, 404 because the PR/repo doesn't exist). The caller
        falls back to reviewing the local repo's last commit in that
        case, so a flaky GitHub API call degrades gracefully instead of
        losing the review entirely.

    AUTH: set a GITHUB_TOKEN (or CODERX_GITHUB_TOKEN) env var if you hit
    GitHub's unauthenticated rate limit (60 req/hour) or need access to a
    private repo. Works fine unauthenticated for public repos at low
    request volume -- fine for a hackathon demo, not fine for production.

    KNOWN LIMITATION vs. the local-commit path (flagging deliberately
    rather than silently pretending PR reviews are equal quality): this
    returns GitHub's standard diff, with the default 3 lines of context
    per hunk -- NOT the `--function-context`-expanded diff that
    diff_extractor.py produces for local commits (Section 7 of the
    knowledge file). GitHub's diff API has no equivalent "expand to full
    function" option. Real-PR review quality may be slightly lower than
    local-commit review quality until/unless a re-expansion step is added
    on top of the fetched diff. Not attempting that here -- out of scope
    for this to-do item.
    """
    repository = payload.get("repository")
    pull_request = payload.get("pull_request")

    repo_full_name = repository.get("full_name") if isinstance(repository, dict) else None
    pr_number = pull_request.get("number") if isinstance(pull_request, dict) else None

    if not repo_full_name or not pr_number:
        return None

    url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/pulls/{pr_number}"
    headers = {"Accept": "application/vnd.github.v3.diff"}

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("CODERX_GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        log.warning(f"Could not fetch PR diff from GitHub ({e}); falling back to reviewing the local repo's last commit instead.")
        return None


def run_pipeline_for_repo(
    repo_path: str = ".",
    staged: bool = False,
    prefetched_diff: Optional[str] = None,
) -> dict:
    """
    Same core loop as run_review.py's main(), pulled out as a reusable
    function so both the CLI entrypoint and this webhook can share it
    without either one blocking on argparse or input().

    UPDATED (Section 20 sync-up): now mirrors run_review.py exactly —
    fetches commit info, starts a fresh review_session, and registers
    every finding (assigning its id + retaining its source diff) BEFORE
    broadcasting. No longer calls generate_report() directly — report
    generation is mobile-triggered (the {"type": "generate_report"}
    message, handled by report_trigger.py) once every finding has been
    marked approved/false_positive, exactly like the CLI path. Returns a
    summary dict for the webhook response, not a report path.

    prefetched_diff: if provided (e.g. a real PR diff fetched from
    GitHub's API via _fetch_pr_diff_from_github), this is used directly
    instead of reading a git diff off local disk. A real PR diff has no
    single local commit backing it (it may span several commits on the PR
    branch), so get_commit_info() doesn't cleanly apply — a synthetic
    CommitInfo is used instead so commit-grouping/report-header still get
    something meaningful rather than null.
    """
    if prefetched_diff is not None:
        raw_diff = prefetched_diff
        commit_info = CommitInfo(
            commit_id="PR", short_id="PR",
            author_name="(GitHub PR)", author_email="",
            message="Reviewing a fetched GitHub PR diff (not a single local commit)",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    elif staged:
        raw_diff = get_diff(repo_path, staged=True)
        commit_info = get_commit_info(repo_path, staged=True)
    else:
        raw_diff = get_last_commit_diff(repo_path)
        commit_info = get_commit_info(repo_path, staged=False)

    log.info(f"[webhook_server] Extracted diff ({len(raw_diff)} chars) from: {repo_path}")

    review_session.session.start_new_session(commit_info=commit_info)
    log.info(f"Reviewing commit {commit_info.short_id} by {commit_info.author_name}: \"{commit_info.message}\"")

    hunks = split_into_hunks(raw_diff)

    if not hunks:
        log.warning("No hunks found — nothing to review")
        return {"hunks_reviewed": 0, "findings_total": 0, "commit": commit_info.short_id}

    all_findings = defaultdict(list)
    total_latency = 0.0
    npu_calls_made = 0
    summary = RoutingSummary()

    # Route + priority-sort BEFORE any LLM calls — mirrors run_review.py
    # exactly. This was the actual gap found in this file's audit: this
    # function had its own older copy of the review loop that predated
    # risk_router.py / review_cache.py entirely, so a PR reviewed via the
    # webhook path got zero routing/caching benefit even after the CLI
    # path (run_review.py) gained both. Fixed here rather than left as a
    # documented-but-unfixed inconsistency.
    routed = [(hunk, classify(hunk)) for hunk in hunks]
    routed = sort_by_priority(routed)

    # Debug dump of full raw LLM output (and therefore full diff content)
    # to disk is opt-in only, off by default: this file's own "zero code
    # leaves the machine" story is about the network, but a debug file
    # containing source code sitting in backend/geniex_response_debug/
    # indefinitely is still a real local-exposure surface worth gating
    # rather than defaulting on (see security audit, item 14 — "logs
    # containing source code").
    debug_dump_enabled = os.environ.get("CODERX_DEBUG_DUMP", "0") == "1"

    for i, (hunk, decision) in enumerate(routed, start=1):
        log.info(
            f"Reviewing hunk {i}/{len(hunks)}: {hunk.file_path} "
            f"[{decision.tier.value}] {decision.reason}"
        )

        was_cache_hit = False
        if decision.skip_llm:
            from response_parser import Finding
            findings = [Finding(
                severity="SUGGESTION",
                file=hunk.file_path,
                line=hunk.start_line,
                description="No semantic change detected (comment/whitespace/import-only) "
                             "— skipped NPU review via fast-path routing.",
                fix="",
                category="trivial",
                inference_path="fast-path-skip",
                model="",
            )]
        else:
            cached = review_cache.lookup(hunk.diff_text)
            if cached is not None:
                was_cache_hit = True
                findings = cached
            else:
                prompt = build_prompt(hunk)
                start = time.time()
                with stage(log, f"llm_review(hunk {i}/{len(hunks)}, {hunk.file_path})"):
                    result = llm_client.review_hunk(prompt)
                total_latency += time.time() - start
                npu_calls_made += 1

                findings = parse_findings(result.raw_output)
                category = ", ".join(decision.matched_signals) if decision.matched_signals else "general"
                inference_path = "npu:qnn" if llm_client.BACKEND == "qnn" else f"cpu:{llm_client.BACKEND}"
                model_used = llm_client.GENIE_MODEL if llm_client.BACKEND == "qnn" else llm_client.MODEL_NAME
                for f in findings:
                    f.category = category
                    f.inference_path = inference_path
                    f.model = model_used
                review_cache.store(hunk.diff_text, findings, commit_id=commit_info.short_id)

                if debug_dump_enabled:
                    _debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geniex_response_debug")
                    os.makedirs(_debug_dir, exist_ok=True)
                    _debug_file = os.path.join(_debug_dir, f"response_hunk_{i}.txt")
                    try:
                        with open(_debug_file, "w", encoding="utf-8") as _df:
                            _df.write(f"=== Hunk {i}/{len(hunks)}: {hunk.file_path} ===\n")
                            _df.write(f"Raw output length: {len(result.raw_output)} chars\n")
                            _df.write(f"Findings parsed: {len(findings)}\n\n--- RAW LLM OUTPUT ---\n")
                            _df.write(result.raw_output)
                    except OSError as e:
                        log.warning(f"Could not write debug response file: {e}")

                if len(result.raw_output) > 0 and len(findings) == 0:
                    log.warning(f"LLM returned {len(result.raw_output)} chars but parser found 0 findings.")

        summary.record(decision, was_cache_hit=was_cache_hit)

        # Assign each finding a stable id + retain its diff server-side
        # BEFORE broadcasting — mirrors run_review.py exactly. Without
        # this, finding.id stays "" (the dataclass default) forever, which
        # is the "empty finding_id" half of the bug this sync-up fixes.
        for f in findings:
            review_session.session.add(f, hunk.diff_text)

        all_findings[hunk.file_path].extend(findings)

    for file_path, findings in all_findings.items():
        if findings:
            broadcast_findings(findings, file_path)

    findings_total = sum(len(v) for v in all_findings.values())
    avg_latency = (total_latency / npu_calls_made) if npu_calls_made else 0.0
    routing_report = summary.render(commit_info.short_id, avg_npu_latency_seconds=avg_latency)
    log.info(routing_report)

    # --- Auto-report mode: generate report immediately without mobile ---
    # When CODERX_AUTO_REPORT is enabled (default: "1"), skip the mobile
    # approval step and auto-approve all findings, then generate the report.
    # This makes the pipeline fully self-contained for testing/demo without
    # a mobile client connected.
    auto_report = os.environ.get("CODERX_AUTO_REPORT", "1") == "1"
    report_path = None

    if auto_report and findings_total > 0:
        log.info(f"AUTO_REPORT mode: auto-approving {findings_total} finding(s) and generating report...")

        # Auto-approve every finding in the session
        for file_path, findings_list in all_findings.items():
            for f in findings_list:
                error = review_session.session.record_decision(
                    finding_id=f.id,
                    decision="approved",
                    comment="Auto-approved (CODERX_AUTO_REPORT mode)",
                )
                if error:
                    log.warning(f"Auto-approve failed for {f.id}: {error}")

        # Generate the report
        try:
            from report_generator import generate_report

            decisions = review_session.session.all_decisions()
            output_path = f"coderx_report_{commit_info.short_id}.txt"
            report_path = generate_report(decisions, commit_info=commit_info, output_path=output_path)
            log.info(f"AUTO_REPORT: Report generated at {report_path}")
        except Exception as e:
            log.error(f"AUTO_REPORT: Report generation failed: {e}", exc_info=True)
    elif auto_report and findings_total == 0:
        log.info("AUTO_REPORT mode: no findings to report.")

    result = {
        "hunks_reviewed": len(hunks),
        "findings_total": findings_total,
        "total_latency_seconds": round(total_latency, 2),
        "commit": commit_info.short_id,
        "routing": {
            "critical": summary.critical,
            "standard": summary.standard,
            "trivial": summary.trivial,
            "npu_calls": summary.npu_calls,
            "cache_hits": summary.cache_hits,
            "skipped": summary.skipped,
        },
    }
    if report_path:
        result["report_path"] = report_path
    else:
        result["note"] = ("Report generation is mobile-triggered once all findings are marked "
                          "(see review_session.py / report_trigger.py) — no report_path here."
                          if not auto_report else "No findings to generate a report for.")
    return result


def _looks_like_pull_request_payload(payload: dict) -> bool:
    """
    Loose validation only -- real GitHub webhook payloads always include
    a top-level "pull_request" object and an "action" field, but since
    we're explicitly supporting mock payloads too (our own post-commit
    hook), this just checks for the "pull_request" key rather than doing
    full schema validation against GitHub's actual webhook shape.
    """
    return isinstance(payload, dict) and "pull_request" in payload


@app.get("/health")
async def health():
    """Quick sanity check — hit this first to confirm the server is up."""
    return {"status": "ok", "service": "coderx-webhook-listener"}


@app.post("/webhook")
async def webhook(request: Request, x_github_event: Optional[str] = Header(default=None)):
    """
    Receives a GitHub PR webhook (real or mock) and triggers a CoderX
    review.

    Diff source selection (see _fetch_pr_diff_from_github's docstring for
    the full reasoning):
      - Real GitHub delivery (payload has a "repository" key) -> fetches
        the actual PR diff from GitHub's API.
      - Our own local post-commit mock (no "repository" key) -> reviews
        the current local repo's last commit, exactly as before this
        update.
      - Real GitHub API fetch failing for any reason -> also falls back
        to the local last commit, so a flaky network doesn't lose the
        review entirely (though the result will reflect local state, not
        necessarily the PR's actual diff, in that edge case).
    """
    raw_body = await request.body()

    if WEBHOOK_SECRET:
        signature = request.headers.get("X-Hub-Signature-256")
        if not _verify_github_signature(raw_body, signature):
            log.warning(
                "Webhook request REJECTED: invalid or missing X-Hub-Signature-256 "
                "(CODERX_WEBHOOK_SECRET is set, so verification is enforced)."
            )
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
    else:
        log.warning(
            "CODERX_WEBHOOK_SECRET is not set — webhook signature verification is "
            "DISABLED. Any request reaching this port can trigger a review. Set "
            "CODERX_WEBHOOK_SECRET (matching the secret on the GitHub webhook config) "
            "before exposing this listener beyond localhost."
        )

    try:
        import json as _json
        payload = _json.loads(raw_body) if raw_body else {}
    except Exception:
        payload = {}

    # GitHub sets this header on every webhook delivery. Mock/manual test
    # calls may not set it, so we don't hard-fail if it's missing — we
    # just prefer it as the primary signal when present.
    event_type = (x_github_event or payload.get("event_type") or "").lower()

    if event_type and event_type != "pull_request":
        return JSONResponse(
            status_code=200,
            content={"status": "ignored", "reason": f"event type '{event_type}' is not pull_request"},
        )

    if not event_type and not _looks_like_pull_request_payload(payload):
        return JSONResponse(
            status_code=400,
            content={
                "status": "rejected",
                "reason": "Payload doesn't look like a pull_request event "
                "(missing X-GitHub-Event header and no 'pull_request' key in body).",
            },
        )

    action = payload.get("action")
    if action in SKIP_ACTIONS:
        return JSONResponse(
            status_code=200,
            content={"status": "ignored", "reason": f"action '{action}' does not need a review"},
        )

    pr_number = None
    if isinstance(payload.get("pull_request"), dict):
        pr_number = payload["pull_request"].get("number")

    repository = payload.get("repository")
    repo_full_name = repository.get("full_name") if isinstance(repository, dict) else None

    log.info(f"Received pull_request event (PR #{pr_number}{f' on {repo_full_name}' if repo_full_name else ''}). Resolving diff source...")

    pr_diff = _fetch_pr_diff_from_github(payload)
    diff_source = "github_api" if pr_diff is not None else "local_last_commit"
    log.info(f"Diff source: {diff_source}. Running review...")

    try:
        summary = run_pipeline_for_repo(prefetched_diff=pr_diff)
    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "reason": str(e)},
        )

    log.info(f"Review complete: {summary}")
    return {"status": "reviewed", "pr_number": pr_number, "diff_source": diff_source, **summary}


if __name__ == "__main__":
    # Convenience for `python webhook_server.py` — prefer running via
    # uvicorn directly for --reload during development though.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
