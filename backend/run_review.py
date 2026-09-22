"""
run_review.py

Orchestrator — the "does the pipeline work at all" script.

false-positive reiteration flow : findings are
now registered into review_session (which assigns each one a stable id and
retains its source diff, server-side only) BEFORE being broadcast, so
mobile-side decisions and the eventual reiteration pass have something to
reference. Report generation is no longer automatic at the end of this
script — it's now triggered by mobile sending {"type": "generate_report"}
once every finding has been marked approved/false_positive (see
ws_broadcaster.py + report_trigger.py). This script's job ends once findings
are broadcast; it just keeps the process (and WS server) alive after that.

UPDATED for structured logging : every phase is now
logged (not just printed) and persisted to logs/coderx.log, with major
blocking steps (git subprocess calls, per-hunk LLM review) wrapped in
stage() so a hang anywhere shows up as a "STAGE START" with no matching
"STAGE END" instead of the terminal just going silent.
"""

import argparse
import time
from collections import defaultdict

from diff_extractor import get_last_commit_diff, get_diff, split_into_hunks, get_commit_info
from prompt_builder import build_prompt
import llm_client
from response_parser import parse_findings, Finding
from ws_broadcaster import broadcast_findings
from ws_broadcaster import _start_server_thread
import review_session
import review_cache
from risk_router import classify, sort_by_priority, RiskTier, RoutingSummary
from devlog import get_logger, stage

log = get_logger(__name__)


def main():
    _start_server_thread()
    parser = argparse.ArgumentParser(description="CoderX skeleton pipeline runner")
    parser.add_argument("--repo", default=".", help="Path to the git repo to review")
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Review staged changes instead of the last commit",
    )
    args = parser.parse_args()

    try:
        with stage(log, "fetch_commit_info"):
            commit_info = get_commit_info(args.repo, staged=args.staged)
    except RuntimeError as e:
        log.error(f"{e}")
        return

    review_session.session.start_new_session(commit_info=commit_info)
    log.info(f"Reviewing commit {commit_info.short_id} by {commit_info.author_name}: "
              f"\"{commit_info.message}\"")

    log.info(f"Extracting diff from: {args.repo}")
    try:
        with stage(log, "extract_diff"):
            if args.staged:
                raw_diff = get_diff(args.repo, staged=True)
            else:
                raw_diff = get_last_commit_diff(args.repo)
    except RuntimeError as e:
        log.error(f"{e}")
        return

    hunks = split_into_hunks(raw_diff)
    log.info(f"Found {len(hunks)} hunk(s) to review")

    if not hunks:
        log.warning("No hunks found — nothing to review (empty diff, or no commits yet).")
        return

    all_findings = defaultdict(list)
    total_latency = 0.0
    npu_calls_made = 0
    summary = RoutingSummary()

    # Route + priority-sort BEFORE any LLM calls: every hunk is classified
    # on-device (CRITICAL_PATH / STANDARD / TRIVIAL) so the highest-risk
    # change is reviewed and surfaced to mobile first, and comment/
    # whitespace-only hunks never touch the NPU at all. See risk_router.py.
    routed = [(hunk, classify(hunk)) for hunk in hunks]
    routed = sort_by_priority(routed)

    for i, (hunk, decision) in enumerate(routed, start=1):
        log.info(
            f"Reviewing hunk {i}/{len(hunks)}: {hunk.file_path} (line {hunk.start_line}) "
            f"[{decision.tier.value}] {decision.reason}"
        )

        was_cache_hit = False
        if decision.skip_llm:
            # TRIVIAL: no semantic change, no NPU call, no cache write —
            # a synthetic SUGGESTION finding keeps the triage/report trail
            # consistent so the developer can see *why* nothing was flagged.
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
            elapsed = 0.0
        else:
            cached = review_cache.lookup(hunk.diff_text)
            if cached is not None:
                was_cache_hit = True
                findings = cached
                elapsed = 0.0
            else:
                prompt = build_prompt(hunk)
                start = time.time()
                with stage(log, f"llm_review(hunk {i}/{len(hunks)}, {hunk.file_path})"):
                    result = llm_client.review_hunk(prompt)
                elapsed = time.time() - start
                total_latency += elapsed
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

        summary.record(decision, was_cache_hit=was_cache_hit)

        # Register each finding into the session BEFORE broadcasting — this
        # assigns finding.id (mutated in place) and retains the hunk's diff
        # text server-side, so a later false-positive decision can trigger
        # a reiteration call with real code context. Order matters here:
        # broadcast_findings() below needs finding.id already set.
        for f in findings:
            review_session.session.add(f, hunk.diff_text)

        all_findings[hunk.file_path].extend(findings)

        log.info(f"    -> {len(findings)} finding(s), {elapsed:.2f}s")

    if npu_calls_made:
        avg_latency = total_latency / npu_calls_made
        log.info(
            f"Total NPU latency: {total_latency:.2f}s across {npu_calls_made} real call(s) "
            f"(avg {avg_latency:.2f}s/call)"
        )
    else:
        avg_latency = 0.0
    routing_report = summary.render(commit_info.short_id, avg_npu_latency_seconds=avg_latency)
    log.info(routing_report)
    print(routing_report)

    for file_path, findings in all_findings.items():
        if findings:
            broadcast_findings(findings, file_path)

    total_findings = sum(len(v) for v in all_findings.values())
    if total_findings:
        log.info(f"{total_findings} finding(s) broadcast. Waiting for mobile to mark "
                  f"each as approved/false_positive, then send 'generate_report'.")
    else:
        log.info("No findings — nothing for mobile to review.")

    print("\n[run_review] Keeping websocket alive. Press Enter to stop.")
    input()


if __name__ == "__main__":
    main()
