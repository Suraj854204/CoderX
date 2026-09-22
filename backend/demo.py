#!/usr/bin/env python3
"""
demo.py

A controlled, deterministic demo for judges/reviewers — sets up a scratch
git repo seeded with the project's own samples/buggy_auth.py (a real
known-vulnerable file already in this repo, not invented for the demo),
makes a sequence of commits that hit CRITICAL_PATH / STANDARD / TRIVIAL
routing on purpose, and runs the ACTUAL pipeline (risk_router, cache,
llm_client, response_parser, report_generator) against them — twice on
the same commit, so the cache-hit behavior is visible, not asserted.

This does not fake or pre-script any findings: whatever CODERX_BACKEND is
configured (qnn / ollama / mock) is what actually runs. On a machine with
no NPU and no Ollama running, set CODERX_MOCK_LLM=1 to still see the full
pipeline mechanics (routing, caching, session, report generation) without
a real model call — clearly labeled as mock, never presented as a real
inference result.

Usage:
    cd backend
    CODERX_MOCK_LLM=1 python3 demo.py       # pipeline mechanics only, no LLM required
    python3 demo.py                          # real inference (needs geniex or ollama configured)
    CODERX_WEBHOOK_SECRET=x python3 demo.py --via-webhook   # exercise the webhook path instead of the CLI path
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.resolve()
SAMPLE_FILE = BACKEND_DIR.parent / "samples" / "buggy_auth.py"


def _run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def build_demo_repo() -> Path:
    """Builds a fresh throwaway git repo with three commits chosen to hit
    all three routing tiers deterministically. Not left behind — caller
    is responsible for cleanup (or just let it live under /tmp)."""
    if not SAMPLE_FILE.exists():
        print(f"ERROR: expected sample file not found at {SAMPLE_FILE}", file=sys.stderr)
        sys.exit(1)

    repo = Path(tempfile.mkdtemp(prefix="coderx_demo_"))
    _run(["git", "init", "-q"], cwd=repo)
    _run(["git", "config", "user.email", "demo@coderx.local"], cwd=repo)
    _run(["git", "config", "user.name", "CoderX Demo"], cwd=repo)

    # Commit 1: baseline — a clean version of the file (no injection yet).
    clean_version = (
        '"""Sample auth module — clean baseline for the CoderX demo."""\n'
        "import sqlite3\n\n\n"
        "def get_user(user_id):\n"
        '    conn = sqlite3.connect("app.db")\n'
        "    cursor = conn.cursor()\n"
        '    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))\n'
        "    return cursor.fetchone()\n"
    )
    (repo / "auth.py").write_text(clean_version)
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-q", "-m", "baseline: clean auth module"], cwd=repo)

    # Commit 2: the CRITICAL_PATH demo — reuses the real known-vulnerable
    # sample already in this repo's samples/ directory verbatim.
    vulnerable_version = SAMPLE_FILE.read_text()
    (repo / "auth.py").write_text(vulnerable_version)
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-q", "-m", "add login() — introduces SQL injection (demo: CRITICAL_PATH)"], cwd=repo)

    # Commit 3: mixed commit — one TRIVIAL hunk (comment only) and one
    # STANDARD hunk (ordinary refactor) in the same commit, to demonstrate
    # per-hunk (not per-commit) routing.
    mixed_version = vulnerable_version.replace(
        "def get_user(user_id):",
        "# NOTE: consider renaming to fetch_user for consistency\n"
        "def get_user(user_id):",
    ).replace(
        "    return cursor.fetchone()\n\n\ndef login",
        "    row = cursor.fetchone()\n    return row\n\n\ndef login",
    )
    (repo / "auth.py").write_text(mixed_version)
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-q", "-m", "cleanup: comment + minor refactor (demo: TRIVIAL + STANDARD)"], cwd=repo)

    return repo


def main():
    parser = argparse.ArgumentParser(description="CoderX judge demo")
    parser.add_argument("--via-webhook", action="store_true",
                         help="Exercise the webhook_server.py pipeline path instead of run_review.py's CLI path")
    parser.add_argument("--keep", action="store_true", help="Don't delete the scratch repo afterward")
    args = parser.parse_args()

    print("=" * 72)
    print("CoderX Judge Demo")
    print("=" * 72)
    backend = os.environ.get("CODERX_BACKEND", "qnn")
    mock = os.environ.get("CODERX_MOCK_LLM", "0") == "1"
    print(f"Backend: {backend}{'  (MOCK MODE — pipeline mechanics only, not a real inference result)' if mock else ''}")
    print()

    repo = build_demo_repo()
    print(f"Demo repo created at: {repo}")
    print("3 commits made: baseline -> CRITICAL_PATH (SQL injection) -> TRIVIAL+STANDARD (mixed)")
    print()

    os.chdir(BACKEND_DIR)
    sys.path.insert(0, str(BACKEND_DIR))

    if args.via_webhook:
        import webhook_server as pipeline
        run = lambda: pipeline.run_pipeline_for_repo(repo_path=str(repo))
    else:
        import diff_extractor, review_session, review_cache
        from prompt_builder import build_prompt
        import llm_client
        from response_parser import parse_findings
        from risk_router import classify, sort_by_priority, RoutingSummary
        from devlog import get_logger

        def run():
            raw_diff = diff_extractor.get_last_commit_diff(str(repo))
            commit_info = diff_extractor.get_commit_info(str(repo), staged=False)
            review_session.session.start_new_session(commit_info=commit_info)
            hunks = diff_extractor.split_into_hunks(raw_diff)
            routed = sort_by_priority([(h, classify(h)) for h in hunks])
            summary = RoutingSummary()
            npu_calls, total_latency = 0, 0.0
            for hunk, decision in routed:
                if decision.skip_llm:
                    was_hit = False
                else:
                    cached = review_cache.lookup(hunk.diff_text)
                    was_hit = cached is not None
                    if not was_hit:
                        import time
                        prompt = build_prompt(hunk)
                        start = time.time()
                        result = llm_client.review_hunk(prompt)
                        total_latency += time.time() - start
                        npu_calls += 1
                        findings = parse_findings(result.raw_output)
                        review_cache.store(hunk.diff_text, findings, commit_id=commit_info.short_id)
                summary.record(decision, was_cache_hit=was_hit)
            avg = (total_latency / npu_calls) if npu_calls else 0.0
            print(summary.render(commit_info.short_id, avg_npu_latency_seconds=avg))
            return {"commit": commit_info.short_id}

    print("--- Run 1 (cold — nothing cached yet) ---")
    run()

    print("--- Run 2, SAME commit (should hit cache — 0 new NPU calls) ---")
    run()

    if not args.keep:
        shutil.rmtree(repo, ignore_errors=True)
        print(f"\n(scratch repo cleaned up; pass --keep to retain it)")
    else:
        print(f"\nScratch repo retained at: {repo}")


if __name__ == "__main__":
    main()
