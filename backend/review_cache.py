"""
review_cache.py

Incremental review cache (closes the "Incremental review caching" item
that was previously listed under Future Work).

Problem: every commit re-sends every hunk to the NPU, even hunks whose
code is byte-identical to something already reviewed in a prior commit
(a common case: a file gets touched in a commit but a given function's
diff hunk didn't actually change, e.g. a rename, a reformat elsewhere in
the file, or the same fix reapplied after a revert). On-NPU decode is the
slowest stage of the pipeline (~23 tok/s measured on Snapdragon X Elite —
see benchmark.py), so skipping redundant hunks is a direct, measurable
latency + battery/thermal win, not a cosmetic one.

Design:
  - Key = sha256(normalized hunk diff_text). Normalization strips diff
    line-number headers (@@ ... @@) so a hunk that shifted a few lines
    because of an earlier edit in the same file still hits the cache if
    the actual changed code is identical.
  - Value = the parsed Finding objects from the last time that exact hunk
    was reviewed, plus metadata (model id, timestamp, commit it was first
    seen in).
  - Storage: a single local SQLite file (backend/coderx_cache.db). No
    network, no server dependency — consistent with the "nothing leaves
    the device" design.
  - Cache entries are invalidated automatically if CODERX_GENIE_MODEL (or
    the Ollama model tag) changes, so switching models never serves stale
    verdicts from a different model.

This module never talks to the LLM itself — run_review.py checks it
before calling review_hunk() and writes to it after.
"""

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from response_parser import Finding
import prompt_builder
from devlog import get_logger

log = get_logger(__name__)

_DB_PATH = Path(__file__).parent / "coderx_cache.db"

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@.*$", re.MULTILINE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_cache (
    hunk_hash   TEXT NOT NULL,
    model_key   TEXT NOT NULL,
    findings_json TEXT NOT NULL,
    first_seen_commit TEXT,
    hit_count   INTEGER DEFAULT 0,
    created_at  REAL,
    last_hit_at REAL,
    PRIMARY KEY (hunk_hash, model_key)
);
"""


def _current_model_key() -> str:
    """Cache is namespaced by (backend/model, prompt template version) so a
    verdict is only ever reused if BOTH the model AND the exact instruction
    wording that produced it are unchanged. Changing either one — swapping
    CODERX_GENIE_MODEL, or editing PROMPT_TEMPLATE in prompt_builder.py —
    produces a different key, so stale verdicts are never silently served."""
    backend = os.getenv("CODERX_BACKEND", "qnn")
    if backend == "ollama":
        base = f"ollama:{os.getenv('CODERX_MODEL', 'qwen3:4b-instruct')}"
    else:
        base = f"qnn:{os.getenv('CODERX_GENIE_MODEL', 'ai-hub-models/Qwen3-4B-Instruct-2507')}"
    return f"{base}@prompt-{prompt_builder.PROMPT_TEMPLATE_VERSION}"


def _normalize(diff_text: str) -> str:
    """Strips line-number headers so line-shifted-but-identical hunks still match."""
    return _HUNK_HEADER_RE.sub("", diff_text).strip()


def _hunk_hash(diff_text: str) -> str:
    return hashlib.sha256(_normalize(diff_text).encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def lookup(diff_text: str) -> Optional[List[Finding]]:
    """Returns cached Finding objects for this exact hunk content, or None on a miss."""
    h = _hunk_hash(diff_text)
    model_key = _current_model_key()
    with _connect() as conn:
        row = conn.execute(
            "SELECT findings_json FROM review_cache WHERE hunk_hash = ? AND model_key = ?",
            (h, model_key),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE review_cache SET hit_count = hit_count + 1, last_hit_at = ? "
            "WHERE hunk_hash = ? AND model_key = ?",
            (time.time(), h, model_key),
        )
    findings_data = json.loads(row[0])
    findings = [Finding(**f) for f in findings_data]
    for f in findings:
        # Re-tag provenance: this specific run served it from cache, even
        # though it was originally produced by an actual NPU/CPU call —
        # inference_path must reflect what happened THIS run, not the run
        # that first produced the verdict.
        f.inference_path = "cache"
    log.info(f"[cache] HIT  {h[:12]} ({len(findings)} finding(s) reused, no NPU call)")
    return findings


def store(diff_text: str, findings: List[Finding], commit_id: str = "") -> None:
    """Persists this hunk's verdict so an identical future hunk skips the LLM."""
    h = _hunk_hash(diff_text)
    model_key = _current_model_key()
    payload = json.dumps([asdict(f) for f in findings])
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO review_cache (hunk_hash, model_key, findings_json, "
            "first_seen_commit, hit_count, created_at, last_hit_at) "
            "VALUES (?, ?, ?, ?, 0, ?, ?) "
            "ON CONFLICT(hunk_hash, model_key) DO UPDATE SET "
            "findings_json = excluded.findings_json, last_hit_at = excluded.last_hit_at",
            (h, model_key, payload, commit_id, now, now),
        )
    log.info(f"[cache] STORE {h[:12]} ({len(findings)} finding(s))")


def stats() -> dict:
    """Summary used by benchmark.py / the report to show cache effectiveness."""
    with _connect() as conn:
        total, hits = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM review_cache"
        ).fetchone()
    return {"cached_hunks": total, "total_cache_hits": hits}


def invalidate_model(model_key: Optional[str] = None) -> int:
    """Drops all cache entries for one model/prompt-version key (defaults to
    the currently configured one). Use this after a deliberate prompt or
    model change if you want a clean slate rather than relying on the
    automatic key-mismatch behavior (which just leaves old rows unused
    rather than deleting them). Returns the number of rows removed."""
    key = model_key or _current_model_key()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM review_cache WHERE model_key = ?", (key,))
        removed = cur.rowcount
    log.info(f"[cache] Invalidated {removed} entr(y/ies) for model_key={key}")
    return removed


def invalidate_all() -> int:
    """Drops the entire cache. Deterministic, explicit, on-device — no
    background TTL/expiry logic, since a hunk's review verdict for a fixed
    (model, prompt) pair never goes 'stale' on its own; it's the developer's
    call when to reset it (e.g. before a benchmark run, or after suspecting
    a bad cached verdict)."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM review_cache")
        removed = cur.rowcount
    log.info(f"[cache] Invalidated ALL {removed} entr(y/ies)")
    return removed


if __name__ == "__main__":
    import sys
    if "--clear" in sys.argv:
        n = invalidate_all()
        print(f"Cleared {n} cache entr(y/ies).")
    elif "--stats" in sys.argv:
        print(stats())
    else:
        print("Usage: python review_cache.py [--stats | --clear]")
        print(f"Current model/prompt key: {_current_model_key()}")
        print(stats())
