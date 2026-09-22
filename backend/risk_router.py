"""
risk_router.py

Risk-Aware Review Scheduler.

Sits between diff_extractor and the LLM call and answers, entirely
on-device with zero model calls: "how much does this specific diff
deserve the NPU's attention, and in what order?"

  CRITICAL_PATH - touches an auth/authz/crypto/secrets/SQL/deserialization/
                  shell-exec/filesystem-write/network-boundary/privilege/
                  unsafe-memory surface (signal groups below). ALWAYS
                  reviewed by the LLM, and sorted to the top of the mobile
                  triage queue — never skipped, never cached-only if the
                  cache lookup misses.
  STANDARD      - ordinary logic change. Reviewed normally (cache-eligible).
  TRIVIAL       - the diff is a no-semantic-change edit: comment-only,
                  whitespace/formatting-only, or pure import-reordering,
                  AND small enough that a heuristic false-negative is
                  implausible. Auto-approved with a synthetic SUGGESTION
                  finding, never sent to the NPU, unless CODERX_FAST_PATH=0.

CONSERVATIVE SAFEGUARDS (why a hunk that structurally looks "trivial" can
still be escalated to STANDARD instead of skipped):
  1. Size cap — a hunk over CODERX_TRIVIAL_MAX_LINES changed lines is never
     auto-skipped, even if every changed line matches a comment/blank/
     import pattern. Large mechanical-looking diffs are exactly the case
     where a heuristic is most likely to miss something (e.g. a bulk
     find-and-replace that also touched a string literal used in a SQL
     query). Default cap is intentionally small (25 lines).
  2. Any risk-signal match always wins over a trivial classification,
     checked first — a "# TODO: skip auth check below" comment line still
     routes the hunk to CRITICAL_PATH.
  3. Import-line safety — an import is only treated as no-semantic-change
     if it's a plain `import X` / `from X import Y` with no call expression
     on the same line. This module does not attempt to resolve whether an
     imported symbol is dangerous by name alone (a heuristic-on-a-heuristic
     prone to false confidence). If that matters for your threat model,
     keep CODERX_FAST_PATH=0 in CI and only enable fast-path for local/
     interactive review.
"""

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from diff_extractor import Hunk
from devlog import get_logger

log = get_logger(__name__)


class RiskTier(str, Enum):
    CRITICAL = "CRITICAL_PATH"
    STANDARD = "STANDARD"
    TRIVIAL = "TRIVIAL"


# Signal groups are kept separate (rather than one flat list) so the matched
# category can be surfaced to the developer/report ("touches: auth, sql")
# instead of a bare boolean.
_RISK_SIGNALS = {
    "auth": re.compile(r"\b(auth|login|session|jwt|oauth|password|credential)\b", re.I),
    "authz": re.compile(r"\b(is_admin|has_permission|role\s*==|@requires_role|check_access|authorize)\b", re.I),
    "crypto": re.compile(r"\b(encrypt|decrypt|hashlib|hmac|AES|RSA|secret[_ ]?key|salt)\b", re.I),
    "secrets": re.compile(r"\b(api[_-]?key|access[_-]?token|private[_-]?key|BEGIN (RSA|EC) PRIVATE KEY)\b", re.I),
    "sql": re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE)\b.*\b(FROM|INTO|WHERE)\b|\bexecute\s*\(", re.I),
    "deserialization": re.compile(r"\b(pickle\.loads?|yaml\.load\(|eval\(|exec\(|marshal\.loads?)\b"),
    "shell_exec": re.compile(r"\b(subprocess\.|os\.system|os\.popen|shell=True)\b"),
    "filesystem_write": re.compile(r"\b(open\([^)]*[\"']w|os\.remove|os\.unlink|shutil\.rmtree|os\.chmod|os\.chown)\b"),
    "network_boundary": re.compile(r"\b(requests\.(get|post)|urlopen|fetch\(|socket\.|listen\(|bind\()\b"),
    "memory_unsafe": re.compile(r"\b(strcpy|memcpy|sprintf|gets\()\b"),
    "privilege_boundary": re.compile(r"\b(setuid|setgid|sudo|runas|impersonate|elevate)\b", re.I),
}

_COMMENT_ONLY_LINE_RE = re.compile(r"^[+-]\s*(#|//|/\*|\*|\"\"\"|''')")
_BLANK_LINE_RE = re.compile(r"^[+-]\s*$")
# Plain import only — no call expression, no aliasing to something executed.
_IMPORT_LINE_RE = re.compile(r"^[+-]\s*(import [\w.]+(\s+as\s+\w+)?|from [\w.]+ import [\w., ]+)\s*$")

# Safeguard #1 (see module docstring). Override via env for CI vs. local use.
DEFAULT_TRIVIAL_MAX_LINES = 25


@dataclass
class RoutingDecision:
    tier: RiskTier
    matched_signals: List[str]
    skip_llm: bool
    reason: str
    changed_line_count: int = 0


@dataclass
class RoutingSummary:
    """Per-commit routing metrics — printed to the developer and logged,
    so the routing decision is visible rather than a silent optimization."""
    total: int = 0
    critical: int = 0
    standard: int = 0
    trivial: int = 0
    npu_calls: int = 0
    cache_hits: int = 0
    skipped: int = 0

    def record(self, decision: RoutingDecision, was_cache_hit: bool = False):
        self.total += 1
        if decision.tier is RiskTier.CRITICAL:
            self.critical += 1
        elif decision.tier is RiskTier.STANDARD:
            self.standard += 1
        else:
            self.trivial += 1

        if decision.skip_llm:
            self.skipped += 1
        elif was_cache_hit:
            self.cache_hits += 1
        else:
            self.npu_calls += 1

    def render(self, commit_short_id: str = "", avg_npu_latency_seconds: float = 0.0) -> str:
        header = f"Commit {commit_short_id}" if commit_short_id else "Commit"
        avoided = self.cache_hits + self.skipped
        hit_rate = (self.cache_hits / (self.cache_hits + self.npu_calls) * 100) \
            if (self.cache_hits + self.npu_calls) else 0.0

        lines = [
            f"\n{header}",
            f"  {self.total} hunk(s)\n",
            f"  CRITICAL_PATH   {self.critical}",
            f"  STANDARD        {self.standard}",
            f"  TRIVIAL         {self.trivial}\n",
            f"  NPU reviews performed: {self.npu_calls}",
            f"  Served from cache:     {self.cache_hits}  (hit rate: {hit_rate:.0f}%)",
            f"  Skipped (fast-path):   {self.skipped}",
        ]
        if avoided and avg_npu_latency_seconds > 0:
            # Only shown when we actually measured a real average latency
            # THIS run (avg_npu_latency_seconds comes from the NPU calls
            # that did happen in this same run) — never a hardcoded or
            # benchmark-file number, since that would misrepresent what
            # was actually saved on this specific commit.
            saved = avoided * avg_npu_latency_seconds
            lines.append(
                f"  Est. latency saved:    ~{saved:.1f}s "
                f"({avoided} call(s) avoided \u00d7 {avg_npu_latency_seconds:.1f}s measured avg this run)"
            )
        elif avoided:
            lines.append(
                f"  Est. latency saved:    not shown (no NPU call happened this run to "
                f"measure a real average against \u2014 avoiding a guessed number)"
            )
        return "\n".join(lines) + "\n"


def _changed_lines(diff_text: str) -> List[str]:
    return [
        ln for ln in diff_text.splitlines()
        if (ln.startswith("+") or ln.startswith("-"))
        and not ln.startswith("+++") and not ln.startswith("---")
    ]


def _is_no_semantic_change(diff_text: str) -> bool:
    """True only if every changed line is a comment, blank line, or a plain
    (non-executing) import — never for anything touching actual logic."""
    changed = _changed_lines(diff_text)
    if not changed:
        return False
    for ln in changed:
        if _COMMENT_ONLY_LINE_RE.match(ln) or _BLANK_LINE_RE.match(ln) or _IMPORT_LINE_RE.match(ln):
            continue
        return False
    return True


def _matched_risk_signals(diff_text: str) -> List[str]:
    return [name for name, pattern in _RISK_SIGNALS.items() if pattern.search(diff_text)]


def classify(hunk: Hunk) -> RoutingDecision:
    changed = _changed_lines(hunk.diff_text)
    n_changed = len(changed)

    # Safeguard #2: risk signals always win, checked before anything else.
    signals = _matched_risk_signals(hunk.diff_text)
    if signals:
        return RoutingDecision(
            tier=RiskTier.CRITICAL,
            matched_signals=signals,
            skip_llm=False,
            reason=f"touches sensitive surface: {', '.join(signals)}",
            changed_line_count=n_changed,
        )

    fast_path_enabled = os.getenv("CODERX_FAST_PATH", "1") != "0"
    trivial_max_lines = int(os.getenv("CODERX_TRIVIAL_MAX_LINES", str(DEFAULT_TRIVIAL_MAX_LINES)))

    if fast_path_enabled and _is_no_semantic_change(hunk.diff_text):
        # Safeguard #1: size cap. A structurally-trivial diff that's large
        # is downgraded to STANDARD rather than skipped — send it to the
        # LLM anyway, just without the CRITICAL priority bump.
        if n_changed > trivial_max_lines:
            return RoutingDecision(
                tier=RiskTier.STANDARD,
                matched_signals=[],
                skip_llm=False,
                reason=f"structurally trivial but {n_changed} lines > "
                       f"{trivial_max_lines}-line safeguard cap — reviewed anyway",
                changed_line_count=n_changed,
            )
        return RoutingDecision(
            tier=RiskTier.TRIVIAL,
            matched_signals=[],
            skip_llm=True,
            reason="comment/whitespace/import-only change — no semantic diff",
            changed_line_count=n_changed,
        )

    return RoutingDecision(
        tier=RiskTier.STANDARD,
        matched_signals=[],
        skip_llm=False,
        reason="ordinary logic change",
        changed_line_count=n_changed,
    )


def sort_by_priority(hunks_with_decisions: List[Tuple[Hunk, RoutingDecision]]):
    """CRITICAL_PATH hunks first, so the highest-risk change is what the
    developer sees at the top of the mobile triage queue, matching how a
    human reviewer would triage their own PR."""
    order = {RiskTier.CRITICAL: 0, RiskTier.STANDARD: 1, RiskTier.TRIVIAL: 2}
    return sorted(hunks_with_decisions, key=lambda pair: order[pair[1].tier])
