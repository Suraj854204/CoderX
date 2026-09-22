"""
tests/test_pipeline.py

Regression tests for the code-verifiable parts of CoderX added/changed in
this pass: risk_router, review_cache, response_parser's provenance fields,
and the JSON audit trail. Deliberately dependency-free (stdlib `unittest`
only, no pytest) so `python3 tests/test_pipeline.py` works with nothing
beyond requirements.txt already installed.

Run from backend/:
    python3 tests/test_pipeline.py -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diff_extractor import Hunk
from response_parser import Finding, parse_findings
from risk_router import classify, sort_by_priority, RiskTier, RoutingSummary, DEFAULT_TRIVIAL_MAX_LINES


class TestRiskRouter(unittest.TestCase):
    def test_auth_and_sql_signal_routes_critical(self):
        h = Hunk("auth.py", 10,
                  '+    password = request.form["password"]\n'
                  '+    db.execute(f"SELECT * FROM users WHERE pw={password}")')
        d = classify(h)
        self.assertEqual(d.tier, RiskTier.CRITICAL)
        self.assertIn("auth", d.matched_signals)
        self.assertIn("sql", d.matched_signals)
        self.assertFalse(d.skip_llm)

    def test_comment_only_hunk_is_trivial_and_skipped(self):
        h = Hunk("utils.py", 5, '+# just a comment\n+   \n+from foo import bar')
        d = classify(h)
        self.assertEqual(d.tier, RiskTier.TRIVIAL)
        self.assertTrue(d.skip_llm)

    def test_ordinary_logic_is_standard(self):
        h = Hunk("math.py", 1, '+def add(a,b):\n+    return a+b')
        d = classify(h)
        self.assertEqual(d.tier, RiskTier.STANDARD)
        self.assertFalse(d.skip_llm)

    def test_large_trivial_looking_hunk_is_escalated_not_skipped(self):
        """Safeguard: a big comment-only-looking diff must NOT be silently
        skipped — this is the exact case the size-cap safeguard exists for."""
        big_diff = "\n".join(f"+# comment {i}" for i in range(DEFAULT_TRIVIAL_MAX_LINES + 5))
        h = Hunk("bulk.py", 1, big_diff)
        d = classify(h)
        self.assertEqual(d.tier, RiskTier.STANDARD)
        self.assertFalse(d.skip_llm)

    def test_risk_signal_overrides_trivial_pattern(self):
        """A comment line that happens to mention a sensitive term must
        still route CRITICAL, never be treated as a harmless comment-only
        edit."""
        h = Hunk("notes.py", 1, '+# TODO: temporarily disabled auth check, subprocess.run below is unrelated\n+subprocess.run(["ls"])')
        d = classify(h)
        self.assertEqual(d.tier, RiskTier.CRITICAL)

    def test_fast_path_disabled_via_env(self):
        os.environ["CODERX_FAST_PATH"] = "0"
        try:
            h = Hunk("utils.py", 5, '+# just a comment')
            d = classify(h)
            self.assertFalse(d.skip_llm)
        finally:
            del os.environ["CODERX_FAST_PATH"]

    def test_sort_by_priority_puts_critical_first(self):
        h1 = Hunk("a.py", 1, '+def x(): pass')
        h2 = Hunk("b.py", 1, '+os.system(cmd)')
        h3 = Hunk("c.py", 1, '+# comment')
        routed = [(h, classify(h)) for h in (h1, h2, h3)]
        sorted_routed = sort_by_priority(routed)
        self.assertEqual(sorted_routed[0][1].tier, RiskTier.CRITICAL)
        self.assertEqual(sorted_routed[-1][1].tier, RiskTier.TRIVIAL)

    def test_routing_summary_counts_and_latency(self):
        s = RoutingSummary()
        h_critical = Hunk("a.py", 1, '+os.system(cmd)')
        h_trivial = Hunk("b.py", 1, '+# comment')
        s.record(classify(h_critical), was_cache_hit=False)
        s.record(classify(h_trivial), was_cache_hit=False)
        self.assertEqual(s.critical, 1)
        self.assertEqual(s.trivial, 1)
        self.assertEqual(s.npu_calls, 1)
        self.assertEqual(s.skipped, 1)
        rendered = s.render("abc123", avg_npu_latency_seconds=10.0)
        self.assertIn("CRITICAL_PATH   1", rendered)
        self.assertIn("Est. latency saved", rendered)

    def test_routing_summary_no_fake_latency_without_measurement(self):
        s = RoutingSummary()
        h_trivial = Hunk("b.py", 1, '+# comment')
        s.record(classify(h_trivial), was_cache_hit=False)
        rendered = s.render("abc123", avg_npu_latency_seconds=0.0)
        self.assertIn("not shown", rendered)
        self.assertNotIn("Est. latency saved:    ~", rendered)


class TestReviewCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        import review_cache
        self.cache = review_cache
        # _DB_PATH is anchored to review_cache.py's own location (by
        # design — the cache shouldn't move depending on the caller's
        # cwd), so isolate each test by pointing it at a fresh temp file
        # directly rather than relying on os.chdir().
        self._orig_db_path = self.cache._DB_PATH
        self.cache._DB_PATH = self._tmp + "/test_cache.db"

    def tearDown(self):
        self.cache._DB_PATH = self._orig_db_path

    def test_store_and_lookup_roundtrip(self):
        f = Finding(severity="MINOR", file="x.py", line=1, description="d", fix="f")
        self.cache.store("+ some diff", [f])
        hit = self.cache.lookup("+ some diff")
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0].description, "d")

    def test_lookup_miss_returns_none(self):
        self.assertIsNone(self.cache.lookup("+ never stored"))

    def test_cache_hit_retags_inference_path(self):
        f = Finding(severity="MINOR", file="x.py", line=1, description="d",
                     inference_path="npu:qnn", model="Qwen3-4B")
        self.cache.store("+ diff", [f])
        hit = self.cache.lookup("+ diff")
        self.assertEqual(hit[0].inference_path, "cache")
        self.assertEqual(hit[0].model, "Qwen3-4B")  # provenance preserved

    def test_prompt_version_change_invalidates_cache(self):
        f = Finding(severity="MINOR", file="x.py", line=1, description="d")
        self.cache.store("+ diff", [f])
        import prompt_builder
        original = prompt_builder.PROMPT_TEMPLATE_VERSION
        prompt_builder.PROMPT_TEMPLATE_VERSION = "v2-test"
        try:
            self.assertIsNone(self.cache.lookup("+ diff"))
        finally:
            prompt_builder.PROMPT_TEMPLATE_VERSION = original

    def test_invalidate_all_clears_cache(self):
        f = Finding(severity="MINOR", file="x.py", line=1, description="d")
        self.cache.store("+ diff", [f])
        removed = self.cache.invalidate_all()
        self.assertEqual(removed, 1)
        self.assertIsNone(self.cache.lookup("+ diff"))

    def test_normalized_hash_ignores_line_number_shift(self):
        """Two hunks with identical code but different @@ line headers
        (e.g. because an earlier edit shifted line numbers) should hit
        the same cache entry."""
        diff_a = "@@ -10,3 +10,3 @@\n+def f(): pass"
        diff_b = "@@ -14,3 +14,3 @@\n+def f(): pass"
        f = Finding(severity="MINOR", file="x.py", line=1, description="d")
        self.cache.store(diff_a, [f])
        self.assertIsNotNone(self.cache.lookup(diff_b))


class TestResponseParser(unittest.TestCase):
    def test_parses_standard_format(self):
        raw = "[CRITICAL] app.py:10 \u2014 SQL injection. Fix: use params."
        findings = parse_findings(raw)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, "CRITICAL")
        self.assertEqual(findings[0].line, 10)

    def test_confidence_defaults_none_never_fabricated(self):
        f = Finding(severity="MINOR", file="x.py", line=1, description="d")
        self.assertIsNone(f.confidence)
        self.assertIn("Not calibrated", f.confidence_note)

    def test_no_issues_found_yields_empty_list(self):
        self.assertEqual(parse_findings("NO_ISSUES_FOUND"), [])


class TestWebhookSignature(unittest.TestCase):
    def test_signature_verification(self):
        import hmac
        import hashlib
        os.environ["CODERX_WEBHOOK_SECRET"] = "test-secret"
        try:
            import importlib
            import webhook_server
            importlib.reload(webhook_server)
            body = b'{"action":"opened"}'
            good = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
            self.assertTrue(webhook_server._verify_github_signature(body, good))
            self.assertFalse(webhook_server._verify_github_signature(body, "sha256=deadbeef"))
            self.assertFalse(webhook_server._verify_github_signature(body, None))
        finally:
            del os.environ["CODERX_WEBHOOK_SECRET"]

    def test_verification_skipped_when_no_secret_configured(self):
        os.environ.pop("CODERX_WEBHOOK_SECRET", None)
        import importlib
        import webhook_server
        importlib.reload(webhook_server)
        # No secret configured -> verification intentionally permissive
        # (documented behavior for local/dev use), never silently False.
        self.assertTrue(webhook_server._verify_github_signature(b"{}", None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
