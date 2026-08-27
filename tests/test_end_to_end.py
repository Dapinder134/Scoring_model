"""Both real capture files, through the whole chain."""

import os
import subprocess
import sys
import unittest

import scoring_model
import sleep_statistics as ss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
NIGHT = os.path.join(DATA, "Night_001_20260821_222832.csv")
SHORT = os.path.join(DATA, "Night_001_20260821_173904.csv")


@unittest.skipUnless(os.path.exists(NIGHT), "capture files not present")
class FullNight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = ss.Config()
        cls.runs, cls.load, cls.timeline, cls.between = ss.build_sessions(NIGHT, cls.cfg)
        cls.stats = ss.compute_statistics(cls.runs[1])

    def test_reads_every_row(self):
        self.assertEqual(self.load.raw_row_count, 3725)
        self.assertEqual(self.load.skipped_rows, 0)

    def test_the_night_is_one_session(self):
        self.assertEqual(len(self.runs), 1)
        self.assertEqual(self.between, 0.0)

    def test_no_recorded_time_is_discarded(self):
        """The session-splitting bug lost 76.5% of this file."""
        span = (self.load.samples[-1].timestamp - self.load.samples[0].timestamp).total_seconds()
        self.assertAlmostEqual(self.stats["total_s"], span, places=3)

    def test_raw_flicker_is_detected_and_removed(self):
        self.assertGreater(self.timeline.raw_transitions_per_h, 500)
        self.assertLess(self.stats["trans_per_hour"], 2.0)

    def test_time_budget_and_percentages_balance(self):
        self.assertAlmostEqual(self.stats["sum_check"], 0.0, places=3)
        self.assertAlmostEqual(self.stats["pct_check"], 100.0, places=3)

    def test_right_side_dominates(self):
        self.assertGreater(self.stats["pct"]["Right Side"], 80)
        self.assertAlmostEqual(self.stats["pct"]["Front"], 0.0)

    def test_most_of_the_night_is_inferred(self):
        self.assertGreater(self.stats["inferred_pct"], 50)

    def test_absence_fails_the_gate_until_the_morning_review(self):
        gate = dict((n, ok) for n, _, _, ok, _ in ss.check_constraints(self.stats, self.cfg))
        self.assertFalse(gate["Unresolved absent time within limit"])

    def test_reassignment_lets_the_night_score(self):
        runs, _, _, _ = ss.build_sessions(NIGHT, self.cfg)
        for run in runs[1]:
            if run.raw_label == ss.UNCLASSIFIED:
                run.x = 4  # the surrounding posture
        stats = ss.compute_statistics(runs[1])
        failed = [n for n, _, _, ok, _ in ss.check_constraints(stats, self.cfg) if not ok]
        self.assertEqual(failed, [])
        results = scoring_model.calculate_sleep_risks(
            stats["pct"]["Back"], stats["pct"]["Front"], stats["pct"]["Left Side"],
            stats["pct"]["Right Side"], stats["longest_bout_h"], stats["trans_per_hour"],
            prone_detectable=False,
        )
        self.assertEqual(len(results), 8)
        for r in results:
            self.assertLessEqual(r["score"], r["capped_at"] + 1e-9)

    def test_prone_is_flagged_as_undetectable(self):
        self.assertTrue(any("prone" in i for i in self.load.issues))


@unittest.skipUnless(os.path.exists(SHORT), "capture files not present")
class ShortCapture(unittest.TestCase):
    def test_ten_second_file_fails_the_gate(self):
        cfg = ss.Config()
        runs, _, _, _ = ss.build_sessions(SHORT, cfg)
        stats = ss.compute_statistics(runs[1])
        failed = [n for n, _, _, ok, _ in ss.check_constraints(stats, cfg) if not ok]
        self.assertIn("Scored length at least the minimum", failed)


@unittest.skipUnless(os.path.exists(NIGHT), "capture files not present")
class BothFormats(unittest.TestCase):
    @staticmethod
    def as_text_log(cfg):
        import tempfile
        rows, _ = ss.load_log(NIGHT, cfg)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            for stamp, label in rows:
                fh.write(f"{stamp.strftime('%Y-%m-%d %H:%M:%S')}, {label}\n")
            return fh.name

    def test_both_formats_agree_on_the_time_budget(self):
        """Everything that does not depend on landmark confidence must match."""
        cfg = ss.Config()
        path = self.as_text_log(cfg)
        try:
            csv_runs, _, _, _ = ss.build_sessions(NIGHT, cfg)
            txt_runs, _, _, _ = ss.build_sessions(path, cfg)
            a = ss.compute_statistics(csv_runs[1])
            b = ss.compute_statistics(txt_runs[1])
            self.assertEqual(len(csv_runs), len(txt_runs))
            self.assertAlmostEqual(a["total_s"], b["total_s"], places=3)
            self.assertAlmostEqual(a["scored_s"], b["scored_s"], places=3)
            self.assertAlmostEqual(a["absent_s"], b["absent_s"], places=3)
        finally:
            os.unlink(path)

    def test_posture_split_may_differ_because_the_text_log_drops_confidence(self):
        """A documented limitation, asserted so it cannot change unnoticed.

        The CSV resolves a contested timestamp using landmark visibility. The
        text log has no such column, so it can only take the later record. On a
        capture where most time is held from single records, that one choice
        before a long gap moves a large block of the night.
        """
        cfg = ss.Config()
        path = self.as_text_log(cfg)
        try:
            a = ss.compute_statistics(ss.build_sessions(NIGHT, cfg)[0][1])
            b = ss.compute_statistics(ss.build_sessions(path, cfg)[0][1])
            self.assertGreater(a["inferred_pct"], 50)
            self.assertNotAlmostEqual(a["pct"]["Left Side"], b["pct"]["Left Side"], places=1)
            # Both still agree on which side dominates.
            self.assertGreater(a["pct"]["Right Side"], a["pct"]["Left Side"])
            self.assertGreater(b["pct"]["Right Side"], b["pct"]["Left Side"])
        finally:
            os.unlink(path)


class CommandLine(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(NIGHT), "capture files not present")
    def test_cli_runs_clean(self):
        out = subprocess.run([sys.executable, "sleep_statistics.py", NIGHT],
                             cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("NIGHT STATISTICS", out.stdout)
        self.assertIn("DATA QUALITY CONSTRAINTS", out.stdout)

    def test_cli_handles_a_junk_file(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("not a record\nnor this\n")
            path = fh.name
        try:
            out = subprocess.run([sys.executable, "sleep_statistics.py", path],
                                 cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("No parseable records", out.stdout)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
