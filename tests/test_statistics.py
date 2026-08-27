"""sleep_statistics: sessions, runs, statistics and the data-quality gate."""

import unittest
from datetime import datetime, timedelta

import sleep_statistics as ss
from sleep_scoring import Config as CaptureConfig
from sleep_scoring import Interval, Posture

BASE = datetime(2026, 8, 21, 22, 0, 0)


def interval(offset_s, seconds, posture, inferred=0.0):
    return Interval(start=BASE + timedelta(seconds=offset_s),
                    end=BASE + timedelta(seconds=offset_s + seconds),
                    posture=posture, inferred_s=inferred)


def runs_from(*specs):
    return ss.make_run_objects(1, [interval(*s) for s in specs])


class SessionSplitting(unittest.TestCase):
    def test_a_night_with_hour_long_gaps_stays_one_session(self):
        """The bug that discarded 76.5% of the real capture."""
        ivs = [interval(0, 3600, Posture.RIGHT),
               interval(3600, 4000, Posture.ABSENT),
               interval(7600, 3600, Posture.RIGHT)]
        sessions, between = ss.split_sessions(ivs, ss.Config().session_gap_s)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(between, 0.0)
        self.assertEqual(sum(i.duration_s for i in sessions[0]), 11200.0)

    def test_a_genuinely_separate_night_does_split(self):
        ivs = [interval(0, 3600, Posture.RIGHT),
               interval(3600, 60000, Posture.ABSENT),
               interval(63600, 3600, Posture.LEFT)]
        sessions, between = ss.split_sessions(ivs, ss.Config().session_gap_s)
        self.assertEqual(len(sessions), 2)
        self.assertEqual(between, 60000.0)

    def test_boundary_time_is_reported_not_silently_dropped(self):
        ivs = [interval(0, 100, Posture.RIGHT),
               interval(100, 50000, Posture.ABSENT),
               interval(50100, 100, Posture.LEFT)]
        sessions, between = ss.split_sessions(ivs, 14400)
        accounted = sum(i.duration_s for s in sessions for i in s) + between
        self.assertEqual(accounted, 50200.0)

    def test_no_intervals(self):
        self.assertEqual(ss.split_sessions([], 600), ([], 0.0))


class Statistics(unittest.TestCase):
    def test_time_budget_balances(self):
        stats = ss.compute_statistics(runs_from(
            (0, 3600, Posture.RIGHT), (3600, 1800, Posture.ABSENT),
            (5400, 3600, Posture.LEFT)))
        self.assertAlmostEqual(stats["sum_check"], 0.0)
        self.assertAlmostEqual(stats["scored_s"], 7200.0)
        self.assertAlmostEqual(stats["absent_s"], 1800.0)

    def test_percentages_are_of_scored_time_and_sum_to_100(self):
        stats = ss.compute_statistics(runs_from(
            (0, 3600, Posture.RIGHT), (3600, 1800, Posture.ABSENT),
            (5400, 3600, Posture.LEFT)))
        self.assertAlmostEqual(stats["pct"]["Right Side"], 50.0)
        self.assertAlmostEqual(stats["pct"]["Left Side"], 50.0)
        self.assertAlmostEqual(stats["pct_check"], 100.0)

    def test_leaving_and_returning_to_the_same_side_is_not_a_transition(self):
        stats = ss.compute_statistics(runs_from(
            (0, 3600, Posture.RIGHT), (3600, 1800, Posture.ABSENT),
            (5400, 3600, Posture.RIGHT)))
        self.assertEqual(stats["transitions"], 0)
        self.assertEqual(stats["n_bouts"], 2)

    def test_absence_breaks_the_bout_chain(self):
        stats = ss.compute_statistics(runs_from(
            (0, 3600, Posture.RIGHT), (3600, 1800, Posture.ABSENT),
            (5400, 3600, Posture.RIGHT)))
        self.assertAlmostEqual(stats["longest_bout_s"], 3600.0)

    def test_real_turn_counts_once(self):
        stats = ss.compute_statistics(runs_from(
            (0, 3600, Posture.RIGHT), (3600, 3600, Posture.LEFT)))
        self.assertEqual(stats["transitions"], 1)

    def test_inferred_time_is_carried_through(self):
        stats = ss.compute_statistics(runs_from(
            (0, 3600, Posture.RIGHT, 3000.0), (3600, 3600, Posture.LEFT, 0.0)))
        self.assertAlmostEqual(stats["inferred_s"], 3000.0)
        self.assertAlmostEqual(stats["inferred_pct"], 100 * 3000 / 7200)

    def test_empty_runs_do_not_divide_by_zero(self):
        stats = ss.compute_statistics([])
        self.assertEqual(stats["trans_per_hour"], 0.0)
        self.assertEqual(stats["mean_bout_s"], 0.0)


class MorningReview(unittest.TestCase):
    def test_reassignment_moves_absent_time_into_a_posture(self):
        runs = runs_from((0, 3600, Posture.RIGHT), (3600, 1800, Posture.ABSENT))
        before = ss.compute_statistics(runs)
        self.assertAlmostEqual(before["absent_s"], 1800.0)
        runs[1].x = 4  # right side
        after = ss.compute_statistics(runs)
        self.assertAlmostEqual(after["absent_s"], 0.0)
        self.assertAlmostEqual(after["pct"]["Right Side"], 100.0)
        self.assertAlmostEqual(after["longest_bout_s"], 5400.0)

    def test_parse_reassign(self):
        self.assertEqual(ss.parse_reassign(["5=1", "11=2"]), {5: 1, 11: 2})
        self.assertEqual(ss.parse_reassign(None), {})


class CaptureModel(unittest.TestCase):
    def test_debounce_is_meaningful_only_above_the_capture_interval(self):
        """Under sample-and-hold the shortest possible run is one interval."""
        self.assertFalse(ss.Config(nominal_interval_s=210, debounce_s=5).debounce_is_meaningful())
        self.assertTrue(ss.Config(nominal_interval_s=210, debounce_s=300).debounce_is_meaningful())
        self.assertFalse(ss.Config(nominal_interval_s=210, debounce_s=210).debounce_is_meaningful())

    def test_zero_interval_does_not_divide_by_zero(self):
        cfg = ss.Config(nominal_interval_s=0)
        self.assertEqual(cfg.max_detectable_tph(), float("inf"))

    def test_capture_config_translation(self):
        cfg = ss.Config(debounce_s=45, bridge_absence_s=90, max_hold_s=600, final_hold_s=210)
        capture = cfg.capture_config()
        self.assertIsInstance(capture, CaptureConfig)
        self.assertEqual(capture.min_dwell_s, 45)
        self.assertEqual(capture.max_bridge_absence_s, 90)
        self.assertEqual(capture.max_hold_s, 600)
        self.assertEqual(capture.tail_hold_s, 210)


class Constraints(unittest.TestCase):
    def test_a_clean_night_passes_every_gate(self):
        runs = runs_from((0, 14400, Posture.RIGHT), (14400, 14400, Posture.LEFT))
        failed = [n for n, _, _, ok, _ in ss.check_constraints(
            ss.compute_statistics(runs), ss.Config()) if not ok]
        self.assertEqual(failed, [])

    def test_two_positions_is_enough(self):
        """Requiring all four failed every sleeper who never lies prone."""
        runs = runs_from((0, 14400, Posture.RIGHT), (14400, 14400, Posture.LEFT))
        stats = ss.compute_statistics(runs)
        self.assertEqual(stats["distinct_positions"], 2)
        gate = dict((n, ok) for n, _, _, ok, _ in ss.check_constraints(stats, ss.Config()))
        self.assertTrue(gate["At least the minimum distinct positions"])

    def test_too_much_absence_fails(self):
        runs = runs_from((0, 14400, Posture.RIGHT), (14400, 14400, Posture.ABSENT))
        gate = dict((n, ok) for n, _, _, ok, _ in ss.check_constraints(
            ss.compute_statistics(runs), ss.Config()))
        self.assertFalse(gate["Unresolved absent time within limit"])

    def test_short_night_fails(self):
        runs = runs_from((0, 600, Posture.RIGHT))
        gate = dict((n, ok) for n, _, _, ok, _ in ss.check_constraints(
            ss.compute_statistics(runs), ss.Config()))
        self.assertFalse(gate["Scored length at least the minimum"])


if __name__ == "__main__":
    unittest.main()
