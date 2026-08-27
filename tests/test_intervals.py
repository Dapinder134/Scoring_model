"""Back end: the hold rule and the three cleaning stages."""

import unittest
from datetime import datetime, timedelta

from sleep_scoring import Config, Posture, Sample, build_timeline

BASE = datetime(2026, 8, 21, 5, 0, 0)


def sample(offset_s, posture, visibility=0.9, row=0):
    return Sample(timestamp=BASE + timedelta(seconds=offset_s), session="T",
                  raw_position=posture, posture=posture,
                  visibility={"LEFT_EYE": visibility}, row_number=row)


def totals(timeline):
    out = {}
    for i in timeline.intervals:
        out[i.posture] = out.get(i.posture, 0.0) + i.duration_s
    return out


class HoldRule(unittest.TestCase):
    def test_row_holds_until_the_next_row(self):
        """05:06:10 -> 05:10:13 is one unbroken 243 s stretch."""
        cfg = Config(min_dwell_s=0, max_bridge_absence_s=0)
        tl = build_timeline([sample(0, Posture.LEFT), sample(243, Posture.RIGHT),
                             sample(300, Posture.RIGHT)], cfg)
        self.assertAlmostEqual(totals(tl)[Posture.LEFT], 243.0)

    def test_a_repeat_continues_the_stretch(self):
        cfg = Config(min_dwell_s=0, max_bridge_absence_s=0)
        tl = build_timeline([sample(0, Posture.RIGHT), sample(60, Posture.RIGHT),
                             sample(120, Posture.RIGHT), sample(180, Posture.LEFT)], cfg)
        self.assertEqual(len([i for i in tl.intervals if i.posture == Posture.RIGHT]), 1)
        self.assertAlmostEqual(totals(tl)[Posture.RIGHT], 180.0)

    def test_max_hold_cap_creates_unmonitored_time(self):
        cfg = Config(min_dwell_s=0, max_bridge_absence_s=0, max_hold_s=60)
        tl = build_timeline([sample(0, Posture.LEFT), sample(600, Posture.LEFT)], cfg)
        self.assertAlmostEqual(totals(tl)[Posture.LEFT], 60.0)
        self.assertAlmostEqual(totals(tl)[Posture.UNMONITORED], 540.0)


class DuplicateTimestamps(unittest.TestCase):
    def test_a_detection_beats_a_non_detection_at_the_same_instant(self):
        cfg = Config(min_dwell_s=0, max_bridge_absence_s=0)
        tl = build_timeline([sample(0, Posture.ABSENT, 0.0, row=1),
                             sample(0, Posture.RIGHT, 0.9, row=2),
                             sample(120, Posture.RIGHT, row=3)], cfg)
        self.assertAlmostEqual(totals(tl)[Posture.RIGHT], 120.0)
        self.assertNotIn(Posture.ABSENT, totals(tl))
        self.assertEqual(tl.conflicting_timestamps, 1)


class Cleaning(unittest.TestCase):
    def test_short_dropout_is_refilled(self):
        cfg = Config(min_dwell_s=0, max_bridge_absence_s=120)
        tl = build_timeline([sample(0, Posture.RIGHT), sample(100, Posture.ABSENT),
                             sample(130, Posture.RIGHT), sample(300, Posture.RIGHT)], cfg)
        self.assertAlmostEqual(totals(tl)[Posture.RIGHT], 300.0)

    def test_long_absence_is_kept(self):
        cfg = Config(min_dwell_s=0, max_bridge_absence_s=120)
        tl = build_timeline([sample(0, Posture.RIGHT), sample(100, Posture.ABSENT),
                             sample(1000, Posture.RIGHT), sample(1200, Posture.RIGHT)], cfg)
        self.assertAlmostEqual(totals(tl)[Posture.ABSENT], 900.0)

    def test_single_frame_blip_is_absorbed(self):
        cfg = Config(min_dwell_s=30, max_bridge_absence_s=0)
        tl = build_timeline([sample(0, Posture.RIGHT), sample(600, Posture.LEFT),
                             sample(602, Posture.RIGHT), sample(1200, Posture.RIGHT)], cfg)
        self.assertEqual(len(tl.intervals), 1)
        self.assertAlmostEqual(totals(tl)[Posture.RIGHT], 1200.0)

    def test_a_genuine_turn_survives(self):
        cfg = Config(min_dwell_s=30, max_bridge_absence_s=0)
        tl = build_timeline([sample(0, Posture.RIGHT), sample(600, Posture.LEFT),
                             sample(1200, Posture.RIGHT), sample(1800, Posture.RIGHT)], cfg)
        self.assertEqual(len(tl.intervals), 3)

    def test_absence_never_swallows_a_real_detection(self):
        """A detection between two absences must stay a detection."""
        cfg = Config(min_dwell_s=60, max_bridge_absence_s=0)
        tl = build_timeline([sample(0, Posture.ABSENT), sample(3600, Posture.RIGHT),
                             sample(3630, Posture.ABSENT), sample(7200, Posture.ABSENT)], cfg)
        self.assertAlmostEqual(totals(tl)[Posture.RIGHT], 30.0)

    def test_alternating_flicker_becomes_one_stretch(self):
        """The dominant artefact in the real captures."""
        cfg = Config(min_dwell_s=30, max_bridge_absence_s=120)
        samples = []
        for n in range(60):
            samples.append(sample(2 * n, Posture.RIGHT, row=2 * n))
            samples.append(sample(2 * n + 1, Posture.ABSENT, 0.0, row=2 * n + 1))
        samples.append(sample(200, Posture.RIGHT, row=999))
        tl = build_timeline(samples, cfg)
        self.assertEqual(len(tl.intervals), 1)
        self.assertEqual(tl.intervals[0].posture, Posture.RIGHT)


class Evidence(unittest.TestCase):
    def test_long_hold_is_marked_inferred(self):
        cfg = Config(min_dwell_s=0, max_bridge_absence_s=0, evidence_gap_s=120)
        tl = build_timeline([sample(0, Posture.LEFT), sample(600, Posture.LEFT),
                             sample(660, Posture.RIGHT)], cfg)
        left = [i for i in tl.intervals if i.posture == Posture.LEFT][0]
        self.assertAlmostEqual(left.inferred_s, 600.0)

    def test_dense_sampling_is_not_inferred(self):
        cfg = Config(min_dwell_s=0, max_bridge_absence_s=0, evidence_gap_s=120)
        tl = build_timeline([sample(10 * n, Posture.RIGHT, row=n) for n in range(50)], cfg)
        self.assertAlmostEqual(sum(i.inferred_s for i in tl.intervals), 0.0)


if __name__ == "__main__":
    unittest.main()
