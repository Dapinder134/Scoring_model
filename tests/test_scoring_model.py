"""scoring_model: validation, banding, and the achievable ceilings."""

import math
import unittest

import scoring_model as sm


class Validation(unittest.TestCase):
    def test_accepts_a_well_formed_night(self):
        self.assertEqual(len(sm.calculate_sleep_risks(0, 0, 11.1, 88.9, 2.5, 0.8)), 8)

    def test_rejects_percentages_that_do_not_sum_to_100(self):
        with self.assertRaises(sm.InvalidPostureInput):
            sm.calculate_sleep_risks(90, 90, 90, 90, 2, 1)

    def test_rejects_negatives(self):
        for args in [(-10, 0, 50, 60, 2, 1), (50, 0, 25, 25, -3, 1), (50, 0, 25, 25, 2, -1)]:
            with self.subTest(args=args), self.assertRaises(sm.InvalidPostureInput):
                sm.calculate_sleep_risks(*args)

    def test_rejects_an_empty_night_rather_than_calling_it_low_risk(self):
        with self.assertRaises(sm.InvalidPostureInput):
            sm.calculate_sleep_risks(0, 0, 0, 0, 0, 0)

    def test_rejects_nan_on_every_input(self):
        """NaN used to pass every guard and score 100.0 / High on all eight."""
        nan = float("nan")
        base = [25.0, 25.0, 25.0, 25.0, 2.0, 1.0]
        for i in range(6):
            args = list(base)
            args[i] = nan
            with self.subTest(position=i), self.assertRaises(sm.InvalidPostureInput):
                sm.calculate_sleep_risks(*args)

    def test_rejects_infinity_on_every_input(self):
        base = [25.0, 25.0, 25.0, 25.0, 2.0, 1.0]
        for i in range(6):
            args = list(base)
            args[i] = float("inf")
            with self.subTest(position=i), self.assertRaises(sm.InvalidPostureInput):
                sm.calculate_sleep_risks(*args)

    def test_rejects_non_numeric(self):
        with self.assertRaises(sm.InvalidPostureInput):
            sm.calculate_sleep_risks("n/a", 0, 50, 50, 1, 1)

    def test_validate_false_skips_the_checks(self):
        self.assertEqual(len(sm.calculate_sleep_risks(90, 90, 90, 90, 2, 1, validate=False)), 8)


class Banding(unittest.TestCase):
    def test_boundaries_land_in_the_specified_bands(self):
        cases = [(29.0, "Low"), (29.1, "Low"), (29.9, "Low"), (30.0, "Moderate"),
                 (59.9, "Moderate"), (60.0, "High"), (100.0, "High")]
        for score, expected in cases:
            with self.subTest(score=score):
                self.assertEqual(sm.band_for(score), expected)


class Ceilings(unittest.TestCase):
    """max_achievable must match what the formula can actually produce."""

    @staticmethod
    def brute_force(prone_allowed, step=5):
        best = {c: 0.0 for c in sm.WEIGHT_MATRIX}
        for a in range(0, 101, step):
            for b in range(0, 101 - a, step):
                if b and not prone_allowed:
                    continue
                for c in range(0, 101 - a - b, step):
                    d = 100 - a - b - c
                    for bout in (0, 1, 2, 4, 8):
                        for tr in (0, 1, 6, 12, 30):
                            for r in sm.calculate_sleep_risks(
                                a, b, c, d, bout, tr, validate=False,
                                prone_detectable=prone_allowed,
                            ):
                                best[r["condition"]] = max(best[r["condition"]], r["score"])
        return best

    def test_matches_brute_force_with_prone_available(self):
        best = self.brute_force(True)
        for condition, weights in sm.WEIGHT_MATRIX.items():
            with self.subTest(condition=condition):
                self.assertAlmostEqual(sm.max_achievable(weights, True), best[condition], places=1)

    def test_matches_brute_force_without_prone(self):
        best = self.brute_force(False)
        for condition, weights in sm.WEIGHT_MATRIX.items():
            with self.subTest(condition=condition):
                self.assertAlmostEqual(sm.max_achievable(weights, False), best[condition], places=1)

    def test_simplex_ceilings_are_reported(self):
        """These two are capped by the weights alone, prone or no prone."""
        self.assertEqual(sm.max_achievable(sm.WEIGHT_MATRIX["Nocturnal reflux (GORD)"], True), 55.0)
        self.assertEqual(
            sm.max_achievable(sm.WEIGHT_MATRIX["Shoulder / hip joint loading"], True), 70.0
        )

    def test_prone_ceilings_are_reported(self):
        self.assertEqual(sm.max_achievable(sm.WEIGHT_MATRIX["Neck strain"], False), 30.0)
        self.assertEqual(sm.max_achievable(sm.WEIGHT_MATRIX["Lower back strain"], False), 45.0)

    def test_no_score_ever_exceeds_its_reported_cap(self):
        mixes = [(25, 25, 25, 25), (100, 0, 0, 0), (0, 100, 0, 0),
                 (0, 0, 0, 100), (0, 0, 100, 0), (40, 10, 20, 30)]
        for prone in (True, False):
            for mix in mixes:
                if mix[1] and not prone:
                    continue
                for bout in (0, 2, 8):
                    for tr in (0, 6, 30):
                        for r in sm.calculate_sleep_risks(*mix, bout, tr,
                                                          prone_detectable=prone):
                            with self.subTest(prone=prone, mix=mix, condition=r["condition"]):
                                self.assertLessEqual(r["score"], r["capped_at"] + 1e-9)


class ProneDetectability(unittest.TestCase):
    def test_a_sleeper_who_was_not_prone_is_not_a_blind_spot(self):
        """prone_pct == 0 with a capable classifier is a finding, not a gap."""
        capable = {r["condition"]: r for r in sm.calculate_sleep_risks(
            50, 0, 25, 25, 2, 1, prone_detectable=True)}
        blind = {r["condition"]: r for r in sm.calculate_sleep_risks(
            50, 0, 25, 25, 2, 1, prone_detectable=False)}
        self.assertEqual(capable["Neck strain"]["capped_at"], 100.0)
        self.assertNotIn("blind spot", capable["Neck strain"]["summary"])
        self.assertEqual(blind["Neck strain"]["capped_at"], 30.0)
        self.assertIn("blind spot", blind["Neck strain"]["summary"])

    def test_default_falls_back_to_inferring_from_the_data(self):
        inferred = {r["condition"]: r for r in sm.calculate_sleep_risks(50, 0, 25, 25, 2, 1)}
        self.assertEqual(inferred["Neck strain"]["capped_at"], 30.0)

    def test_structural_cap_is_noted_even_when_prone_is_available(self):
        results = {r["condition"]: r for r in sm.calculate_sleep_risks(
            0, 0, 0, 100, 4, 0, prone_detectable=True)}
        self.assertIn("never reach High", results["Nocturnal reflux (GORD)"]["summary"])


if __name__ == "__main__":
    unittest.main()
