import unittest

from hypothesis_factory.models import UncertaintyZone
from hypothesis_factory.scoring.profiles import profile_options, scoring_profile_weights, scoring_weights
from hypothesis_factory.scoring.scorer import calculate_uncertainty_importance, final_score


class ScoringTest(unittest.TestCase):
    def test_uncertainty_importance_formula(self):
        zone = UncertaintyZone(
            id="Z",
            type="coverage_gap",
            description="",
            target_kpi="kpi",
            kpi_relevance=1.0,
            gap_severity=0.8,
            contradiction_strength=0.5,
            indirect_mechanism_strength=0.25,
        )
        self.assertEqual(calculate_uncertainty_importance(zone), 0.7)

    def test_final_score_penalizes_risk_and_cost(self):
        score = final_score(
            {
                "value": 1,
                "novelty": 1,
                "feasibility": 1,
                "evidence": 1,
                "uncertainty_importance": 1,
                "risk": 1,
                "cost": 1,
            },
            {"value": 1, "novelty": 1, "feasibility": 1, "evidence": 1, "uncertainty": 1, "risk": 1, "cost": 1},
        )
        self.assertEqual(score, 3.0)

    def test_scoring_profile_returns_expected_weights(self):
        weights = scoring_profile_weights("balanced")

        self.assertEqual(weights["value"], 0.8)
        self.assertEqual(weights["novelty"], 0.6)
        self.assertEqual(weights["expert_alignment"], 0.5)

    def test_expert_weights_override_profile(self):
        weights = scoring_weights("balanced", {"value": 0.2, "risk": 0.95})

        self.assertEqual(weights["value"], 0.2)
        self.assertEqual(weights["risk"], 0.95)
        self.assertEqual(weights["novelty"], 0.6)

    def test_all_profile_weights_are_between_zero_and_one(self):
        for profile_key in profile_options():
            with self.subTest(profile_key=profile_key):
                for value in scoring_profile_weights(profile_key).values():
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

    def test_expert_weight_outside_range_is_rejected(self):
        with self.assertRaises(ValueError):
            scoring_weights("balanced", {"value": 1.5})


if __name__ == "__main__":
    unittest.main()

