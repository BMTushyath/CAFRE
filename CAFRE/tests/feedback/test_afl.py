import unittest
import pandas as pd
from unittest.mock import MagicMock
from CAFRE.feedback.afl import (
    AgenticFeedbackLoop,
    ValidateWithAlternativeClassifier,
    SubpopulationDisparityAnalysis,
    DeepCausalDiscovery
)

class TestAgenticFeedbackLoop(unittest.TestCase):
    def setUp(self):
        self.afl = AgenticFeedbackLoop(max_loops=2)
        
        # Candidate actions
        self.actions = [
            ValidateWithAlternativeClassifier("gender"),
            SubpopulationDisparityAnalysis("gender"),
            DeepCausalDiscovery("gender", "income")
        ]
        
        self.df = pd.DataFrame({
            "gender": [0, 1, 0, 1],
            "income": [10, 20, 10, 30],
            "target": [0, 1, 0, 1]
        })

    def test_pareto_filtering(self):
        # We test that pareto filtering correctly removes dominated actions
        non_dominated = self.afl.filter_pareto_dominated(self.actions)
        self.assertGreater(len(non_dominated), 0)
        
        # Test selection
        best = self.afl.select_best_action(non_dominated)
        self.assertIsNotNone(best)

    def test_feedback_loop_execution(self):
        # Mock cre_engine
        cre_mock = MagicMock()
        
        # Define mock results for CRE
        # First iteration suspect exists
        r1 = MagicMock()
        r1.column_name = "gender"
        r1.priority = "HIGH"
        r1.confidence = 50.0
        r1.is_uncertain = True
        
        r2 = MagicMock()
        r2.column_name = "income"
        r2.priority = "LOW"
        r2.confidence = 90.0
        r2.is_uncertain = False
        
        # Second iteration: no suspects
        r_ok = MagicMock()
        r_ok.column_name = "gender"
        r_ok.priority = "HIGH"
        r_ok.confidence = 90.0
        r_ok.is_uncertain = False
        
        cre_mock.analyze_dataset.side_effect = [
            [r1, r2],  # Loop 0 CRE check
            [r_ok, r2] # Loop 1 CRE check
        ]
        
        fee_init = {"gender": {"predictive_impact": 0.01}}
        ref_init = {"gender": {"disparate_impact": 0.5}}
        trace = []
        
        updated_fee, actions = self.afl.run_feedback_loop(
            self.df, "target", cre_mock, fee_init, ref_init, trace
        )
        
        # Should have executed at least 1 action and updated the evidence
        self.assertGreater(len(actions), 0)
        self.assertIn("gender", updated_fee)
        self.assertTrue(len(trace) > 0)

if __name__ == "__main__":
    unittest.main()
