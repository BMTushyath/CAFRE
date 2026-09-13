import unittest
import pandas as pd
import numpy as np
from CAFRE.evidence.ref import RegisteredFairnessEvidences

class TestRegisteredFairnessEvidences(unittest.TestCase):
    def setUp(self):
        self.ref = RegisteredFairnessEvidences()
        
        # Create a synthetic dataset with known bias
        # 100 samples
        n_samples = 100
        gender = ['Male'] * 50 + ['Female'] * 50
        
        # Male selection rate: 40/50 = 0.8
        # Female selection rate: 10/50 = 0.2
        target = [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40
        
        self.df = pd.DataFrame({
            "gender": gender,
            "target": target
        })

    def test_positive_outcome_detection(self):
        # Numeric binary
        s1 = pd.Series([0, 1, 0, 1, 1])
        self.assertEqual(self.ref.detect_positive_outcome(s1), 1)
        
        # Keyword strings
        s2 = pd.Series(["hired", "rejected", "rejected"])
        self.assertEqual(self.ref.detect_positive_outcome(s2), "hired")
        
        # High outcome string
        s3 = pd.Series([">50K", "<=50K", "<=50K"])
        self.assertEqual(self.ref.detect_positive_outcome(s3), ">50K")

    def test_compute_metrics(self):
        metrics = self.ref.compute_metrics(
            self.df,
            protected_col="gender",
            target_col="target",
            positive_outcome=1
        )
        
        # Disparate Impact should be 0.2 / 0.8 = 0.25
        self.assertAlmostEqual(metrics["disparate_impact"], 0.25)
        
        # Demographic Parity Difference should be 0.8 - 0.2 = 0.6
        self.assertAlmostEqual(metrics["demographic_parity_difference"], 0.6)
        
        # Equal Opportunity Difference (since target itself is used, TPR for both is 1.0)
        # TPR = TP/P. For Males, P=40, decisions=40 (all TP), TPR=1.0
        # For Females, P=10, decisions=10 (all TP), TPR=1.0
        # Diff = 0.0
        self.assertAlmostEqual(metrics["equal_opportunity_difference"], 0.0)

if __name__ == "__main__":
    unittest.main()
