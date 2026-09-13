import unittest
import pandas as pd
import numpy as np
from CAFRE.evidence.mpem import FairnessEvidenceExtractor
from CAFRE.investigation.planner import InvestigationExecutionPlan, FeatureInvestigationPlan

class TestFEE(unittest.TestCase):
    def setUp(self):
        self.fee = FairnessEvidenceExtractor()
        
        # Create a synthetic dataset
        np.random.seed(42)
        n_samples = 200
        
        # Age is random
        age = np.random.randint(18, 70, size=n_samples)
        
        # Zipcode is mostly 2 values
        zipcode = np.random.choice(["10001", "90210"], size=n_samples)
        
        # Income is correlated with age
        income = age * 1000 + np.random.normal(0, 10000, size=n_samples)
        
        # Target variable (approved) is heavily dependent on income, slightly on zipcode
        approved = np.where((income > 40000) & (zipcode == "10001"), 1, 0)
        
        # Flip a few for noise
        noise_idx = np.random.choice(n_samples, size=10, replace=False)
        approved[noise_idx] = 1 - approved[noise_idx]
        
        self.dataset = pd.DataFrame({
            "age": age,
            "zipcode": zipcode,
            "income": income,
            "loan_approved": approved
        })
        
        # Create execution plan
        self.plan = InvestigationExecutionPlan(
            domain="Finance",
            target_variable="loan_approved",
            feature_plans=[
                FeatureInvestigationPlan(
                    feature_name="age",
                    run_shap=True,
                    run_mutual_information=True,
                    run_causal_discovery=False
                ),
                FeatureInvestigationPlan(
                    feature_name="zipcode",
                    run_shap=True,
                    run_mutual_information=True,
                    run_causal_discovery=True
                ),
                FeatureInvestigationPlan(
                    feature_name="income",
                    run_shap=True,
                    run_mutual_information=True,
                    run_causal_discovery=False
                )
            ]
        )

    def test_execution(self):
        results = self.fee.execute_plan(self.dataset, self.plan)
        
        # Check all features are present
        self.assertIn("age", results)
        self.assertIn("zipcode", results)
        self.assertIn("income", results)
        
        # Check specific tests were run based on the plan
        self.assertIn("predictive_impact", results["age"])
        self.assertIn("mutual_information", results["age"])
        self.assertNotIn("causal_indicator", results["age"]) # False in plan
        
        self.assertIn("causal_indicator", results["zipcode"]) # True in plan
        
        # Verify metric reasonability
        income_impact = results["income"]["predictive_impact"]
        self.assertGreater(income_impact, 0.0)
        
        # Zipcode should be causally linked
        zipcode_causal = results["zipcode"]["causal_indicator"]
        self.assertIn("p_value", zipcode_causal)
        self.assertIn("is_causally_linked", zipcode_causal)
        
        # Distribution checks
        self.assertEqual(results["age"]["distribution"]["missing_percentage"], 0.0)

if __name__ == '__main__':
    unittest.main()
