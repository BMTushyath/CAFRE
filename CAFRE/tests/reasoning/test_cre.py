import unittest
import pandas as pd
from CAFRE.reasoning.cre import ContextAwareReasoningEngine

class TestContextAwareReasoningEngine(unittest.TestCase):
    def setUp(self):
        self.ontology = {
            "domain": "Finance",
            "protected_attributes": ["gender", "race"],
            "historical_biases": ["Redlining"],
            "known_proxies": {
                "gender": ["prefix"],
                "race": ["zipcode"]
            }
        }
        self.cre = ContextAwareReasoningEngine(
            domain="Finance",
            cke_ontology=self.ontology,
            target_col="loan_approved"
        )
        
        # Simple dataset
        self.df = pd.DataFrame({
            "gender": [1, 0, 1, 0],
            "zipcode": [10001, 10002, 10001, 10002],
            "income": [50000, 60000, 50000, 70000],
            "loan_approved": [1, 1, 0, 0]
        })
        
        # Mock FEE outputs
        self.fee_results = {
            "gender": {"predictive_impact": 0.02, "mutual_information": 0.05, "causal_indicator": {"is_causally_linked": False}},
            "zipcode": {"predictive_impact": 0.08, "mutual_information": 0.12, "causal_indicator": {"is_causally_linked": True, "p_value": 0.01}},
            "income": {"predictive_impact": 0.15, "mutual_information": 0.18, "causal_indicator": {"is_causally_linked": True, "p_value": 0.005}}
        }
        
        # Mock REF outputs
        self.ref_results = {
            "gender": {"disparate_impact": 0.85, "demographic_parity_difference": 0.05, "equal_opportunity_difference": 0.02}
        }

    def test_cre_analysis(self):
        results = self.cre.analyze_dataset(self.df, self.fee_results, self.ref_results)
        
        # Sort results by column name for ease of test assertions
        res_dict = {r.column_name: r for r in results}
        
        # Assert roles and categories
        self.assertTrue(res_dict["gender"].is_protected)
        self.assertEqual(res_dict["gender"].semantic_category, "Demographic")
        
        self.assertTrue(res_dict["zipcode"].is_proxy)
        self.assertEqual(res_dict["zipcode"].proxy_for, "race")
        
        # Assert scoring and priorities
        # Gender final score should have been calculated
        self.assertGreater(res_dict["gender"].final_score, 0.0)
        
        # Income is a standard feature but should have higher statistical score due to impact
        self.assertGreater(res_dict["income"].stat_score, 0.0)

if __name__ == "__main__":
    unittest.main()
