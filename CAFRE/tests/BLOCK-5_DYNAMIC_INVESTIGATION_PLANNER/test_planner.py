import unittest
from CAFRE.investigation.planner import InvestigationPlanner

class TestInvestigationPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = InvestigationPlanner()
        
        self.dataset_metadata = {
            "columns": ["age", "income", "zipcode", "credit_score", "loan_approved"]
        }
        
        self.leco_ontology = {
            "domain": "Finance",
            "protected_attributes": ["race", "gender", "age"],
            "historical_biases": ["redlining"],
            "known_proxies": {
                "race": ["zipcode"]
            }
        }
        
        self.target_variable = "loan_approved"

    def test_plan_generation(self):
        plan = self.planner.generate_plan(
            context="Finance",
            dataset_metadata=self.dataset_metadata,
            leco_ontology=self.leco_ontology,
            target_variable=self.target_variable
        )
        
        self.assertEqual(plan.domain, "Finance")
        self.assertEqual(plan.target_variable, "loan_approved")
        
        # We have 5 columns, 1 is target, so we should have 4 feature plans
        self.assertEqual(len(plan.feature_plans), 4)
        
        # Convert list of plans to a dict for easy checking
        plans_dict = {fp.feature_name: fp for fp in plan.feature_plans}
        
        # 'age' is protected. Should have all tests including causal
        self.assertTrue(plans_dict["age"].run_causal_discovery)
        self.assertTrue(plans_dict["age"].run_shap)
        
        # 'zipcode' is a known proxy. Should have all tests including causal
        self.assertTrue(plans_dict["zipcode"].run_causal_discovery)
        
        # 'income' and 'credit_score' are standard. Should NOT have causal discovery
        self.assertFalse(plans_dict["income"].run_causal_discovery)
        self.assertFalse(plans_dict["credit_score"].run_causal_discovery)
        
        # But they should have SHAP
        self.assertTrue(plans_dict["income"].run_shap)
        self.assertTrue(plans_dict["credit_score"].run_shap)
        
        # All should have distribution analysis
        for fp in plan.feature_plans:
            self.assertTrue(fp.run_distribution_analysis)

if __name__ == '__main__':
    unittest.main()
