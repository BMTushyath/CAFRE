import unittest
import tempfile
import os
from CAFRE.explainability.ferc import FERCExplainabilityEngine
from CAFRE.reasoning.cre import CREResult

class TestFERCExplainabilityEngine(unittest.TestCase):
    def setUp(self):
        self.ontology = {
            "domain": "Finance",
            "protected_attributes": ["gender"],
            "historical_biases": ["Redlining"],
            "known_proxies": {"gender": ["prefix"]}
        }
        
        # Mock results
        self.cre_results = [
            CREResult(
                column_name="gender",
                semantic_role="Feature",
                semantic_category="Demographic",
                is_protected=True,
                is_proxy=False,
                proxy_for=None,
                base_score=80.0,
                stat_score=75.0,
                final_score=78.0,
                confidence=95.0,
                priority="HIGH",
                is_uncertain=False,
                uncertainty_reasons=[],
                trace=["Semantics: gender is protected", "Evidence: stat score is 75.0"]
            )
        ]
        
        self.afl_actions = [
            {
                "iteration": 1,
                "feature": "gender",
                "action": "ValidateWithAlternativeClassifier",
                "objectives": {"evidence_gain": 6.0, "confidence_improvement": 8.0, "explainability_benefit": 5.0, "cost": 3.0},
                "result": {"alternative_model_impact": 0.05}
            }
        ]
        
        self.fee_results = {"gender": {"predictive_impact": 0.05}}
        self.ref_results = {"gender": {"disparate_impact": 0.5}}

    def test_report_generation(self):
        engine = FERCExplainabilityEngine(
            domain="Finance",
            target_col="loan_approved",
            cke_ontology=self.ontology,
            cre_results=self.cre_results,
            afl_actions=self.afl_actions,
            fee_results=self.fee_results,
            ref_results=self.ref_results
        )
        
        json_report = engine.compile_json_report()
        self.assertEqual(json_report["metadata"]["domain"], "Finance")
        self.assertEqual(len(json_report["reasoning_results"]), 1)
        
        md_report = engine.generate_markdown_report(json_report)
        self.assertIn("CAFRE Explainable Fairness Audit Report", md_report)
        self.assertIn("gender", md_report)
        self.assertIn("Disparate Impact Ratio", md_report)

        # Test writing files
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path, md_path = engine.write_reports(output_dir=tmpdir)
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(md_path))

if __name__ == "__main__":
    unittest.main()
