import unittest
from CAFRE.mitigation.engine import BiasMitigationEngine
from CAFRE.reasoning.cre import CREResult

class TestBiasMitigationEngine(unittest.TestCase):
    def test_mitigation_recommendations(self):
        engine = BiasMitigationEngine(domain="Hiring")
        
        cre_results = [
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
                trace=[]
            ),
            CREResult(
                column_name="zipcode",
                semantic_role="Feature",
                semantic_category="Proxy",
                is_protected=False,
                is_proxy=True,
                proxy_for="race",
                base_score=60.0,
                stat_score=65.0,
                final_score=62.0,
                confidence=90.0,
                priority="MEDIUM",
                is_uncertain=False,
                uncertainty_reasons=[],
                trace=[]
            )
        ]
        
        recs = engine.generate_recommendations(cre_results)
        self.assertEqual(len(recs), 2)
        
        # Test that gender has preprocessing reweighting
        gender_rec = [r for r in recs if r["feature"] == "gender"][0]
        strategies = [s["strategy_name"] for s in gender_rec["recs"]]
        self.assertIn("Sample Reweighting", strategies)
        
        # Test that zipcode has proxy suppression
        zip_rec = [r for r in recs if r["feature"] == "zipcode"][0]
        zip_strategies = [s["strategy_name"] for s in zip_rec["recs"]]
        self.assertIn("Proxy Feature Suppression / Masking", zip_strategies)

if __name__ == "__main__":
    unittest.main()
