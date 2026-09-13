import unittest
import os
import tempfile
import json
from unittest.mock import patch
from pydantic import ValidationError
from CAFRE.knowledge.leco_generator import CKEGenerator, CKEOntology

class TestCKEGenerator(unittest.TestCase):
    def setUp(self):
        # Use a temporary directory for cache to avoid polluting real cache during tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self.generator = CKEGenerator(cache_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_schema_validation_success(self):
        valid_data = {
            "domain": "Finance",
            "protected_attributes": ["race", "gender"],
            "historical_biases": ["redlining", "gender-based interest rates"],
            "known_proxies": {
                "race": ["zipcode"],
                "gender": ["name"]
            }
        }
        # Should instantiate without error
        ontology = CKEOntology(**valid_data)
        self.assertEqual(ontology.domain, "Finance")
        self.assertEqual(len(ontology.protected_attributes), 2)

    def test_schema_validation_failure(self):
        invalid_data = {
            "domain": "Finance",
            # missing protected_attributes
            "historical_biases": ["redlining"],
            "known_proxies": {}
        }
        with self.assertRaises(ValidationError):
            CKEOntology(**invalid_data)

    def test_cache_hit(self):
        # Pre-populate cache
        domain = "Finance"
        valid_data = {
            "domain": domain,
            "protected_attributes": ["race"],
            "historical_biases": ["redlining"],
            "known_proxies": {"race": ["zipcode"]}
        }
        cache_path = self.generator._get_cache_path(domain)
        with open(cache_path, 'w') as f:
            json.dump(valid_data, f)

        # Call get_ontology, it should return the cached data without invoking LLM
        with patch.object(self.generator, '_generate_ontology_with_llm') as mock_llm:
            result = self.generator.get_ontology(domain)
            mock_llm.assert_not_called()
            self.assertEqual(result["domain"], "Finance")

    def test_llm_invocation_and_caching(self):
        # Test that if cache misses, LLM is called and result is validated and cached
        domain = "Healthcare"
        mock_llm_output = {
            "domain": domain,
            "protected_attributes": ["age", "disability"],
            "historical_biases": ["denial of care"],
            "known_proxies": {"age": ["years_since_grad"]}
        }

        with patch.object(self.generator, '_generate_ontology_with_llm', return_value=mock_llm_output) as mock_llm:
            result = self.generator.get_ontology(domain)
            mock_llm.assert_called_once_with(domain)
            
            # Verify result matches
            self.assertEqual(result["domain"], "Healthcare")
            
            # Verify it was cached
            cache_path = self.generator._get_cache_path(domain)
            self.assertTrue(os.path.exists(cache_path))
            
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
                # Verify that the order is preserved exactly
                self.assertEqual(cached_data["protected_attributes"], ["age", "disability"])

    def test_duplicate_resolution(self):
        domain = "Hiring"
        mock_llm_output = {
            "domain": domain,
            "protected_attributes": ["gender", "gender"], # Duplicate
            "historical_biases": [],
            "known_proxies": {"race": ["zipcode"]} # Race is not in protected_attributes
        }
        
        with patch.object(self.generator, '_generate_ontology_with_llm', return_value=mock_llm_output):
            result = self.generator.get_ontology(domain)
            
            # Should deduplicate gender, and auto-add race
            self.assertIn("gender", result["protected_attributes"])
            self.assertIn("race", result["protected_attributes"])
            
            # Ensure only 2 elements exist
            self.assertEqual(len(result["protected_attributes"]), 2)

if __name__ == '__main__':
    unittest.main()
