import unittest
from CAFRE.context.detector import ContextDetector

class TestContextDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize once for all tests to save model loading time
        # This will download the model if not cached, which is expected during testing.
        cls.detector = ContextDetector()

    def test_finance_domain(self):
        metadata = {
            "columns": ["applicant_age", "credit_history", "loan_amount", "interest_rate", "defaulted"]
        }
        domain, confidence = self.detector.detect_context(metadata)
        self.assertEqual(domain, "Finance")
        self.assertGreater(confidence, 0.3)

    def test_healthcare_domain(self):
        metadata = {
            "columns": ["patient_id", "blood_pressure", "bmi", "disease_severity", "treatment_outcome"]
        }
        domain, confidence = self.detector.detect_context(metadata)
        self.assertEqual(domain, "Healthcare")
        self.assertGreater(confidence, 0.3)

    def test_empty_metadata(self):
        metadata = {"columns": []}
        domain, confidence = self.detector.detect_context(metadata)
        self.assertEqual(domain, "Unknown")
        self.assertEqual(confidence, 0.0)
        
    def test_unknown_domain(self):
        # A domain completely unrelated to the anchors
        metadata = {
            "columns": ["alien_species", "spaceship_model", "lightyears_traveled", "plasma_cannon_power"]
        }
        domain, confidence = self.detector.detect_context(metadata)
        
        # We assert it runs without error. Depending on model, it might still map to something
        # with low confidence, or drop to 'Unknown'.
        self.assertIsInstance(domain, str)
        self.assertIsInstance(confidence, float)

if __name__ == '__main__':
    unittest.main()
