import numpy as np
from typing import Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class ContextDetector:
    """
    Detects the domain context of a dataset using Zero-Shot Embedding mapping.
    It generates a dataset semantic signature from metadata and compares it 
    against predefined domain anchors using cosine similarity.
    """
    
    # Predefined domain anchors with rich semantic descriptions
    DOMAIN_ANCHORS = {
        "Finance": "credit score, loan amount, default risk, banking, finance, mortgage, interest rate, income",
        "Healthcare": "patient, diagnosis, treatment, blood pressure, disease, medical history, clinical trial, symptom",
        "Hiring": "resume, interview score, previous experience, education, salary, performance review, promotion, job title",
        "Education": "grades, gpa, student, admission, test score, attendance, degree, school, university",
        "Insurance": "claim amount, premium, coverage, policy holder, accident history, deductible, risk profile"
    }

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the ContextDetector with a specific embedding model.
        
        Args:
            model_name (str): The name of the sentence-transformers model to use.
        """
        self.model_name = model_name
        logger.info(f"Loading sentence-transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Precompute embeddings for domain anchors
        self.domains = list(self.DOMAIN_ANCHORS.keys())
        anchor_texts = list(self.DOMAIN_ANCHORS.values())
        self.domain_embeddings = self.model.encode(anchor_texts)
        logger.debug(f"Precomputed embeddings for {len(self.domains)} domains.")

    def _generate_dataset_signature(self, dataset_metadata: Dict[str, Any]) -> str:
        """
        Converts the dataset metadata into a single descriptive string.
        
        Args:
            dataset_metadata (Dict[str, Any]): A dictionary containing column names, types, etc.
                Expected format: {'columns': ['age', 'loan_amount', 'default']}
                
        Returns:
            str: A concatenated string of column names and descriptions representing the dataset.
        """
        columns = dataset_metadata.get('columns', [])
        if not columns:
            logger.warning("Dataset metadata provided no columns.")
            return ""
            
        # A simple concatenation of column names works well for semantic signature
        signature = " ".join([str(col).replace('_', ' ') for col in columns])
        return signature

    def detect_context(self, dataset_metadata: Dict[str, Any]) -> Tuple[str, float]:
        """
        Detects the most likely domain for the given dataset metadata.
        
        Args:
            dataset_metadata (Dict[str, Any]): Dictionary containing dataset info.
                Must include a 'columns' key with a list of column names.
                
        Returns:
            Tuple[str, float]: The detected domain name and the confidence score (0 to 1).
            Returns ("Unknown", 0.0) if detection fails or confidence is too low.
        """
        signature = self._generate_dataset_signature(dataset_metadata)
        if not signature:
            return "Unknown", 0.0
            
        signature_embedding = self.model.encode([signature])
        
        # Calculate cosine similarity against all domains
        similarities = cosine_similarity(signature_embedding, self.domain_embeddings)[0]
        
        best_match_idx = int(np.argmax(similarities))
        confidence = float(similarities[best_match_idx])
        detected_domain = self.domains[best_match_idx]
        
        # We can set a reasonable threshold, e.g., 0.15 for zero-shot text matching
        if confidence < 0.15:
            logger.warning(f"Low confidence ({confidence:.2f}) for domain {detected_domain}. Defaulting to Unknown.")
            return "Unknown", confidence
            
        logger.info(f"Detected context: {detected_domain} (Confidence: {confidence:.2f})")
        return detected_domain, confidence
