"""
semantic_engine.py — Column Understanding via Knowledge Base Mapping
======================================================================

Instead of statistical guessing, this engine asks: "What does this concept mean?"
It loads the external Knowledge Base (`knowledge.json`) and uses fuzzy string 
matching heuristics to map dataset columns to known semantic concepts.

This explicitly enables autonomous Target/Outcome detection based on meaning
rather than just entropy/cardinality.
"""

import json
import os
import difflib
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

try:
    from . import config, utils
except ImportError:
    import config  # type: ignore
    import utils   # type: ignore

@dataclass
class SemanticConcept:
    name: str
    role: str           # e.g., Feature, Identifier, Outcome, Decision, Derived
    category: str       # e.g., Demographic, Employment, Financial
    is_sensitive: bool
    aliases: List[str]

@dataclass
class ColumnSemantics:
    column_name: str
    concept_match: Optional[SemanticConcept]
    match_confidence: float
    detected_dtype: str
    inferred_role: str

class SemanticEngine:
    def __init__(self, kb_path: str = "knowledge.json"):
        self.kb_path = os.path.join(os.path.dirname(__file__), kb_path)
        self.concepts: List[SemanticConcept] = []
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Load the ontology from the JSON file."""
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge Base not found: {self.kb_path}")
        
        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for name, props in data.get("concepts", {}).items():
            self.concepts.append(
                SemanticConcept(
                    name=name,
                    role=props.get("role", "Feature"),
                    category=props.get("category", "Unknown"),
                    is_sensitive=props.get("is_sensitive", False),
                    aliases=props.get("aliases", [])
                )
            )

    def understand_dataset(self, df: pd.DataFrame) -> List[ColumnSemantics]:
        """Analyze all columns and assign semantic meaning."""
        results = []
        for col in df.columns:
            dtype = utils.detect_column_dtype(df[col])
            
            # Semantic matching
            best_concept = None
            best_score = 0.0
            
            col_lower = col.lower().replace("_", " ")
            
            for concept in self.concepts:
                for alias in concept.aliases:
                    alias_lower = alias.lower().replace("_", " ")
                    # Fuzzy match ratio (0-100)
                    matcher = difflib.SequenceMatcher(None, col_lower, alias_lower)
                    score = matcher.ratio() * 100.0
                    
                    # Exact substring match bonus
                    if alias_lower in col_lower or col_lower in alias_lower:
                        score = max(score, 85.0)
                        
                    if score > best_score:
                        best_score = score
                        best_concept = concept

            # Threshold logic
            if best_score >= config.SEMANTIC_MATCH_THRESHOLD and best_concept:
                inferred_role = best_concept.role
                match_conf = best_score
            else:
                best_concept = None
                match_conf = 0.0
                # Fallback heuristics if KB doesn't know it
                if "id" in col_lower or "uuid" in col_lower:
                    inferred_role = "Identifier"
                else:
                    inferred_role = "Feature"

            results.append(
                ColumnSemantics(
                    column_name=col,
                    concept_match=best_concept,
                    match_confidence=match_conf,
                    detected_dtype=dtype,
                    inferred_role=inferred_role
                )
            )
            
        return results

    def identify_primary_outcome(self, semantics: List[ColumnSemantics]) -> Optional[str]:
        """
        Autonomously identify the target purely based on semantic meaning.
        Returns the first column explicitly defined as 'Outcome' in the KB.
        """
        outcomes = [s for s in semantics if s.inferred_role == "Outcome"]
        if outcomes:
            # Sort by match confidence descending
            outcomes.sort(key=lambda x: x.match_confidence, reverse=True)
            return outcomes[0].column_name
        return None
