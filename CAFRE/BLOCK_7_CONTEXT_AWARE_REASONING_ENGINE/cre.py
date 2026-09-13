import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CREResult:
    column_name: str
    semantic_role: str
    semantic_category: str
    is_protected: bool
    is_proxy: bool
    proxy_for: Optional[str]
    base_score: float
    stat_score: float
    final_score: float
    confidence: float
    priority: str
    is_uncertain: bool
    uncertainty_reasons: List[str]
    trace: List[str]

class ContextAwareReasoningEngine:
    """
    Context-Aware Reasoning Engine (CRE).
    Combines semantic metadata (CKE), plan goals (DIP), and empirical evidence (FEE, REF)
    to perform multi-dimensional reasoning about bias and proxies, estimating confidence
    and flagging uncertainty.
    """
    def __init__(self, domain: str, cke_ontology: Dict[str, Any], target_col: str):
        self.domain = domain
        self.ontology = cke_ontology
        self.target_col = target_col
        self.protected_attributes = cke_ontology.get("protected_attributes", [])
        self.known_proxies = cke_ontology.get("known_proxies", {})
        
        # Build inverse map for proxies: proxy -> protected_attr
        self.proxy_map = {}
        for protected_attr, proxies in self.known_proxies.items():
            for proxy in proxies:
                self.proxy_map[proxy] = protected_attr

    def compute_statistical_score(self, col: str, fee_data: Dict[str, Any], ref_data: Optional[Dict[str, Any]] = None) -> float:
        """
        Aggregates raw empirical statistics (FEE) and fairness metrics (REF) into a 0-100 score.
        """
        scores = []
        
        # 1. Predictive Impact (permutation importance)
        pred_impact = fee_data.get("predictive_impact", 0.0)
        # Normalize: permutation importance > 0.05 is significant, scale 0.0-0.1 to 0-100
        scores.append(min(pred_impact * 1000.0, 100.0))
        
        # 2. Dependency (mutual information)
        mi = fee_data.get("mutual_information", 0.0)
        # Normalize: MI > 0.1 is significant, scale 0.0-0.2 to 0-100
        scores.append(min(mi * 500.0, 100.0))
        
        # 3. Causal indicator
        causal = fee_data.get("causal_indicator", {})
        if causal.get("is_causally_linked", False):
            # Based on p-value
            p_val = causal.get("p_value", 1.0)
            scores.append(max(0.0, (1.0 - p_val) * 100.0))
            
        # 4. Fairness Metrics (REF) - only applicable if computed
        if ref_data:
            # Disparate Impact
            di = ref_data.get("disparate_impact", 1.0)
            di_deviation = abs(1.0 - di)
            # Scale DI deviation: 0.2 deviation (0.8 or 1.2) maps to 100
            scores.append(min(di_deviation * 500.0, 100.0))
            
            # Demographic Parity Difference
            dpd = ref_data.get("demographic_parity_difference", 0.0)
            scores.append(min(dpd * 500.0, 100.0))
            
            # Equal Opportunity Difference
            eod = ref_data.get("equal_opportunity_difference", 0.0)
            scores.append(min(eod * 500.0, 100.0))
            
        if not scores:
            return 0.0
            
        return float(np.mean(scores))

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        fee_results: Dict[str, Dict[str, Any]],
        ref_results: Dict[str, Dict[str, Any]]
    ) -> List[CREResult]:
        """
        Executes reasoning for each column in the dataset and outputs the synthesized result.
        """
        logger.info(f"Context-Aware Reasoning Engine executing for domain: {self.domain}")
        results = []
        
        for col in df.columns:
            if col == self.target_col:
                # Target variable is ignored for bias scan (it is the outcome itself)
                continue
                
            trace = []
            is_protected = col in self.protected_attributes
            is_proxy = col in self.proxy_map
            proxy_for = self.proxy_map.get(col)
            
            # Determine base semantic score
            if is_protected:
                base_score = 80.0
                trace.append(f"Semantics: '{col}' is a declared protected attribute in {self.domain} ontology.")
            elif is_proxy:
                base_score = 60.0
                trace.append(f"Semantics: '{col}' is a known proxy for protected attribute '{proxy_for}' in {self.domain} ontology.")
            elif "id" in col.lower() or "uuid" in col.lower() or col.lower() == "candidate_id":
                base_score = 0.0
                trace.append(f"Semantics: '{col}' is classified as an identifier.")
            else:
                base_score = 30.0
                trace.append(f"Semantics: '{col}' is a standard feature column.")

            # Skip identifiers from detailed evaluation
            if base_score == 0.0:
                results.append(CREResult(
                    column_name=col,
                    semantic_role="Identifier",
                    semantic_category="Administrative",
                    is_protected=False,
                    is_proxy=False,
                    proxy_for=None,
                    base_score=0.0,
                    stat_score=0.0,
                    final_score=0.0,
                    confidence=100.0,
                    priority="IGNORE",
                    is_uncertain=False,
                    uncertainty_reasons=[],
                    trace=trace
                ))
                continue

            # Gather statistical evidence for the feature
            fee_data = fee_results.get(col, {})
            ref_data = ref_results.get(col)
            
            stat_score = self.compute_statistical_score(col, fee_data, ref_data)
            trace.append(f"Evidence: Empirical statistics aggregated score is {stat_score:.1f}/100.")
            
            # Additional Causal/Dependency Proxy Analysis for proxy suspects
            # If standard feature has high correlation with a protected attribute, flag it.
            potential_hidden_proxy = False
            proxy_target = None
            if not is_protected and not is_proxy:
                # Check correlation/dependency with each protected attribute in the dataset
                for prot_col in self.protected_attributes:
                    if prot_col in df.columns:
                        try:
                            # Using spearman correlation as a dependency proxy
                            corr = abs(df[col].astype(float).corr(df[prot_col].astype(float), method='spearman'))
                            if corr > 0.4:
                                potential_hidden_proxy = True
                                proxy_target = prot_col
                                trace.append(f"Warning: Standard feature '{col}' exhibits high correlation ({corr:.2f}) with protected attribute '{prot_col}'.")
                        except:
                            pass

            if potential_hidden_proxy:
                base_score += 15.0
                trace.append(f"-> Increased semantic risk due to correlation with protected attribute '{proxy_target}'.")

            # Final Score Fusion: 60% semantic expectation, 40% statistical evidence
            final_score = (base_score * 0.6) + (stat_score * 0.4)
            trace.append(f"Fusion: Combined final bias score is {final_score:.1f}/100.")

            # Priority Assignment
            if final_score >= 70.0:
                priority = "HIGH"
            elif final_score >= 40.0:
                priority = "MEDIUM"
            else:
                priority = "LOW"
            trace.append(f"Decision: Assigned priority level {priority}.")

            # Confidence Estimation (Agreement logic)
            # High semantic risk + high stats = high confidence
            # Low semantic risk + low stats = high confidence
            # Conflict (high semantic risk + low stats OR low semantic risk + high stats) = low confidence
            diff = abs(base_score - stat_score)
            confidence = max(0.0, 100.0 - diff)
            if diff > 35:
                trace.append(f"Warning: Conflict detected between semantic expectations ({base_score:.1f}) and empirical evidence ({stat_score:.1f}). Confidence reduced to {confidence:.1f}%.")
            else:
                trace.append(f"Agreement: Semantics and statistics agree. Confidence is {confidence:.1f}%.")

            # Uncertainty Handling
            is_uncertain = False
            uncertainty_reasons = []
            
            # Missingness check
            missing_pct = fee_data.get("distribution", {}).get("missing_percentage", 0.0)
            if missing_pct > 0.2:
                is_uncertain = True
                reason = f"High missingness: {missing_pct*100:.1f}% data is missing."
                uncertainty_reasons.append(reason)
                trace.append(f"Uncertainty: {reason}")
                
            # Sample size check
            if len(df) < 50:
                is_uncertain = True
                reason = f"Small dataset size: only {len(df)} samples available."
                uncertainty_reasons.append(reason)
                trace.append(f"Uncertainty: {reason}")

            if is_uncertain:
                confidence *= 0.7  # Penalize confidence
                trace.append(f"-> Confidence penalized to {confidence:.1f}% due to data quality uncertainty.")

            # Classify category
            category = "Demographic" if is_protected else "Proxy" if is_proxy else "Feature"

            results.append(CREResult(
                column_name=col,
                semantic_role="Feature",
                semantic_category=category,
                is_protected=is_protected,
                is_proxy=is_proxy,
                proxy_for=proxy_for,
                base_score=base_score,
                stat_score=stat_score,
                final_score=final_score,
                confidence=confidence,
                priority=priority,
                is_uncertain=is_uncertain,
                uncertainty_reasons=uncertainty_reasons,
                trace=trace
            ))

        # Sort results by final score descending
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results
