import logging
from typing import Dict, Any, List
from CAFRE.reasoning.cre import CREResult

logger = logging.getLogger(__name__)

class BiasMitigationEngine:
    """
    Bias Mitigation Engine (Beta Advisory).
    Analyzes the CRE reasoning results and generates a prioritized, domain-aware set of
    corrective preprocessing, in-processing, and post-processing recommendations.
    Provides practical python code snippets for developer integration.
    """
    def __init__(self, domain: str):
        self.domain = domain

    def generate_recommendations(self, cre_results: List[CREResult]) -> List[Dict[str, Any]]:
        """
        Scans for biased features and builds tailored mitigation recommendations.
        """
        logger.info(f"Bias Mitigation Engine generating suggestions for domain: {self.domain}")
        recommendations = []

        for r in cre_results:
            if r.priority == "IGNORE":
                continue
                
            # We focus recommendations on HIGH and MEDIUM bias priority columns
            if r.priority not in ("HIGH", "MEDIUM"):
                continue

            col = r.column_name
            is_protected = r.is_protected
            is_proxy = r.is_proxy
            proxy_for = r.proxy_for

            # Build feature-specific recommendations
            feat_recs = []

            # 1. Preprocessing recommendation
            if is_protected:
                feat_recs.append({
                    "strategy_name": "Sample Reweighting",
                    "type": "Preprocessing",
                    "priority": "HIGH",
                    "suitability_rationale": (
                        f"Since '{col}' is a protected attribute with statistical bias (final score {r.final_score:.1f}), "
                        "sample reweighting is highly suitable. It dynamically adjusts sample weights to balance outcomes "
                        "without distorting the underlying feature representations."
                    ),
                    "implementation_snippet": (
                        "from sklearn.utils.class_weight import compute_sample_weight\n"
                        "# Calculate weights to balance the protected attribute groups relative to target outcomes\n"
                        f"sample_weights = compute_sample_weight('balanced', df['{col}'].astype(str) + df[target_col].astype(str))\n"
                        "model.fit(X, y, sample_weight=sample_weights)"
                    )
                })
                
                feat_recs.append({
                    "strategy_name": "Resampling (Oversampling)",
                    "type": "Preprocessing",
                    "priority": "MEDIUM",
                    "suitability_rationale": (
                        f"Resampling can be used to balance representation of minority subgroups of '{col}' in the training set. "
                        "This helps prevent the classifier from overfitting to the majority group's patterns."
                    ),
                    "implementation_snippet": (
                        "from imblearn.over_sampling import SMOTE\n"
                        "smote = SMOTE(random_state=42)\n"
                        "X_resampled, y_resampled = smote.fit_resample(X, y)"
                    )
                })

            elif is_proxy:
                feat_recs.append({
                    "strategy_name": "Proxy Feature Suppression / Masking",
                    "type": "Preprocessing",
                    "priority": "HIGH",
                    "suitability_rationale": (
                        f"'{col}' is a known proxy for protected attribute '{proxy_for}'. Standard classifiers "
                        f"will exploit '{col}' to reconstruct the sensitive context, creating proxy discrimination. "
                        "Removing this feature, or masking its values, is the safest way to break the proxy path."
                    ),
                    "implementation_snippet": (
                        "# Drop the proxy column to prevent proxy leakage\n"
                        f"X_debiased = X.drop(columns=['{col}'])"
                    )
                })

            else:
                # Suspected proxy/standard feature
                feat_recs.append({
                    "strategy_name": "Adversarial Debiasing / Feature Orthogonalization",
                    "type": "In-processing",
                    "priority": "MEDIUM",
                    "suitability_rationale": (
                        f"'{col}' is a standard feature that has been flagged with bias risks. Instead of dropping it "
                        "and losing predictive value, an in-processing adversarial model or fairness constraint can be used "
                        "to minimize mutual information between model predictions and the protected attribute."
                    ),
                    "implementation_snippet": (
                        "# Example using Fairlearn's Exponentiated Gradient optimizer to enforce fairness constraints\n"
                        "from fairlearn.reductions import ExponentiatedGradient, DemographicParity\n"
                        "estimator = RandomForestClassifier(random_state=42)\n"
                        "mitigator = ExponentiatedGradient(estimator, constraints=DemographicParity())\n"
                        "mitigator.fit(X, y, sensitive_features=sensitive_series)"
                    )
                })

            # Add a post-processing recommendation for all high bias features
            if r.priority == "HIGH":
                feat_recs.append({
                    "strategy_name": "Reject Option Classification (ROC) / Equalized Odds Post-processing",
                    "type": "Post-processing",
                    "priority": "MEDIUM",
                    "suitability_rationale": (
                        "Post-processing is highly non-disruptive as it adjusts predictions after model training. "
                        f"For high-risk features like '{col}', shifting the decision threshold slightly for the "
                        "unprivileged group helps ensure demographic or odds parity is satisfied near the boundary."
                    ),
                    "implementation_snippet": (
                        "# Example post-processing adjustment\n"
                        "# Shift predictions for unprivileged group within the margin of uncertainty\n"
                        "unpriv_mask = (df[sensitive_col] == unprivileged_group)\n"
                        "probs = model.predict_proba(X)[:, 1]\n"
                        "new_preds = (probs > 0.5).astype(int)\n"
                        "# Apply a lower threshold to the unprivileged group to mitigate outcome disparity\n"
                        "new_preds[unpriv_mask] = (probs[unpriv_mask] > 0.45).astype(int)"
                    )
                })

            recommendations.append({
                "feature": col,
                "priority_level": r.priority,
                "recs": feat_recs
            })

        return recommendations
