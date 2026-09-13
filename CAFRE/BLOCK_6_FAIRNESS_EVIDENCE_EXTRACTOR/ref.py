import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class RegisteredFairnessEvidences:
    """
    Registered Fairness Evidences (REF).
    Defines and computes standard mathematical fairness metrics on a dataset,
    both for ground truth outcomes (dataset bias) and model predictions (model bias).
    """

    @staticmethod
    def detect_positive_outcome(series: pd.Series) -> Any:
        """
        Autonomously detects which value in a target series represents the positive outcome.
        """
        unique_vals = series.dropna().unique()
        if len(unique_vals) == 0:
            return None
        
        # Priority 1: Common positive strings/numbers
        positive_keywords = {"1", 1, "hired", "approved", "yes", "pass", ">50k", "true", "selected"}
        for val in unique_vals:
            if str(val).lower().strip() in [str(k).lower() for k in positive_keywords]:
                return val
                
        # Priority 2: Invert negative keywords
        negative_keywords = {"0", 0, "rejected", "denied", "no", "fail", "<=50k", "false", "unselected"}
        for val in unique_vals:
            if str(val).lower().strip() not in [str(k).lower() for k in negative_keywords]:
                # If there's only two classes and one is negative, return the other
                if len(unique_vals) == 2:
                    return val

        # Fallback: Sort alphabetically and take the last one (e.g. 'Yes' over 'No', 'hired' over 'not_hired')
        sorted_vals = sorted([str(v) for v in unique_vals])
        for val in unique_vals:
            if str(val) == sorted_vals[-1]:
                return val
                
        return unique_vals[0]

    def __init__(self):
        pass

    def compute_metrics(
        self,
        df: pd.DataFrame,
        protected_col: str,
        target_col: str,
        predictions: Optional[np.ndarray] = None,
        positive_outcome: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Computes standard fairness metrics for a protected attribute.
        
        Args:
            df: The dataset containing protected_col and target_col.
            protected_col: The sensitive attribute column name.
            target_col: The target label column name.
            predictions: Optional predictions array. If provided, model bias is evaluated.
            positive_outcome: Optional positive outcome label. If None, auto-detected.
            
        Returns:
            Dict containing disparate_impact, demographic_parity_difference,
            equal_opportunity_difference, equalized_odds_difference, and details.
        """
        if protected_col not in df.columns:
            logger.warning(f"Protected column '{protected_col}' not found in DataFrame.")
            return {}
            
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found in DataFrame.")
            return {}

        temp_df = df.copy()
        
        # Binarize continuous targets dynamically to > median
        if pd.api.types.is_numeric_dtype(temp_df[target_col]) and temp_df[target_col].nunique() > 20:
            median_val = temp_df[target_col].median()
            logger.info(f"Target '{target_col}' is continuous. Binarizing at median: > {median_val}")
            temp_df[target_col] = (temp_df[target_col] > median_val).astype(int)
            positive_outcome = 1
        
        # Auto-detect positive outcome if not specified
        if positive_outcome is None:
            positive_outcome = self.detect_positive_outcome(temp_df[target_col])
            
        # Determine actual outcome series (binary 0/1 representation)
        y_true = (temp_df[target_col] == positive_outcome).astype(int)
        
        # Determine decision outcome series (binary 0/1 representation)
        if predictions is not None:
            # Align predictions with dataframe shape
            if len(predictions) == len(temp_df):
                y_dec = (predictions == positive_outcome).astype(int)
            else:
                logger.error("Predictions length does not match DataFrame. Falling back to ground truth.")
                y_dec = y_true
        else:
            y_dec = y_true
            
        temp_df["_y_true"] = y_true
        temp_df["_y_dec"] = y_dec

        # Dynamic Identification of Privileged and Unprivileged Groups
        # Privileged = group with highest selection rate in decisions
        groups = temp_df[protected_col].dropna().unique()
        if len(groups) < 2:
            logger.warning(f"Protected attribute '{protected_col}' has fewer than 2 groups. Cannot compute fairness metrics.")
            return {
                "disparate_impact": 1.0,
                "demographic_parity_difference": 0.0,
                "equal_opportunity_difference": 0.0,
                "equalized_odds_difference": 0.0,
                "details": {"message": "Insufficient groups"}
            }

        selection_rates = {}
        for g in groups:
            group_df = temp_df[temp_df[protected_col] == g]
            if len(group_df) == 0:
                selection_rates[g] = 0.0
            else:
                selection_rates[g] = float(group_df["_y_dec"].mean())

        # Sort groups by selection rate
        sorted_groups = sorted(selection_rates.items(), key=lambda x: x[1], reverse=True)
        privileged_group = sorted_groups[0][0]
        privileged_rate = sorted_groups[0][1]
        
        # Unprivileged = group with lowest selection rate
        unprivileged_group = sorted_groups[-1][0]
        unprivileged_rate = sorted_groups[-1][1]

        logger.debug(f"Attribute: '{protected_col}' | Privileged: '{privileged_group}' ({privileged_rate:.3f}) | Unprivileged: '{unprivileged_group}' ({unprivileged_rate:.3f})")

        # 1. Disparate Impact (DI)
        if privileged_rate == 0.0:
            disparate_impact = 1.0
        else:
            disparate_impact = unprivileged_rate / privileged_rate

        # 2. Demographic Parity Difference (DPD)
        demographic_parity_diff = privileged_rate - unprivileged_rate

        # 3. Equal Opportunity Difference (EOD)
        # Difference in True Positive Rates (TPR) between groups
        # TPR = P(y_dec = 1 | y_true = 1)
        tpr_rates = {}
        fpr_rates = {}
        for g in [privileged_group, unprivileged_group]:
            group_df = temp_df[temp_df[protected_col] == g]
            actual_positives = group_df[group_df["_y_true"] == 1]
            actual_negatives = group_df[group_df["_y_true"] == 0]
            
            if len(actual_positives) == 0:
                tpr_rates[g] = 1.0  # Avoid division by zero
            else:
                tpr_rates[g] = float(actual_positives["_y_dec"].mean())
                
            if len(actual_negatives) == 0:
                fpr_rates[g] = 0.0
            else:
                fpr_rates[g] = float(actual_negatives["_y_dec"].mean())

        equal_opportunity_diff = abs(tpr_rates[privileged_group] - tpr_rates[unprivileged_group])

        # 4. Equalized Odds Difference (EODD)
        # Max of |TPR_p - TPR_u| and |FPR_p - FPR_u|
        fpr_diff = abs(fpr_rates[privileged_group] - fpr_rates[unprivileged_group])
        equalized_odds_diff = max(equal_opportunity_diff, fpr_diff)

        return {
            "disparate_impact": disparate_impact,
            "demographic_parity_difference": demographic_parity_diff,
            "equal_opportunity_difference": equal_opportunity_diff,
            "equalized_odds_difference": equalized_odds_diff,
            "details": {
                "privileged_group": str(privileged_group),
                "privileged_rate": privileged_rate,
                "unprivileged_group": str(unprivileged_group),
                "unprivileged_rate": unprivileged_rate,
                "privileged_tpr": tpr_rates[privileged_group],
                "unprivileged_tpr": tpr_rates[unprivileged_group],
                "privileged_fpr": fpr_rates[privileged_group],
                "unprivileged_fpr": fpr_rates[unprivileged_group]
            }
        }
