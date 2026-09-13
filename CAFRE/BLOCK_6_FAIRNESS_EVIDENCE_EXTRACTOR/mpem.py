import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from scipy.stats import chi2_contingency

from CAFRE.BLOCK_5_DYNAMIC_INVESTIGATION_PLANNER.planner import InvestigationExecutionPlan

logger = logging.getLogger(__name__)

class FairnessEvidenceExtractor:
    """
    Fairness Evidence Extractor (FEE).
    Executes targeted statistical tests to generate evidence for cognitive reasoning.
    Provides Predictive Impact (Permutation Importance), Mutual Information (Dependency),
    and Causal approximations (Chi-Square independence test).
    """
    def __init__(self):
        self.model = None
        self.model_trained = False
        self.permutation_importances = None
        
    def _encode_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes categorical columns for statistical processing."""
        encoded_df = df.copy()
        for col in encoded_df.select_dtypes(include=['str', 'category', 'object']).columns:
            encoded_df[col] = LabelEncoder().fit_transform(encoded_df[col].astype(str))
        return encoded_df

    def _bin_if_continuous(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
            try:
                return pd.qcut(series, q=10, duplicates='drop').astype(str)
            except Exception:
                pass
        return series

    def _train_model_if_needed(self, X: pd.DataFrame, y: pd.Series):
        if not self.model_trained:
            logger.info("Training baseline model for predictive impact calculations...")
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
                self.model = RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=10)
            else:
                self.model = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10)
                
            self.model.fit(X, y)
            self.model_trained = True
            
            # Compute permutation importances once for all features to optimize compute
            result = permutation_importance(self.model, X, y, n_repeats=3, random_state=42, n_jobs=-1)
            self.permutation_importances = {col: imp for col, imp in zip(X.columns, result.importances_mean)}

    def _compute_predictive_impact(self, feature_name: str, X: pd.DataFrame) -> float:
        """Returns permutation importance for a feature."""
        if feature_name not in X.columns or self.permutation_importances is None:
            return 0.0
        return float(self.permutation_importances.get(feature_name, 0.0))

    def _compute_mutual_information(self, feature: pd.Series, target: pd.Series) -> float:
        binned_feature = self._bin_if_continuous(feature)
        binned_target = self._bin_if_continuous(target)
        return float(normalized_mutual_info_score(binned_feature, binned_target))
        
    def _compute_causal_indicator(self, feature: pd.Series, target: pd.Series) -> Dict[str, Any]:
        """
        Uses Chi-Square test of independence as a lightweight proxy for a causal link.
        """
        binned_feature = self._bin_if_continuous(feature)
        binned_target = self._bin_if_continuous(target)
        contingency_table = pd.crosstab(binned_feature, binned_target)
        if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            return {"p_value": 1.0, "is_causally_linked": False}
            
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        return {
            "p_value": float(p),
            "is_causally_linked": bool(p < 0.05)
        }
        
    def _compute_distribution(self, feature: pd.Series) -> Dict[str, Any]:
        return {
            "missing_percentage": float(feature.isnull().mean()),
            "unique_values": int(feature.nunique())
        }

    def execute_plan(self, dataset: pd.DataFrame, plan: InvestigationExecutionPlan) -> Dict[str, Any]:
        """
        Executes the provided investigation plan on the dataset using Fairness Evidence Extractor.
        """
        logger.info(f"Executing Fairness Evidence Extractor for domain: {plan.domain}")
        
        target_col = plan.target_variable
        if target_col not in dataset.columns:
            raise ValueError(f"Target variable '{target_col}' not found in dataset.")
            
        encoded_df = self._encode_dataset(dataset)
        encoded_df = encoded_df.dropna(subset=[target_col])
        
        # Impute missing values to prevent sklearn NaN errors
        encoded_df = encoded_df.fillna(0)
        
        X = encoded_df.drop(columns=[target_col])
        y = encoded_df[target_col]
        
        matrix_results = {}
        
        for feature_plan in plan.feature_plans:
            fname = feature_plan.feature_name
            if fname not in encoded_df.columns:
                logger.warning(f"Feature '{fname}' requested by planner but missing from dataset.")
                continue
                
            results = {}
            feature_series = encoded_df[fname]
            
            if feature_plan.run_distribution_analysis:
                results["distribution"] = self._compute_distribution(feature_series)
                
            if feature_plan.run_mutual_information:
                results["mutual_information"] = self._compute_mutual_information(feature_series, y)
                
            if feature_plan.run_causal_discovery:
                results["causal_indicator"] = self._compute_causal_indicator(feature_series, y)
                
            if feature_plan.run_shap:
                self._train_model_if_needed(X, y)
                results["predictive_impact"] = self._compute_predictive_impact(fname, X)
                
            matrix_results[fname] = results
            
        return matrix_results
