import logging
import abc
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Set
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)

class ModelValidator(abc.ABC):
    """
    Abstract interface for pluggable validation models.
    Enables pluggable models to be evaluated during the agentic feedback loop
    without modifying the AFL itself.
    """
    @abc.abstractmethod
    def validate_feature_impact(self, df: pd.DataFrame, target_col: str, feature_col: str) -> float:
        """
        Trains a model and computes the importance/impact score of the target feature.
        """
        pass

class LogisticRegressionValidator(ModelValidator):
    """
    Validation classifier using Logistic Regression.
    """
    def validate_feature_impact(self, df: pd.DataFrame, target_col: str, feature_col: str) -> float:
        try:
            # Drop NaN and encode categories
            clean_df = df[[feature_col, target_col]].dropna()
            if len(clean_df) < 10:
                return 0.0
            
            X = pd.get_dummies(clean_df[[feature_col]], drop_first=True)
            y = clean_df[target_col].astype(str)
            
            clf = LogisticRegression(random_state=42, max_iter=200)
            clf.fit(X, y)
            
            # Simple feature impact proxy: permutation importance
            result = permutation_importance(clf, X, y, n_repeats=3, random_state=42)
            return float(np.mean(result.importances_mean))
        except Exception as e:
            logger.debug(f"LogisticRegressionValidator failed for {feature_col}: {e}")
            return 0.0

class DecisionTreeValidator(ModelValidator):
    """
    Validation classifier using a Decision Tree.
    """
    def validate_feature_impact(self, df: pd.DataFrame, target_col: str, feature_col: str) -> float:
        try:
            clean_df = df[[feature_col, target_col]].dropna()
            if len(clean_df) < 10:
                return 0.0
            
            X = pd.get_dummies(clean_df[[feature_col]], drop_first=True)
            y = clean_df[target_col].astype(str)
            
            clf = DecisionTreeClassifier(max_depth=4, random_state=42)
            clf.fit(X, y)
            
            result = permutation_importance(clf, X, y, n_repeats=3, random_state=42)
            return float(np.mean(result.importances_mean))
        except Exception as e:
            logger.debug(f"DecisionTreeValidator failed for {feature_col}: {e}")
            return 0.0

class AFLAction(abc.ABC):
    """
    Abstract representation of an agentic investigation candidate action.
    """
    def __init__(self, action_name: str, target_feature: str):
        self.action_name = action_name
        self.target_feature = target_feature

    @abc.abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """
        Returns objective scores for the action:
        - evidence_gain (0 to 10)
        - confidence_improvement (0 to 10)
        - explainability_benefit (0 to 10)
        - cost (0 to 10, lower is cheaper)
        """
        pass

    @abc.abstractmethod
    def execute(self, df: pd.DataFrame, target_col: str, current_fee_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the investigation action and returns new evidence to merge.
        """
        pass

class ValidateWithAlternativeClassifier(AFLAction):
    """
    Trains alternative models (Logistic Regression, Decision Tree) to check if predictive impact is stable.
    """
    def __init__(self, target_feature: str, validator: Optional[ModelValidator] = None):
        super().__init__("ValidateWithAlternativeClassifier", target_feature)
        self.validator = validator or LogisticRegressionValidator()

    def evaluate(self) -> Dict[str, float]:
        return {
            "evidence_gain": 6.0,
            "confidence_improvement": 8.0,
            "explainability_benefit": 5.0,
            "cost": 3.0  # Low computational cost
        }

    def execute(self, df: pd.DataFrame, target_col: str, current_fee_results: Dict[str, Any]) -> Dict[str, Any]:
        impact = self.validator.validate_feature_impact(df, target_col, self.target_feature)
        logger.info(f"AFL Action: Verified predictive impact for '{self.target_feature}' using alternative model: {impact:.4f}")
        return {
            "alternative_model_impact": impact,
            "alternative_model_verified": True
        }

class SubpopulationDisparityAnalysis(AFLAction):
    """
    Checks outcomes for subset partitions to see if disparities are amplified.
    """
    def __init__(self, target_feature: str):
        super().__init__("SubpopulationDisparityAnalysis", target_feature)

    def evaluate(self) -> Dict[str, float]:
        return {
            "evidence_gain": 8.0,
            "confidence_improvement": 5.0,
            "explainability_benefit": 9.0,
            "cost": 1.0  # Very low cost
        }

    def execute(self, df: pd.DataFrame, target_col: str, current_fee_results: Dict[str, Any]) -> Dict[str, Any]:
        # Compute selection rate variance across subgroups
        try:
            clean_df = df[[self.target_feature, target_col]].dropna()
            rates = clean_df.groupby(self.target_feature)[target_col].apply(
                lambda x: (x == x.value_counts().index[0]).mean() if len(x) > 0 else 0.0
            )
            disparity_variance = float(np.var(rates))
            logger.info(f"AFL Action: Subpopulation selection rate variance for '{self.target_feature}' is {disparity_variance:.4f}")
            return {
                "subpopulation_disparity_variance": disparity_variance,
                "subpopulation_analysis_run": True
            }
        except Exception as e:
            logger.debug(f"SubpopulationDisparityAnalysis failed: {e}")
            return {"subpopulation_disparity_variance": 0.0, "subpopulation_analysis_run": False}

class DeepCausalDiscovery(AFLAction):
    """
    Lightweight dependency analysis verifying conditional independence.
    Checks if Target is conditionally independent of Feature given another variable.
    """
    def __init__(self, target_feature: str, control_feature: str):
        super().__init__("DeepCausalDiscovery", target_feature)
        self.control_feature = control_feature

    def evaluate(self) -> Dict[str, float]:
        return {
            "evidence_gain": 9.0,
            "confidence_improvement": 9.0,
            "explainability_benefit": 8.0,
            "cost": 6.0  # Medium cost
        }

    def execute(self, df: pd.DataFrame, target_col: str, current_fee_results: Dict[str, Any]) -> Dict[str, Any]:
        # Lightweight Conditional Dependency Analysis: Chi-Square Independence
        try:
            clean_df = df[[self.target_feature, self.control_feature, target_col]].dropna()
            # Perform Chi2 test across subsets of control feature
            p_values = []
            for val in clean_df[self.control_feature].unique():
                subset = clean_df[clean_df[self.control_feature] == val]
                contingency = pd.crosstab(subset[self.target_feature], subset[target_col])
                if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                    _, p, _, _ = chi2_contingency(contingency)
                    p_values.append(p)
            
            mean_p = float(np.mean(p_values)) if p_values else 1.0
            is_causal = mean_p < 0.05
            logger.info(f"AFL Action: Conditional independence test for '{self.target_feature}' given '{self.control_feature}' p-value = {mean_p:.4f}")
            return {
                "conditional_p_value": mean_p,
                "is_conditionally_linked": is_causal,
                "causal_control": self.control_feature
            }
        except Exception as e:
            logger.debug(f"DeepCausalDiscovery failed: {e}")
            return {"conditional_p_value": 1.0, "is_conditionally_linked": False}

class AgenticFeedbackLoop:
    """
    Agentic Feedback Loop (AFL).
    Coordinates recursive investigation using a Pareto-inspired multi-objective decision strategy.
    """
    def __init__(self, max_loops: int = 2):
        self.max_loops = max_loops
        self.history: Set[str] = set()

    def filter_pareto_dominated(self, actions: List[AFLAction]) -> List[AFLAction]:
        """
        Removes dominated actions. An action is dominated if there exists another action
        that is better in at least one objective and not worse in any objective.
        Note: Cost is treated as a penalty (minimized), other objectives are maximized.
        """
        non_dominated = []
        
        for a in actions:
            a_eval = a.evaluate()
            dominated = False
            for b in actions:
                if a == b:
                    continue
                b_eval = b.evaluate()
                
                # Check if B dominates A
                b_is_better_or_equal = (
                    b_eval["evidence_gain"] >= a_eval["evidence_gain"] and
                    b_eval["confidence_improvement"] >= a_eval["confidence_improvement"] and
                    b_eval["explainability_benefit"] >= a_eval["explainability_benefit"] and
                    b_eval["cost"] <= a_eval["cost"]
                )
                b_is_strictly_better = (
                    b_eval["evidence_gain"] > a_eval["evidence_gain"] or
                    b_eval["confidence_improvement"] > a_eval["confidence_improvement"] or
                    b_eval["explainability_benefit"] > a_eval["explainability_benefit"] or
                    b_eval["cost"] < a_eval["cost"]
                )
                
                if b_is_better_or_equal and b_is_strictly_better:
                    dominated = True
                    break
            
            if not dominated:
                non_dominated.append(a)
                
        return non_dominated

    def select_best_action(self, actions: List[AFLAction]) -> AFLAction:
        """
        Applies a tie-breaking weighted sum to choose the best action from the Pareto frontier.
        Weights: gain=0.3, confidence=0.3, explainability=0.2, cost_benefit (10-cost)=0.2.
        """
        best_action = actions[0]
        best_score = -1.0
        
        for action in actions:
            evals = action.evaluate()
            cost_benefit = 10.0 - evals["cost"]
            score = (
                0.3 * evals["evidence_gain"] +
                0.3 * evals["confidence_improvement"] +
                0.2 * evals["explainability_benefit"] +
                0.2 * cost_benefit
            )
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action

    def run_feedback_loop(
        self,
        df: pd.DataFrame,
        target_col: str,
        cre_engine: Any,
        fee_results: Dict[str, Dict[str, Any]],
        ref_results: Dict[str, Dict[str, Any]],
        trace_logger: List[str]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Executes the recursive feedback loop.
        """
        logger.info("Agentic Feedback Loop starting...")
        trace_logger.append("=== Agentic Feedback Loop (AFL) Pipeline ===")
        
        loop_counter = 0
        afl_executed_actions = []
        current_fee = fee_results.copy()
        
        while loop_counter < self.max_loops:
            logger.info(f"AFL Iteration {loop_counter + 1}/{self.max_loops}")
            # Step 1: Run CRE on current evidence base
            cre_results = cre_engine.analyze_dataset(df, current_fee, ref_results)
            
            # Step 2: Identify candidate features requiring further evidence
            suspect_features = []
            for r in cre_results:
                # Features with HIGH/MEDIUM risk but low confidence (< 75%) or flagged as uncertain
                if r.priority in ("HIGH", "MEDIUM") and (r.confidence < 75.0 or r.is_uncertain):
                    suspect_features.append(r)
                # Or standard features that show proxy correlation warnings
                elif r.priority == "MEDIUM" and r.is_proxy:
                    suspect_features.append(r)
                    
            if not suspect_features:
                logger.info("AFL: No features require further investigation. High confidence achieved.")
                trace_logger.append(f"Iteration {loop_counter+1}: Sufficient confidence achieved. Loop terminated.")
                break
                
            # Step 3: Generate candidate actions
            candidates: List[AFLAction] = []
            for sf in suspect_features:
                f_name = sf.column_name
                
                # Check history to avoid repeat actions
                act_key_clf = f"{f_name}_classifier"
                act_key_sub = f"{f_name}_subpopulation"
                
                if act_key_clf not in self.history:
                    candidates.append(ValidateWithAlternativeClassifier(f_name))
                if act_key_sub not in self.history:
                    candidates.append(SubpopulationDisparityAnalysis(f_name))
                    
                # If it correlates with another variable, suggest deep causal discovery given that control
                for r2 in cre_results:
                    if r2.column_name != f_name and r2.priority != "IGNORE":
                        act_key_cau = f"{f_name}_causal_{r2.column_name}"
                        if act_key_cau not in self.history:
                            candidates.append(DeepCausalDiscovery(f_name, r2.column_name))

            if not candidates:
                logger.info("AFL: Candidate actions exhausted.")
                trace_logger.append(f"Iteration {loop_counter+1}: No new actions available. Loop terminated.")
                break

            trace_logger.append(f"Iteration {loop_counter+1}: Identified {len(candidates)} candidate investigation paths.")

            # Step 4: Pareto Filtering
            pareto_set = self.filter_pareto_dominated(candidates)
            trace_logger.append(f"Iteration {loop_counter+1}: Filtered to {len(pareto_set)} Pareto-efficient actions.")

            # Step 5: Select best action
            chosen_action = self.select_best_action(pareto_set)
            trace_logger.append(f"Iteration {loop_counter+1}: Selected action '{chosen_action.action_name}' for feature '{chosen_action.target_feature}'.")

            # Step 6: Execute chosen action
            new_evidence = chosen_action.execute(df, target_col, current_fee.get(chosen_action.target_feature, {}))
            
            # Track history
            act_history_key = f"{chosen_action.target_feature}_{chosen_action.action_name}"
            if isinstance(chosen_action, DeepCausalDiscovery):
                act_history_key = f"{chosen_action.target_feature}_causal_{chosen_action.control_feature}"
            self.history.add(act_history_key)
            
            # Step 7: Merge new evidence
            if chosen_action.target_feature not in current_fee:
                current_fee[chosen_action.target_feature] = {}
            current_fee[chosen_action.target_feature].update(new_evidence)
            
            # Log executed action details
            afl_executed_actions.append({
                "iteration": loop_counter + 1,
                "feature": chosen_action.target_feature,
                "action": chosen_action.action_name,
                "objectives": chosen_action.evaluate(),
                "result": new_evidence
            })

            loop_counter += 1

        if loop_counter >= self.max_loops:
            logger.warning(f"AFL: Reached recursion limit of {self.max_loops}.")
            trace_logger.append(f"AFL warning: Loop hit maximum recursion limit ({self.max_loops}).")

        return current_fee, afl_executed_actions
