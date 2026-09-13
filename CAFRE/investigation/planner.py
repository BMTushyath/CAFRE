import logging
from typing import Dict, List, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class FeatureInvestigationPlan(BaseModel):
    feature_name: str = Field(..., description="The name of the feature in the dataset.")
    run_shap: bool = Field(default=False, description="Whether to compute predictive impact using SHAP.")
    run_mutual_information: bool = Field(default=False, description="Whether to compute dependency using Mutual Information.")
    run_causal_discovery: bool = Field(default=False, description="Whether to run causal graph discovery.")
    run_distribution_analysis: bool = Field(default=True, description="Always run basic integrity checks.")

class InvestigationExecutionPlan(BaseModel):
    domain: str = Field(..., description="The detected domain context.")
    target_variable: str = Field(..., description="The target variable for the model.")
    feature_plans: List[FeatureInvestigationPlan] = Field(..., description="Targeted statistical plans per feature.")

class InvestigationPlanner:
    """
    Dynamic Investigation Planner.
    Formulates a targeted statistical extraction plan based on context and LECO rules.
    Prevents p-hacking and limits computational waste by selectively applying heavy tests 
    like causal discovery only to high-risk features.
    """
    def __init__(self):
        pass

    def generate_plan(self, context: str, dataset_metadata: Dict[str, Any], leco_ontology: Dict[str, Any], target_variable: str) -> InvestigationExecutionPlan:
        """
        Generates a targeted statistical execution plan.
        
        Args:
            context: Detected domain string.
            dataset_metadata: Dictionary containing 'columns' list.
            leco_ontology: The validated LECO JSON dictionary.
            target_variable: The name of the target column.
            
        Returns:
            InvestigationExecutionPlan: A Pydantic model detailing what to test.
        """
        logger.info(f"Generating investigation plan for domain: {context}")
        
        columns = dataset_metadata.get("columns", [])
        protected_attributes = leco_ontology.get("protected_attributes", [])
        known_proxies_map = leco_ontology.get("known_proxies", {})
        
        # Flatten known proxies for easy lookup
        all_proxies = []
        for proxies in known_proxies_map.values():
            all_proxies.extend(proxies)
            
        feature_plans = []
        
        for col in columns:
            if col == target_variable:
                continue
                
            plan = FeatureInvestigationPlan(feature_name=col)
            
            # If the feature is a known protected attribute
            if col in protected_attributes:
                plan.run_mutual_information = True
                plan.run_causal_discovery = True
                plan.run_shap = True
                
            # If the feature is a known proxy for a protected attribute
            elif col in all_proxies:
                plan.run_mutual_information = True
                plan.run_causal_discovery = True
                plan.run_shap = True
                
            # Standard features
            else:
                plan.run_shap = True
                plan.run_mutual_information = True
                # Skip heavy causal discovery for non-risk features to save compute
                plan.run_causal_discovery = False
                
            feature_plans.append(plan)
            
        execution_plan = InvestigationExecutionPlan(
            domain=context,
            target_variable=target_variable,
            feature_plans=feature_plans
        )
        
        logger.info(f"Generated plan with {len(feature_plans)} feature investigations.")
        return execution_plan
