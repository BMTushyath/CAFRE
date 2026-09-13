import json
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from CAFRE.BLOCK_7_CONTEXT_AWARE_REASONING_ENGINE.cre import CREResult

logger = logging.getLogger(__name__)

class FERCExplainabilityEngine:
    """
    FERC Explainability Engine.
    Generates human-readable explanations, trace justifications, and confidence explanations
    for all bias decisions in CAFRE. Outputs reports as JSON and Markdown.
    """
    def __init__(
        self,
        domain: str,
        target_col: str,
        cke_ontology: Dict[str, Any],
        cre_results: List[CREResult],
        afl_actions: List[Dict[str, Any]],
        fee_results: Dict[str, Dict[str, Any]],
        ref_results: Dict[str, Dict[str, Any]]
    ):
        self.domain = domain
        self.target_col = target_col
        self.ontology = cke_ontology
        self.cre_results = cre_results
        self.afl_actions = afl_actions
        self.fee_results = fee_results
        self.ref_results = ref_results

    def compile_json_report(self) -> Dict[str, Any]:
        """
        Compiles the entire CAFRE pipeline trace and results into a single dictionary.
        """
        report_data = {
            "metadata": {
                "framework": "CAFRE (Context-Aware Fairness Reasoning Engine)",
                "domain": self.domain,
                "target_variable": self.target_col
            },
            "cke_ontology": {
                "protected_attributes": self.ontology.get("protected_attributes", []),
                "historical_biases": self.ontology.get("historical_biases", []),
                "known_proxies": self.ontology.get("known_proxies", {})
            },
            "empirical_evidence": {
                "fee_raw_stats": self.fee_results,
                "ref_fairness_metrics": self.ref_results
            },
            "agentic_feedback_loop": {
                "iterations_run": len(self.afl_actions),
                "actions_executed": self.afl_actions
            },
            "reasoning_results": []
        }

        for r in self.cre_results:
            report_data["reasoning_results"].append({
                "column_name": r.column_name,
                "category": r.semantic_category,
                "role": r.semantic_role,
                "is_protected": r.is_protected,
                "is_proxy": r.is_proxy,
                "proxy_for": r.proxy_for,
                "bias_scores": {
                    "base_semantic_score": r.base_score,
                    "empirical_stat_score": r.stat_score,
                    "final_fused_score": r.final_score
                },
                "confidence": r.confidence,
                "priority_level": r.priority,
                "uncertainty": {
                    "is_uncertain": r.is_uncertain,
                    "reasons": r.uncertainty_reasons
                },
                "explanation_trace": r.trace
            })

        return report_data

    def generate_markdown_report(self, json_report: Dict[str, Any]) -> str:
        """
        Formats the JSON report dictionary into a clean, markdown audit report.
        """
        md = []
        md.append("# CAFRE Explainable Fairness Audit Report")
        md.append(f"**Domain Context:** {json_report['metadata']['domain']}")
        md.append(f"**Target Outcome Variable:** `{json_report['metadata']['target_variable']}`")
        md.append("")
        
        md.append("## 1. Context Knowledge Ontology (CKE)")
        md.append("The Context Knowledge Engine loaded the following legal and historical boundaries:")
        md.append(f"- **Protected Attributes:** {', '.join(json_report['cke_ontology']['protected_attributes'])}")
        md.append("- **Historical Domain Biases Identified:**")
        for hb in json_report["cke_ontology"]["historical_biases"]:
            md.append(f"  * {hb}")
        md.append("- **Known Proxy Mappings:**")
        for k, v in json_report["cke_ontology"]["known_proxies"].items():
            md.append(f"  * `{k}` proxies: {', '.join(v)}")
        md.append("")

        md.append("## 2. Agentic Feedback Loop (AFL) Trace")
        md.append(f"The AFL ran **{json_report['agentic_feedback_loop']['iterations_run']}** Pareto-efficient iterations to verify uncertain or conflicting evidence:")
        if not json_report['agentic_feedback_loop']['actions_executed']:
            md.append("No additional feedback investigations were required. Baseline evidence was sufficient.")
        else:
            for idx, act in enumerate(json_report['agentic_feedback_loop']['actions_executed']):
                md.append(f"### Iteration {act['iteration']}: Feature `{act['feature']}`")
                md.append(f"- **Investigation Action:** `{act['action']}`")
                md.append("- **Pareto Objectives Evaluated:**")
                md.append(f"  * Expected Gain: {act['objectives']['evidence_gain']}/10")
                md.append(f"  * Confidence Boost: {act['objectives']['confidence_improvement']}/10")
                md.append(f"  * Explainability Value: {act['objectives']['explainability_benefit']}/10")
                md.append(f"  * Cost Penalty: {act['objectives']['cost']}/10")
                md.append("- **Acquired Findings:**")
                for k, v in act["result"].items():
                    md.append(f"  * `{k}`: {v}")
        md.append("")

        md.append("## 3. Cognitive Reasoning & Bias Analysis")
        md.append("The Context-Aware Reasoning Engine (CRE) evaluated and ranked the variables:")
        
        for r in json_report["reasoning_results"]:
            if r["priority_level"] == "IGNORE":
                continue
                
            md.append(f"### Column: `{r['column_name']}` (Priority: **{r['priority_level']}**)")
            md.append(f"- **Category:** {r['category']} | **Role:** {r['role']}")
            md.append(f"- **Bias Risk Score:** {r['bias_scores']['final_fused_score']:.1f}/100")
            md.append(f"- **Confidence Level:** {r['confidence']:.1f}%")
            if r["uncertainty"]["is_uncertain"]:
                md.append(f"- **Uncertainty Warning:** [!] {', '.join(r['uncertainty']['reasons'])}")
            md.append("- **Trace Justification:**")
            for t in r["explanation_trace"]:
                md.append(f"  * {t}")
            
            # Print specific metrics if they exist
            fee_stats = self.fee_results.get(r['column_name'], {})
            ref_stats = self.ref_results.get(r['column_name'], {})
            
            if fee_stats or ref_stats:
                md.append("- **Supporting Empirical Evidence:**")
                if "predictive_impact" in fee_stats:
                    md.append(f"  * Permutation Predictive Impact: {fee_stats['predictive_impact']:.4f}")
                if "mutual_information" in fee_stats:
                    md.append(f"  * Normalized Mutual Info: {fee_stats['mutual_information']:.4f}")
                if "causal_indicator" in fee_stats:
                    md.append(f"  * Chi-square Independent p-value: {fee_stats['causal_indicator'].get('p_value', 1.0):.4f}")
                if ref_stats:
                    md.append(f"  * Disparate Impact Ratio: {ref_stats.get('disparate_impact', 1.0):.3f}")
                    md.append(f"  * Demographic Parity Difference: {ref_stats.get('demographic_parity_difference', 0.0):.3f}")
                    md.append(f"  * Equal Opportunity Difference: {ref_stats.get('equal_opportunity_difference', 0.0):.3f}")
            md.append("")

        return "\n".join(md)

    def write_reports(self, output_dir: str = "output") -> Tuple[str, str]:
        """
        Compiles the report, writes JSON and Markdown to disk, and returns the paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        json_report = self.compile_json_report()
        
        # Write JSON
        json_path = os.path.join(output_dir, "cafre_fairness_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=4)
            
        # Generate and Write Markdown
        md_content = self.generate_markdown_report(json_report)
        md_path = os.path.join(output_dir, "cafre_fairness_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        logger.info(f"FERC Explainability Engine saved reports to {json_path} and {md_path}")
        return json_path, md_path
