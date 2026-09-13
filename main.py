"""
main.py — CAFRE End-to-End Pipeline Orchestrator
==================================================
Context-Aware Fairness Reasoning Engine (CAFRE)

Executes the complete CAFRE pipeline from dataset input to final
explainable fairness report.

Pipeline Order:
  1. Dataset Input
  2. Semantic Recognition Engine (SRE)
  3. Context Detection Engine (CDE)
  4. Context Knowledge Engine (CKE)
  5. Dynamic Investigation Planner (DIP)
  6. Fairness Evidence Extractor (FEE)
  7. Registered Fairness Evidences (REF)
  8. Context-Aware Reasoning Engine (CRE)
  9. Agentic Feedback Loop (AFL) — Pareto-Inspired
 10. FERC Explainability Engine
 11. Bias Mitigation Engine (Beta Advisory)
 12. Final Explainable Fairness Report

Usage:
    python main.py <path_to_dataset.csv>
"""

import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import logging
import warnings
import textwrap

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("cafre.main")

# ---------------------------------------------------------------------------
# Add project root to path so all modules resolve correctly
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

# BLOCK-2: Semantic Recognition Engine
SRE_PATH = os.path.join(PROJECT_ROOT, "CAFRE", "BLOCK-2_SEMANTIC_RECOGNITION_ENGINE")
sys.path.insert(0, SRE_PATH)
from CAFRE.BLOCK_2_SEMANTIC_RECOGNITION_ENGINE.semantic_engine import SemanticEngine
from CAFRE.BLOCK_2_SEMANTIC_RECOGNITION_ENGINE.utils import load_dataset

# BLOCK-3: Context Detection Engine
from CAFRE.BLOCK_3_CONTEXT_DETECTION_ENGINE.detector import ContextDetector

# BLOCK-4: Context Knowledge Engine
from CAFRE.BLOCK_4_CONTEXT_KNOWLEDGE_ENGINE.leco_generator import CKEGenerator

# BLOCK-5: Dynamic Investigation Planner
from CAFRE.BLOCK_5_DYNAMIC_INVESTIGATION_PLANNER.planner import InvestigationPlanner

# BLOCK-6: Fairness Evidence Extractor
from CAFRE.BLOCK_6_FAIRNESS_EVIDENCE_EXTRACTOR.mpem import FairnessEvidenceExtractor
from CAFRE.BLOCK_6_FAIRNESS_EVIDENCE_EXTRACTOR.ref import RegisteredFairnessEvidences

# BLOCK-7: Context-Aware Reasoning Engine
from CAFRE.BLOCK_7_CONTEXT_AWARE_REASONING_ENGINE.cre import ContextAwareReasoningEngine

# BLOCK-8: Agentic Feedback Loop
from CAFRE.BLOCK_8_AGENTIC_FEEDBACK_LOOP.afl import AgenticFeedbackLoop

# BLOCK-9: FERC Explainability Engine
from CAFRE.BLOCK_9_FERC_EXPLAINABILITY_ENGINE.ferc import FERCExplainabilityEngine

# BLOCK-10: Bias Mitigation Engine
from CAFRE.BLOCK_10_BIAS_MITIGATION_ENGINE.engine import BiasMitigationEngine


def print_section(title: str, width: int = 70):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step(step_num: int, total_steps: int, label: str):
    print(f"\n[Step {step_num}/{total_steps}] {label}")
    print("-" * 60)


def run_pipeline(dataset_path: str, output_dir: str = "output"):
    print_section("CAFRE — Context-Aware Fairness Reasoning Engine", width=70)
    print("  Domain-Aware AI Bias Detection & Explainability Framework")
    print("=" * 70)

    total_steps = 11
    pipeline_start = time.time()

    # -----------------------------------------------------------------------
    # Step 1 — Dataset Input
    # -----------------------------------------------------------------------
    print_step(1, total_steps, "Dataset Input")
    df = load_dataset(dataset_path)
    print(f"  Dataset loaded: {dataset_path}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {', '.join(df.columns.tolist())}")

    # -----------------------------------------------------------------------
    # Step 2 — Semantic Recognition Engine (SRE)
    # -----------------------------------------------------------------------
    print_step(2, total_steps, "Semantic Recognition Engine (SRE)")
    sre = SemanticEngine(kb_path=os.path.join(PROJECT_ROOT, "CAFRE", "BLOCK_2_SEMANTIC_RECOGNITION_ENGINE", "knowledge.json"))
    semantics = sre.understand_dataset(df)
    target_col = sre.identify_primary_outcome(semantics)

    if target_col is None:
        # Fallback: pick the last binary/categorical column
        for col in reversed(df.columns.tolist()):
            if df[col].nunique() <= 5:
                target_col = col
                break

    print(f"  Detected Target Variable: '{target_col}'")
    print(f"  Column semantic roles identified:")
    for s in semantics:
        marker = "[SENSITIVE]" if (s.concept_match and s.concept_match.is_sensitive) else ""
        print(f"    • {s.column_name:<25} Role: {s.inferred_role:<12} {marker}")

    # -----------------------------------------------------------------------
    # Step 3 — Context Detection Engine (CDE)
    # -----------------------------------------------------------------------
    print_step(3, total_steps, "Context Detection Engine (CDE)")
    cde = ContextDetector()
    dataset_metadata = {"columns": df.columns.tolist()}
    detected_domain, domain_confidence = cde.detect_context(dataset_metadata)
    print(f"  Detected Domain: {detected_domain}  (Confidence: {domain_confidence:.2%})")

    # -----------------------------------------------------------------------
    # Step 4 — Context Knowledge Engine (CKE)
    # -----------------------------------------------------------------------
    print_step(4, total_steps, "Context Knowledge Engine (CKE)")
    cke = CKEGenerator()
    cke_ontology = cke.get_ontology(detected_domain)
    if not cke_ontology:
        logger.warning("CKE returned empty ontology (domain unknown). Proceeding with empty rules.")
        cke_ontology = {"protected_attributes": [], "historical_biases": [], "known_proxies": {}}

    print(f"  Protected Attributes: {', '.join(cke_ontology.get('protected_attributes', [])) or 'None detected'}")
    print(f"  Historical Bias Patterns: {len(cke_ontology.get('historical_biases', []))}")
    print(f"  Known Proxy Mappings: {len(cke_ontology.get('known_proxies', {}))}")

    # -----------------------------------------------------------------------
    # Step 5 — Dynamic Investigation Planner (DIP)
    # -----------------------------------------------------------------------
    print_step(5, total_steps, "Dynamic Investigation Planner (DIP)")
    dip = InvestigationPlanner()
    investigation_plan = dip.generate_plan(
        context=detected_domain,
        dataset_metadata=dataset_metadata,
        leco_ontology=cke_ontology,
        target_variable=target_col
    )
    total_features = len(investigation_plan.feature_plans)
    shap_count = sum(1 for p in investigation_plan.feature_plans if p.run_shap)
    causal_count = sum(1 for p in investigation_plan.feature_plans if p.run_causal_discovery)
    print(f"  Investigation planned for {total_features} features.")
    print(f"  Features with predictive impact analysis: {shap_count}")
    print(f"  Features with causal discovery: {causal_count}")

    # -----------------------------------------------------------------------
    # Step 6 — Fairness Evidence Extractor (FEE)
    # -----------------------------------------------------------------------
    print_step(6, total_steps, "Fairness Evidence Extractor (FEE)")
    fee = FairnessEvidenceExtractor()
    fee_results = fee.execute_plan(df, investigation_plan)
    print(f"  Evidence extracted for {len(fee_results)} features.")
    # Surface top findings
    for fname, ev in fee_results.items():
        pi = ev.get("predictive_impact", 0.0)
        mi = ev.get("mutual_information", 0.0)
        if pi > 0.01 or mi > 0.05:
            print(f"    • {fname:<25} Predictive Impact: {pi:.4f}  MI: {mi:.4f}")

    # -----------------------------------------------------------------------
    # Step 7 — Registered Fairness Evidences (REF)
    # -----------------------------------------------------------------------
    print_step(7, total_steps, "Registered Fairness Evidences (REF)")
    ref_engine = RegisteredFairnessEvidences()
    ref_results = {}
    protected_in_data = [
        col for col in cke_ontology.get("protected_attributes", [])
        if col in df.columns
    ]
    # Also check flat proxies
    for proxies in cke_ontology.get("known_proxies", {}).values():
        for proxy in proxies:
            if proxy in df.columns and proxy not in protected_in_data:
                protected_in_data.append(proxy)

    if not protected_in_data:
        # Fallback: compute REF for all non-target low-cardinality columns
        protected_in_data = [
            c for c in df.columns if c != target_col and df[c].nunique() <= 10
        ]

    for col in protected_in_data:
        metrics = ref_engine.compute_metrics(df, protected_col=col, target_col=target_col)
        if metrics:
            ref_results[col] = metrics
            di = metrics.get("disparate_impact", 1.0)
            dpd = metrics.get("demographic_parity_difference", 0.0)
            print(f"  • {col:<25} DI: {di:.3f}  DPD: {dpd:.3f}  {'[!] Biased' if di < 0.8 else 'OK'}")

    # -----------------------------------------------------------------------
    # Step 8 — Context-Aware Reasoning Engine (CRE)
    # -----------------------------------------------------------------------
    print_step(8, total_steps, "Context-Aware Reasoning Engine (CRE)")
    cre = ContextAwareReasoningEngine(
        domain=detected_domain,
        cke_ontology=cke_ontology,
        target_col=target_col
    )
    cre_results = cre.analyze_dataset(df, fee_results, ref_results)
    high = [r for r in cre_results if r.priority == "HIGH"]
    med  = [r for r in cre_results if r.priority == "MEDIUM"]
    low  = [r for r in cre_results if r.priority == "LOW"]
    print(f"  Reasoning complete. Results → HIGH: {len(high)}  MEDIUM: {len(med)}  LOW: {len(low)}")
    for r in cre_results:
        if r.priority in ("HIGH", "MEDIUM"):
            uncertain_tag = " [UNCERTAIN]" if r.is_uncertain else ""
            print(f"    • {r.column_name:<25} Score: {r.final_score:5.1f}  Priority: {r.priority}{uncertain_tag}")

    # -----------------------------------------------------------------------
    # Step 9 — Agentic Feedback Loop (AFL) — Pareto-Inspired
    # -----------------------------------------------------------------------
    print_step(9, total_steps, "Agentic Feedback Loop (AFL) — Pareto-Inspired")
    afl = AgenticFeedbackLoop(max_loops=2)
    afl_trace = []
    refined_fee_results, afl_actions = afl.run_feedback_loop(
        df=df,
        target_col=target_col,
        cre_engine=cre,
        fee_results=fee_results,
        ref_results=ref_results,
        trace_logger=afl_trace
    )
    print(f"  AFL completed {len(afl_actions)} Pareto-optimised investigation(s).")
    for line in afl_trace:
        print(f"    {line}")

    # Re-run CRE with refined evidence from AFL
    cre_results = cre.analyze_dataset(df, refined_fee_results, ref_results)

    # -----------------------------------------------------------------------
    # Step 10 — FERC Explainability Engine
    # -----------------------------------------------------------------------
    print_step(10, total_steps, "FERC Explainability Engine")
    ferc = FERCExplainabilityEngine(
        domain=detected_domain,
        target_col=target_col,
        cke_ontology=cke_ontology,
        cre_results=cre_results,
        afl_actions=afl_actions,
        fee_results=refined_fee_results,
        ref_results=ref_results
    )
    json_path, md_path = ferc.write_reports(output_dir=output_dir)
    print(f"  JSON Report: {json_path}")
    print(f"  Markdown Report: {md_path}")

    # -----------------------------------------------------------------------
    # Step 11 — Bias Mitigation Engine (Beta Advisory)
    # -----------------------------------------------------------------------
    print_step(11, total_steps, "Bias Mitigation Engine (Beta Advisory)")
    mitigation_engine = BiasMitigationEngine(domain=detected_domain)
    recommendations = mitigation_engine.generate_recommendations(cre_results)
    total_strategies = sum(len(r["recs"]) for r in recommendations)
    print(f"  Generated {total_strategies} mitigation strategies across {len(recommendations)} feature(s).")

    # -----------------------------------------------------------------------
    # Final Report — Terminal Summary
    # -----------------------------------------------------------------------
    print_section("CAFRE — Final Explainable Fairness Report", width=70)
    print(f"  Domain Context : {detected_domain} ({domain_confidence:.0%} confidence)")
    print(f"  Target Variable: {target_col}")
    print(f"  Dataset Size   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print()

    print("  BIAS RISK RANKING")
    print("  " + "-" * 66)
    print(f"  {'Feature':<25} {'Score':>6}  {'Priority':<8}  {'Confidence':>10}  Category")
    print("  " + "-" * 66)
    for r in cre_results:
        if r.priority == "IGNORE":
            continue
        unc = " [!]" if r.is_uncertain else ""
        print(
            f"  {r.column_name:<25} {r.final_score:>6.1f}  {r.priority:<8}  "
            f"{r.confidence:>9.1f}%  {r.semantic_category}{unc}"
        )
    print("  " + "-" * 66)

    if ref_results:
        print()
        print("  FAIRNESS METRICS (Registered Fairness Evidences - REF)")
        print("  " + "-" * 66)
        print(f"  {'Feature':<25} {'DI':>6}  {'DPD':>6}  {'EOD':>6}  {'EODD':>6}  Status")
        print("  " + "-" * 66)
        for col, m in ref_results.items():
            di   = m.get("disparate_impact", 1.0)
            dpd  = m.get("demographic_parity_difference", 0.0)
            eod  = m.get("equal_opportunity_difference", 0.0)
            eodd = m.get("equalized_odds_difference", 0.0)
            status = "[!] BIASED" if di < 0.8 or dpd > 0.2 else "[OK] FAIR"
            print(f"  {col:<25} {di:>6.3f}  {dpd:>6.3f}  {eod:>6.3f}  {eodd:>6.3f}  {status}")
        print("  " + "-" * 66)

    if recommendations:
        print()
        print("  BIAS MITIGATION RECOMMENDATIONS (Beta Advisory)")
        print("  " + "-" * 66)
        for rec in recommendations:
            print(f"\n  Feature: {rec['feature']}  (Priority: {rec['priority_level']})")
            for s in rec["recs"]:
                print(f"    [{s['type']}] {s['strategy_name']}  - Priority: {s['priority']}")
                wrapped_rationale = textwrap.fill(s['suitability_rationale'], width=90, initial_indent="    Rationale: ", subsequent_indent="               ")
                print(wrapped_rationale)

    if afl_actions:
        print()
        print("  AGENTIC FEEDBACK LOOP - Pareto-Optimised Investigations")
        print("  " + "-" * 66)
        for act in afl_actions:
            ev = act["objectives"]
            print(
                f"  Iter {act['iteration']} | Feature: {act['feature']:<20} "
                f"Action: {act['action']}"
            )
            print(
                f"           Pareto Objectives -> Gain:{ev['evidence_gain']}  "
                f"Conf:{ev['confidence_improvement']}  "
                f"Expl:{ev['explainability_benefit']}  "
                f"Cost:{ev['cost']}"
            )

    elapsed = time.time() - pipeline_start
    print()
    print("  " + "─" * 66)
    print(f"  Reports saved to: {os.path.abspath(output_dir)}/")
    print(f"  Pipeline executed in {elapsed:.2f}s")
    print_section("CAFRE Pipeline Completed Successfully", width=70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python main.py <path_to_dataset.csv>")
        print("Example: python main.py data/biased_dataset_train.csv")
        sys.exit(1)
    run_pipeline(dataset_path=sys.argv[1])
