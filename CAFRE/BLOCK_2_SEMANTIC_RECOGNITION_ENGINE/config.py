"""
config.py — Configuration for the Semantic Reasoning Engine (SRE)
===================================================================

Stores configuration for semantic similarity thresholds, rules, and
statistical fallbacks.
"""

# Similarity threshold for fuzzy matching column names to KB concepts (0-100).
SEMANTIC_MATCH_THRESHOLD = 75

# Weights for Statistical Evidence (only used if column is NOT ignored by reasoning).
STAT_WEIGHTS = {
    "distribution":       0.20,
    "dependency":         0.30,
    "mutual_information": 0.20,
    "correlation":        0.15,
    "data_quality":       0.15,
}

# Priority Thresholds for the final Fused Score.
PRIORITY_HIGH = 75
PRIORITY_MEDIUM = 45

# Rules for Reasoning Engine priority assignment based on Semantic Roles.
ROLE_BASE_SCORES = {
    "Outcome":    0.0,   # Outcomes cannot be causes; skip stats.
    "Identifier": 0.0,   # IDs carry no meaning; skip stats.
    "Derived":    20.0,  # Derived columns are low priority.
    "Decision":   60.0,  # Intermediate decisions are highly interesting.
    "Feature":    30.0,  # Normal features have base priority; heavily modified by stats/sensitivity.
}

# Priority modifiers based on semantic traits.
MODIFIER_SENSITIVE = 30.0  # Boost for sensitive demographic traits.
