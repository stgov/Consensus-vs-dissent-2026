# Consensus-vs-dissent-2026
Repository for the paper: Consensus vs. Dissent: Dynamic LLM Modeling of Subjective Preferences in Group Recommenders



## Content:

strats: functions to calculate social choice-based aggregation strategies 

config_gen: functions to generate group configurations (uniform, divergent, coalitional, minority)

basemodels: generate LLM-as-judge assessments using base models (using either Cerebras or Ollama) -> but easily adjustable for use with e.g. Huggingface
