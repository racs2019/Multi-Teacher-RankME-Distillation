*GRACE: Reliability-Gated Graph Adaptation under Domain Shift*

GRACE is a framework for unsupervised model combination under domain shift,
which selectively applies graph-based refinement only when it is reliable.

---

OVERVIEW

Pretrained models are widely reused under domain shift, but selecting or combining
them without labels is challenging.

We study this setting and find:

- TENT-style test-time adaptation often fails
- Graph-based refinement is very strong
- But graph methods can be brittle and propagate errors

---

KEY INSIGHT

Adaptation is not always safe.
The effectiveness of refinement depends on the local predictive geometry
of the target domain.

---

APPROACH

GRACE combines a strong anchor prediction with graph refinement:

p_final = (1 - g) * p_anchor + g * p_graph

p_anchor: uniform ensemble
p_graph: graph refinement
g: reliability gate

GRACE applies refinement only when supported by the data.

---

MAIN FINDINGS

Across DomainNet and TerraIncognita:

- Uniform ensemble is a strong baseline
- Graph refinement improves performance in moderate shift
- Graph methods degrade under structured noise
- GRACE preserves gains while preventing degradation

---

STRESS TEST

We corrupt pseudo-label seeds:

Graph: degrades
GRACE: remains stable

This shows reliability gating is necessary.

---

REPO STRUCTURE

scripts/
  DomainNet/
  TerraIncognita/
scripts/figures/
features/
results/
final_results/
final_results_summary/
stress_results/
figures/

---

PIPELINE

1. Prepare manifest
2. Extract teachers
3. Train probes
4. Run methods
5. Aggregate results
6. Run stress test
7. Generate figures

---

DATASETS

- DomainNet
- TerraIncognita

---

AUTHOR

Richard Acs
Florida Atlantic University
