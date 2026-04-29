# GRACE: Reliability-Gated Graph Adaptation under Domain Shift

GRACE is a framework for **unsupervised model combination under domain shift** that selectively applies graph-based refinement only when it is reliable.

---

## 🧠 Overview

Pretrained models are widely reused under domain shift, but selecting or combining them **without labels** is challenging.

This project studies that setting and finds:

- ❌ TENT-style test-time adaptation often fails  
- ✅ Graph-based refinement is a strong baseline  
- ⚠️ Graph methods can be brittle and propagate errors  

---

## 🔥 Key Insight

> **Adaptation is not always safe.**

The effectiveness of refinement depends on the **local predictive geometry** of the target domain.  
When local structure is unreliable, adaptation can degrade performance instead of improving it.

---

## 🚀 Method

```
p_final = (1 - g) * p_anchor + g * p_graph
```

- **p_anchor**: uniform ensemble (robust baseline)  
- **p_graph**: graph-based refinement (e.g., LAME-style)  
- **g**: per-sample reliability gate  

GRACE applies refinement **only when supported by the data**.

---

## 📊 Main Findings

Across DomainNet and TerraIncognita:

- Uniform ensembling is a strong and stable baseline  
- Graph refinement helps under moderate shift  
- Graph methods degrade under structured noise  
- **GRACE preserves gains while preventing degradation**

---

## 🧪 Stress Test (Key Result)

We introduce controlled corruption into pseudo-label seeds:

- Graph refinement degrades as corruption increases  
- GRACE remains stable by suppressing unreliable updates  

👉 Reliability gating is necessary for safe adaptation.

---

## 📁 Repository Structure

```
scripts/
  DomainNet/        # Core pipeline
  TerraIncognita/   # Terra run scripts
  figures/          # Plotting scripts

features/           # Extracted features
results/            # Probe outputs
final_results/      # Method outputs
final_results_summary/
stress_results/
figures/            # Generated plots
```

---

## ⚙️ Pipeline

1. Prepare dataset manifest  
2. Extract teacher features  
3. Train linear probes  
4. Run methods (uniform, graph, GRACE)  
5. Aggregate results  
6. Run stress tests  
7. Generate figures  

---

## 📚 Datasets

- DomainNet  
- TerraIncognita  

---

## 🧠 Key Takeaway

> The main challenge is not only how to adapt, but **when adaptation should be trusted**.

GRACE provides a simple and effective mechanism to enforce this principle.

---

## 👤 Author

Richard Acs
Ph.D. Student Department of Computer Science
Florida Atlantic University
