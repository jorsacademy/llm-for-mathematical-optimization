# Literature map: LLM-based automatic heuristic design

This note separates the research lineage that motivates this repository from the claims made by the implementation.

The project is an independent, compact research benchmark. It does not copy published repositories and it does not claim to reproduce paper-level results.

## 1. Program search and automatic heuristic design foundations

### FunSearch — Nature 2024

Romera-Paredes et al., *Mathematical discoveries from program search with large language models*, Nature 625, 468–475 (2024).

- Program search combines an LLM proposal mechanism with an external evaluator.
- The system evolves the critical part of a program rather than asking the LLM to directly output final solutions.
- Online bin packing is one of the demonstrated algorithmic applications.
- The published system uses a substantially larger, distributed search architecture than this repository.

Primary source:
https://doi.org/10.1038/s41586-023-06924-6

Official code:
https://github.com/google-deepmind/funsearch

### Evolution of Heuristics (EoH) — ICML 2024

Liu et al., *Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model*, ICML 2024, PMLR 235:32201–32223.

- EoH explicitly frames the task as Automatic Heuristic Design (AHD).
- It evolves both natural-language heuristic ideas ("thoughts") and executable implementations.
- The paper emphasizes LLM-query efficiency as a computational-budget dimension.
- The approach differs from this repository because the present implementation searches a restricted expression DSL rather than jointly evolving free-form thoughts and Python functions.

Primary source:
https://proceedings.mlr.press/v235/liu24bs.html

## 2. 2024–2025 algorithm-evolution developments

### LLaMEA — IEEE Transactions on Evolutionary Computation

van Stein and Bäck, *LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics*, IEEE Transactions on Evolutionary Computation 29(2), 331–345 (2025), DOI 10.1109/TEVC.2024.3497793.

- LLaMEA treats runtime evaluation as feedback for iterative generation, mutation, and selection of algorithms.
- It highlights benchmark cost and function-evaluation budgets as first-class experimental quantities.
- This repository borrows the methodological principle of evaluator-grounded selection, not the implementation.

DOI:
https://doi.org/10.1109/TEVC.2024.3497793

### MCTS-AHD — ICML 2025

Zheng et al., *Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design*, ICML 2025, PMLR 267:78338–78373.

- MCTS-AHD replaces a fixed-size population-only view with a tree-search organization of generated heuristics.
- The motivation is to reduce premature loss of temporarily weak but promising branches.
- This repository intentionally remains population-based; MCTS is a possible future extension rather than an unimplemented claim.

Primary source:
https://proceedings.mlr.press/v267/zheng25o.html

## 3. 2026 directions

### CALM — ICLR 2026

Huang et al., *CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design*, ICLR 2026.

- CALM studies co-evolution in which the generator itself is adapted rather than kept fixed.
- That is outside the scope of this repository: the optional real LLM backend is treated as a fixed proposal service.

OpenReview:
https://openreview.net/forum?id=x6bG2Hoqdf

### Knowledge-first AHD — 2026 preprint

Nguyen et al., *Back to the Beginning of Heuristic Design: Bridging Code and Knowledge with LLMs* (2026 preprint).

- The paper argues for treating reusable knowledge/hypotheses, rather than only code, as a search object.
- This is relevant to future interpretability work but is not implemented here.

Preprint:
https://arxiv.org/abs/2605.06123

### Generative LNS and operator co-design — 2026 preprint line

Recent 2026 work such as G-LNS extends LLM-based AHD from scalar priority rules toward coupled destroy/repair operator design.

- This broadens the design space substantially.
- The present repository deliberately stays with online bin-packing priority expressions because they are cheap to evaluate, interpretable, and easier to sandbox.

Preprint:
https://arxiv.org/abs/2602.08253

## 4. How this repository differs

| Dimension | FunSearch / EoH / later AHD systems | This repository |
| --- | --- | --- |
| Generated artifact | General program/function or thought + code | Restricted arithmetic priority expression |
| Search | Large evolutionary / tree / co-evolution variants | Small population archive with deterministic fallback |
| Execution | General generated program execution needs strong sandboxing | AST-whitelisted DSL plus separate-process timeout |
| LLM | Central generator | Optional backend; offline deterministic backend is first-class |
| Evaluation | Task-specific external evaluator | Deterministic online bin-packing simulator |
| Selection | Evaluator score | Train score for search; validation for final candidate selection |
| Final report | Benchmark-specific | Frozen final holdout + optional OR-Library evaluation |
| Budget | Often LLM calls/samples and runtime | Calls, tokens, latency, candidate evaluations, timeouts, optional cost estimate |
| Claim | Paper-specific | Research/education benchmark; no SOTA claim |

## 5. Methodological implications

The literature motivates several rules enforced here:

1. **Evaluator-grounded selection.** LLM preference is never the objective.
2. **Search/test separation.** Search sees train; candidate choice may see validation; final holdout is evaluated after freezing.
3. **Budget reporting.** Proposal calls and tokens are part of computational cost, not hidden overhead.
4. **Execution safety.** Generated artifacts must pass syntax/interface checks and runtime containment before they can influence the archive.
5. **Negative results remain valid.** A generated heuristic that does not beat Best Fit on final held-out instances is reported as such.
6. **No paper-level equivalence claim.** A small expression-space benchmark is not a reproduction of distributed FunSearch, EoH, LLaMEA, MCTS-AHD, or CALM.

## 6. Open research extensions

Natural next experiments include:

- MCTS or quality-diversity archive organization;
- thought/knowledge-level mutation in addition to expression mutation;
- TSP local-search operator discovery;
- scheduling dispatch-rule discovery;
- paired multi-seed comparisons of proposal strategies;
- richer resource accounting, including evaluator CPU time and externally supplied model prices;
- a hardened container/micro-VM executor if the search space is expanded from expressions to arbitrary Python functions.
