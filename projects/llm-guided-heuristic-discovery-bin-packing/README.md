# LLM-Guided Heuristic Discovery for Online Bin Packing

A reproducible research benchmark for **LLM-assisted automatic heuristic design (AHD)** in one-dimensional online bin packing.

The project studies the loop:

LLM / offline proposer → candidate heuristic → syntax/interface validation → isolated deterministic evaluation → score → retain/mutate/revise → next generation.

It is intentionally different from mathematical-optimization autoformulation. The parent repository studies translating problem descriptions into mathematical models; this project studies **designing executable decision heuristics** and evaluating them as algorithms.

This is an independent research/education implementation. It is not a reproduction of FunSearch, Evolution of Heuristics (EoH), LLaMEA, MCTS-AHD, or another published codebase.

## Motivation

Many optimization problems have heuristics that are easy to evaluate but difficult to design. LLM-based AHD uses a language model as a proposal mechanism while keeping the objective and evaluator external.

That distinction is central here:

- the LLM does not decide whether its own heuristic is good;
- generated candidates do not receive credit until they pass validation and deterministic evaluation;
- training/development performance is not final evidence;
- proposal calls, tokens, evaluator work, and failures are computational cost.

The first domain is online bin packing because a heuristic can be represented as a compact priority rule, evaluated cheaply, and independently compared with classical baselines.

## Research Question

> Can a proposal-and-evaluation search loop discover compact online bin-packing priority rules that generalize beyond the instances used for search and remain competitive with classical First Fit / Best Fit / Worst Fit baselines under explicit computational budgets?

Secondary questions are:

- Does a train improvement survive validation-based selection and final held-out evaluation?
- Does an LLM proposal backend add value over a seeded grammar/mutation backend?
- How much proposal and evaluator budget is required for any observed improvement?
- Are discovered rules still useful under scale/distribution shift?
- Does a compact restricted expression language provide enough search capacity without requiring arbitrary generated Python execution?

## Mathematical / Decision Problem

Items arrive sequentially. For capacity C and item sizes a_t:

- each item must be assigned immediately;
- previously packed items cannot be moved;
- each bin load must not exceed C;
- the objective is to minimize the number of opened bins.

The discovered heuristic is not an offline exact solver. It is a priority score evaluated for every currently feasible open bin:

~~~text
score(
    item,
    capacity,
    remaining,
    residual,
    fill_ratio,
    item_ratio,
    tightness,
    step_fraction,
    open_bins
)
~~~

The feasible bin with maximum score receives the item. If no open bin can hold the item, a new bin is opened.

## Methodology

### Search loop

The search maintains an archive of measured candidate expressions.

1. seed the archive with small human-readable rules;
2. rank candidates by deterministic train fitness;
3. expose top candidates and measured feedback to the proposer;
4. generate new expressions;
5. validate the expression language;
6. evaluate each unique candidate in an isolated process;
7. retain measured candidates;
8. repeat for the configured number of generations.

The final train-ranked population is then evaluated on validation. One expression is selected by validation fitness, frozen, and only then evaluated on the final holdout.

The final holdout is not used to generate, mutate, rank, or select the heuristic.

### Proposal backends

Two backends are available.

**DeterministicProposer**

- seeded grammar/mutation generator;
- no network;
- no API key;
- deterministic under fixed seed;
- used by CI;
- used as a fallback and random-generated baseline.

**OpenAIProposer**

- optional OpenAI Responses API integration;
- API key is read through the normal OpenAI client/environment;
- no key is stored in the repository;
- model output is only a proposal source;
- reported API usage is captured when available.

CI never makes a real paid API call.

## Safety / Sandbox Boundary

Generated text is **not imported as arbitrary Python**.

Candidates are limited to an arithmetic expression DSL. The AST validator only permits:

- whitelisted state variables;
- numeric constants;
- arithmetic/comparison/conditional expressions;
- whitelisted mathematical functions.

The validator rejects imports, assignments, attributes, subscripts, lambdas, comprehensions, arbitrary names, arbitrary calls, file/network primitives, and stateful program structures. Expression length, numeric constants, and exponent magnitude are also bounded.

After syntax/interface validation, evaluation occurs in a **separate process**. The parent process applies a configurable wall-clock timeout and terminates an overrun worker. Runtime errors become failed candidates rather than entering the archive.

This is deliberately narrower than a general arbitrary-code sandbox. If a future research version evolves full Python functions/operators, it should move to a stronger OS/container/micro-VM isolation boundary instead of treating this expression executor as sufficient.

## Baselines

The repository contains or reports:

- First Fit;
- Best Fit;
- Worst Fit;
- a fixed hand-written priority expression;
- a seeded grammar-generated random expression;
- deterministic evolutionary search;
- optional LLM-guided evolutionary search.

Best Fit is the principal paired comparator for discovered priority rules.

A negative result is valid. The project does not assume the learned/evolved candidate wins.

## Train / Validation / Final Holdout Discipline

The discovery command uses three disjoint seed namespaces:

- **train**: candidate search only;
- **validation**: final candidate selection among train-discovered candidates;
- **final holdout**: one frozen-candidate report.

The canonical protocol command similarly provides train / validation / generalization suites.

Using final-holdout results to revise a rule invalidates that split as final evidence.

See docs/EXPERIMENT_PROTOCOL.md.

## Evaluation Metrics

Candidate quality:

- mean bins used;
- excess bins above the volume lower bound;
- relative excess ratio;
- win/tie/loss counts against Best Fit;
- per-family metrics.

Paired final comparisons:

- mean candidate-minus-Best-Fit bin delta;
- median paired delta;
- exact two-sided sign test;
- deterministic paired bootstrap 95% confidence interval.

The volume bound

~~~text
ceil(sum(item_sizes) / capacity)
~~~

is a lower bound, not the online optimum.

## Computational Budget

The search reports evaluator work separately from proposal work.

Candidate/evaluator telemetry includes:

- candidate evaluations;
- duplicates;
- invalid candidates;
- sandbox timeouts;
- runtime/evaluation failures;
- cumulative candidate-evaluation wall time.

Proposal telemetry includes:

- proposer/API call count;
- requested proposals;
- returned proposals;
- failed calls;
- input tokens;
- cached input tokens;
- output tokens;
- total tokens;
- proposal latency.

Optional limits:

- maximum real-LLM calls;
- maximum reported total tokens;
- maximum output tokens per response;
- candidate evaluation timeout.

### API cost estimates

Provider/model prices are **not hard-coded** because prices change.

If current USD-per-million input/output token rates are supplied through CLI arguments, the result JSON computes an estimated token cost. Otherwise token usage is still reported and the cost field stays null.

This estimate is research accounting, not an audited provider invoice.

## Reproducibility

Create an environment:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
~~~

Run tests:

~~~bash
pytest
~~~

Run deterministic discovery:

~~~bash
llm-binpack discover \
  --provider deterministic \
  --generations 20 \
  --population 24 \
  --candidate-timeout 10 \
  --output results/deterministic.json
~~~

The deterministic backend allows the complete proposal/search/sandbox/selection/reporting path to run without external API access.

## Optional Real LLM Backend

Install:

~~~bash
python -m pip install -e ".[openai]"
export OPENAI_API_KEY=...
~~~

Run with explicit budgets:

~~~bash
llm-binpack discover \
  --provider openai \
  --model YOUR_MODEL \
  --generations 20 \
  --population 24 \
  --max-llm-calls 20 \
  --max-total-tokens 200000 \
  --max-output-tokens 1500 \
  --candidate-timeout 10 \
  --output results/openai.json
~~~

Optional cost estimation:

~~~bash
llm-binpack discover \
  --provider openai \
  --model YOUR_MODEL \
  --input-cost-per-million INPUT_RATE \
  --output-cost-per-million OUTPUT_RATE
~~~

No current provider price is assumed by the repository.

## Frozen Candidate Evaluation

Evaluate a fixed candidate across canonical train, validation, and generalization splits:

~~~bash
llm-binpack protocol \
  --expression "(2.0 + tightness) if residual < 0.25 * item else remaining / capacity"
~~~

Compare on one split:

~~~bash
llm-binpack compare \
  --expression "-residual" \
  --suite validation
~~~

An external OR-Library file can be loaded locally without redistributing it:

~~~bash
llm-binpack compare \
  --expression "-residual" \
  --orlib data/binpack1.txt
~~~

## Repository Structure

~~~text
.
├── docs/
│   ├── EXPERIMENT_PROTOCOL.md
│   └── LITERATURE_MAP.md
├── experiments/
├── scripts/
├── src/llm_binpack/
│   ├── baselines.py
│   ├── benchmarks.py
│   ├── cli.py
│   ├── evaluator.py
│   ├── expressions.py
│   ├── orlib.py
│   ├── paired.py
│   ├── problem.py
│   ├── proposers.py
│   ├── protocol.py
│   ├── sandbox.py
│   ├── search.py
│   └── telemetry.py
├── tests/
│   ├── test_evaluator.py
│   ├── test_expressions.py
│   ├── test_orlib.py
│   ├── test_paired.py
│   ├── test_problem.py
│   ├── test_proposers.py
│   ├── test_protocol.py
│   └── test_sandbox.py
├── pyproject.toml
└── README.md
~~~

The active GitHub Actions workflow is at the **parent repository root**. The nested workflow is retained so this project remains easy to extract as a standalone repository, but GitHub does not execute nested workflow files inside a monorepo.

## CI

The parent repository CI contains a scoped heuristic-design job for Python 3.11 and 3.12.

It checks:

- editable installation;
- dependency consistency;
- Ruff lint;
- Ruff formatting;
- pytest methodology tests;
- deterministic end-to-end discovery smoke run;
- smoke artifact schema.

No paid LLM is used in CI.

Tests cover:

- classical baseline behavior;
- deterministic benchmark generation;
- AST rejection of unsafe expressions;
- process-isolated evaluation;
- wall-clock timeout behavior;
- runtime exception containment;
- deterministic search;
- validation-only candidate selection;
- proposer token/call/cost accounting using a fake offline client;
- proposal budget enforcement;
- OR-Library parsing;
- paired statistical reporting.

## Literature Context

The design is informed by, but is not a reproduction of, several AHD lines.

### FunSearch

Romera-Paredes et al., *Mathematical discoveries from program search with large language models*, Nature 625, 468–475 (2024).

FunSearch combines an LLM proposal mechanism with an external evaluator and program database, including online bin packing as an algorithm-design example.

https://doi.org/10.1038/s41586-023-06924-6

### Evolution of Heuristics (EoH)

Liu et al., *Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model*, ICML 2024, PMLR 235.

EoH explicitly evolves heuristic thoughts and code and treats LLM-query budget as an efficiency dimension.

https://proceedings.mlr.press/v235/liu24bs.html

### LLaMEA

van Stein and Bäck, *LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics*, IEEE Transactions on Evolutionary Computation 29(2), 331–345 (2025).

LLaMEA uses runtime evaluation feedback to iteratively generate/refine optimization algorithms.

https://doi.org/10.1109/TEVC.2024.3497793

### MCTS-AHD

Zheng et al., *Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design*, ICML 2025, PMLR 267.

MCTS-AHD organizes generated heuristics in a tree rather than relying only on a fixed population.

https://proceedings.mlr.press/v267/zheng25o.html

### 2026 directions

CALM (ICLR 2026) studies co-evolution of algorithms and the generator itself. Other 2026 work explores knowledge-first AHD and operator-level generation such as generative LNS. These are research context, not implemented features.

See docs/LITERATURE_MAP.md for a fuller chronology and scope comparison.

## Recorded Experiment

A previously recorded experiment froze the expression:

~~~text
(2.0 + tightness) if residual < 0.25 * item else remaining / capacity
~~~

before evaluating larger OR-Library scales. The repository contains the corresponding experiment notes under experiments/orlib/.

Those recorded results should be interpreted as one empirical trajectory, not a claim that the framework, LLM backend, or expression is universally superior.

## Experimental Interpretation

Possible outcomes include:

- train improvement that disappears on validation;
- validation-selected improvement that disappears on final holdout;
- fewer bins but much higher proposal/evaluator budget;
- deterministic grammar search matching or beating the real-LLM proposer;
- a real-LLM backend producing useful structural mutations under fewer calls;
- Best Fit remaining strongest;
- scale/distribution shift reversing a development-set gain.

None of these outcomes is hidden or redefined as success.

## Limitations

- The first domain is synthetic/benchmark online 1D bin packing, not an industrial deployment.
- The search space is a restricted priority-expression DSL, not arbitrary algorithm synthesis.
- The process sandbox is not a hardened container for unrestricted hostile Python.
- The current evolutionary archive is intentionally simpler than island-based, MCTS, co-evolutionary, or quality-diversity AHD systems.
- LLM behavior depends on model/provider versions and stochastic sampling.
- Cost estimation requires user-supplied current prices.
- Wall-clock measurements depend on hardware and process-start overhead.
- OR-Library offline best-known values are not automatically equivalent to online arrival-order optima.
- One search run is not sufficient for a strong scientific conclusion.

## Claims Boundary

This repository does **not** claim:

- state-of-the-art AHD performance;
- reproduction of FunSearch, EoH, LLaMEA, MCTS-AHD, or CALM;
- that an LLM is necessary for heuristic discovery;
- that development-set improvement implies final decision quality;
- that a generated heuristic is universally better than Best Fit;
- that the expression executor safely supports arbitrary untrusted Python;
- that estimated API cost equals an audited bill;
- that CI smoke output is a scientific benchmark;
- production readiness or industrial savings.

The defensible claim is narrower: the repository provides an auditable AHD benchmark in which proposal generation, restricted execution, deterministic scoring, data splitting, baseline comparison, and computational-budget accounting are explicit and testable.

## Related Repository Context

Within this portfolio:

- the parent autoformulation project studies natural language → mathematical model generation;
- this project studies proposal → heuristic program → measured decision performance;
- learning-augmented solver projects study ML guidance inside exact optimization/search machinery.

The code remains standalone at the project level; these research lines are conceptually related but are not runtime dependencies.

## References

- Romera-Paredes, B. et al. (2024). *Mathematical discoveries from program search with large language models*. Nature 625, 468–475. https://doi.org/10.1038/s41586-023-06924-6
- Liu, F. et al. (2024). *Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model*. ICML 2024, PMLR 235. https://proceedings.mlr.press/v235/liu24bs.html
- van Stein, N., Bäck, T. (2025). *LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics*. IEEE Transactions on Evolutionary Computation 29(2), 331–345. https://doi.org/10.1109/TEVC.2024.3497793
- Zheng, Z., Xie, Z., Wang, Z., Hooi, B. (2025). *Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design*. ICML 2025, PMLR 267. https://proceedings.mlr.press/v267/zheng25o.html
- Huang, Z. et al. (2026). *CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design*. ICLR 2026. https://openreview.net/forum?id=x6bG2Hoqdf
- Nguyen, K. et al. (2026 preprint). *Back to the Beginning of Heuristic Design: Bridging Code and Knowledge with LLMs*. https://arxiv.org/abs/2605.06123
- Zhao, B., Wang, H., Zeng, L. (2026 preprint). *G-LNS: Generative Large Neighborhood Search for LLM-Based Automatic Heuristic Design*. https://arxiv.org/abs/2602.08253
- Martello, S., Toth, P. (1990). *Knapsack Problems: Algorithms and Computer Implementations*. Wiley.
