# Experiment Protocol

This document defines the evaluation discipline for heuristic-discovery experiments in this repository.

The goal is to separate **search performance**, **candidate selection**, **final generalization evidence**, and **computational budget**. A candidate that wins only on the instances used to evolve it is not treated as a discovered improvement.

## 1. Data partitions

The canonical synthetic protocol contains three deterministic, non-overlapping partitions.

| Split | Default size | Purpose |
| --- | ---: | --- |
| Train | 12 instances × 3 families × 200 items | Program search only |
| Validation | 12 × 3 × 200 items | Candidate/model selection and diagnostics |
| Generalization | 12 × 3 × 500 items | Final frozen-candidate report |

The random seeds for all three partitions are disjoint.

The generalization partition also changes problem scale from 200 to 500 arriving items. This is a simple scale-shift test; it is not claimed to cover every form of distribution shift.

The 'discover' command follows the same discipline with explicit train, validation, and final-holdout suites. It evolves only on train, selects among the final train-ranked population using validation fitness, freezes one expression, and only then evaluates the final holdout.

## 2. Leakage rule

The search loop may evaluate candidates on **train only**.

Validation may be used to choose between already discovered candidates, tune search settings, or diagnose overfitting.

Once a final expression has been selected, freeze it before evaluating **generalization/final holdout** and any external OR-Library file.

Do not repeatedly inspect final-holdout results and then revise the heuristic. Doing so converts the final holdout into another validation set.

## 3. Candidate execution boundary

Generated artifacts are intentionally restricted to a small arithmetic expression language.

Before execution, every candidate is:

1. parsed in eval mode;
2. checked against an AST node whitelist;
3. checked against a variable/function whitelist;
4. bounded in source length, constant magnitude, and exponent magnitude.

Imports, assignments, attribute access, subscripts, comprehensions, lambdas, arbitrary names, files, network primitives, state, and arbitrary calls are therefore unavailable through the candidate language.

Accepted expressions are then evaluated in a **separate process**. The parent process applies a configurable wall-clock timeout and terminates the worker on overrun. Runtime exceptions are converted into failed-candidate records and do not crash or enter the search archive.

This is a deliberately narrow sandbox. It is materially safer than importing arbitrary generated Python, but it is not presented as a general container, VM, seccomp profile, or security boundary for unrestricted hostile code. If future versions expand to full Python heuristic functions, a stronger OS/container isolation layer is required.

## 4. Proposal backends

Two proposal modes are supported.

### Offline/deterministic

DeterministicProposer is a seeded grammar/mutation backend. It exists for:

- CI;
- reproducibility;
- offline experimentation;
- a random-generated-heuristic baseline;
- fallback when an optional external LLM is unavailable or its budget is exhausted.

### Optional OpenAI backend

OpenAIProposer is optional and reads credentials through the OpenAI client/environment. No API key is stored in the repository.

CI never instantiates the paid backend.

The Responses API output is treated only as a proposal source. The local deterministic evaluator remains authoritative.

## 5. Computational budget accounting

Proposal cost is part of the method.

For every proposer call, the repository can retain:

- API/proposer call count;
- requested and returned proposal count;
- input tokens;
- cached input tokens;
- output tokens;
- total tokens;
- proposal latency;
- failed proposal calls.

Search-level accounting also reports:

- candidate evaluations;
- duplicate candidates;
- invalid candidates;
- sandbox timeouts;
- sandbox/runtime failures;
- cumulative candidate-evaluation wall time;
- fallback proposer usage.

Optional hard controls are available for maximum LLM calls, maximum reported total tokens, and maximum output tokens per response.

### Cost estimates

No model price is hard-coded because provider pricing changes over time.

If the user supplies input/output USD-per-million-token rates, the report computes an **estimated** token cost from reported API usage. Without explicit rates, token counts are still reported and cost remains null.

The estimate does not independently reconstruct provider invoices and should not be described as audited billing.

## 6. Baselines

The final-holdout report distinguishes:

- First Fit;
- Best Fit;
- Worst Fit;
- a fixed hand-written scoring expression;
- a seeded grammar-generated random expression;
- the selected evolutionary/LLM-loop candidate.

Best Fit remains the principal paired comparator because the discovered priority expressions operate in the same online packing setting.

A method that loses to these baselines is not relabeled as successful.

## 7. Paired comparison

Every candidate is compared against Best Fit on exactly the same ordered instances.

Reported paired statistics are:

- mean candidate bins;
- mean Best Fit bins;
- mean and median paired delta, defined as candidate bins minus Best Fit bins;
- wins / ties / losses;
- exact two-sided sign-test p-value, with ties excluded;
- deterministic 95% paired bootstrap confidence interval for the mean bin delta.

A negative mean delta favors the candidate.

The sign test is distribution-free but uses only the direction of non-tied differences. The bootstrap interval estimates the uncertainty of the mean paired effect. Neither should be treated as sufficient evidence in isolation.

## 8. OR-Library external benchmark

The loader accepts the classic one-dimensional OR-Library binpack1.txt through binpack8.txt format:

~~~text
number_of_instances
instance_name
capacity item_count best_known
item_1
...
item_n
~~~

The third header field is stored as reference_optimum. It comes from the offline benchmark metadata and is **not** treated as an achievable optimum for the online arrival-order problem.

The repository does not redistribute OR-Library files. Download them from the original source when running external experiments:

- OR-Library index: https://people.brunel.ac.uk/~mastjjb/jeb/info.html
- Data directory: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/

## 9. Commands

Run deterministic discovery with an isolated candidate timeout:

~~~bash
llm-binpack discover \
  --provider deterministic \
  --generations 20 \
  --population 24 \
  --candidate-timeout 10
~~~

Run with an optional OpenAI proposer and explicit budgets:

~~~bash
export OPENAI_API_KEY=...

llm-binpack discover \
  --provider openai \
  --model YOUR_MODEL \
  --max-llm-calls 20 \
  --max-total-tokens 200000 \
  --max-output-tokens 1500 \
  --candidate-timeout 10
~~~

If cost estimation is wanted, pass current rates explicitly:

~~~bash
llm-binpack discover \
  --provider openai \
  --model YOUR_MODEL \
  --input-cost-per-million INPUT_RATE \
  --output-cost-per-million OUTPUT_RATE
~~~

Evaluate a frozen expression on the complete canonical synthetic protocol:

~~~bash
llm-binpack protocol \
  --expression "(2.0 + fill_ratio) if residual < 0.15 * capacity else remaining"
~~~

Add an external OR-Library file:

~~~bash
llm-binpack protocol \
  --expression "(2.0 + fill_ratio) if residual < 0.15 * capacity else remaining" \
  --orlib data/binpack1.txt
~~~

Run a paired comparison on one split:

~~~bash
llm-binpack compare \
  --expression "-residual" \
  --suite validation
~~~

## 10. Interpretation

A candidate should not be described as better than Best Fit merely because its training fitness is higher.

A stronger result has the following pattern:

1. improvement on train;
2. validation selection without looking at final holdout;
3. negative paired mean delta on the frozen final holdout;
4. bootstrap interval and win/loss pattern consistent with the effect;
5. similar behavior on external OR-Library instances;
6. compact, interpretable expression;
7. repeated discovery across independent search seeds;
8. computational cost that remains defensible relative to the observed gain.

Training-set wins, CI smoke runs, or one favorable LLM trajectory are not scientific benchmark conclusions.
