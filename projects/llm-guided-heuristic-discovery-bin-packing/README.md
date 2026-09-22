# LLM-Guided Heuristic Discovery for Online Bin Packing

A research-oriented framework for **programmatic heuristic discovery** in one-dimensional online bin packing.

The core idea is deliberately narrower than "ask an LLM to solve bin packing": an LLM proposes **interpretable priority-score expressions**, a deterministic evaluator tests them on benchmark distributions, and an evolutionary loop retains and recombines the strongest candidates.

This follows the broader program-search paradigm demonstrated by FunSearch while keeping the implementation small, auditable, reproducible, and suitable for Operations Research experiments.

## Research question

> Can an LLM-guided program-search loop discover compact online bin-packing heuristics that generalize beyond their training distribution and outperform classical First Fit / Best Fit baselines?

The repository is designed to make that question falsifiable. Every candidate is evaluated by the same deterministic harness; no LLM judgment is used as the objective.

## Scope

This project studies **online 1D bin packing**:

- bin capacity is fixed;
- items arrive sequentially;
- an item must be assigned immediately;
- previously packed items cannot be moved;
- the objective is to minimize the number of bins used.

A discovered heuristic is a scoring expression

```text
score(item, remaining, residual, fill_ratio, item_ratio, tightness, step_fraction, open_bins)
```

evaluated for every feasible open bin. The feasible bin with the largest score receives the incoming item. If no current bin can hold the item, a new bin is opened.

## Why this is genuinely learning-guided optimization

The LLM does **not** replace the optimizer and does not estimate the final answer directly.

Instead:

1. a population of executable heuristics is maintained;
2. high-performing heuristics are selected;
3. the LLM receives elite programs and benchmark feedback;
4. it proposes new candidate scoring rules;
5. candidates are syntax-checked and sandboxed to a restricted expression language;
6. the deterministic evaluator measures objective quality;
7. only measured performance determines survival.

That distinction matters: the language model supplies a learned proposal distribution, while the optimization loop and objective remain explicit.

## Architecture

```text
                     +----------------------+
                     |  classical seeds     |
                     |  best-fit / variants |
                     +-----------+----------+
                                 |
                                 v
+-------------+        +---------+----------+        +------------------+
| LLM proposer|------->| candidate programs |------->| AST validator    |
+-------------+        +--------------------+        +--------+---------+
                                                              |
                                                              v
                                                     +--------+---------+
                                                     | benchmark       |
                                                     | evaluator       |
                                                     +--------+---------+
                                                              |
                                                              v
                                                     +--------+---------+
                                                     | population DB   |
                                                     | select / evolve |
                                                     +------------------+
```

The LLM backend is optional. A deterministic proposer is included so the complete search/evaluation pipeline can be tested without API access.

## Safety boundary

Generated text is **never executed as arbitrary Python**.

Candidates are restricted to arithmetic expressions over a small whitelist of variables, operators, and mathematical functions. Expressions are parsed with Python's `ast` module and rejected if they contain unsupported syntax.

This makes the experiment reproducible and sharply separates *program search* from unrestricted code execution.

## Benchmarks

The initial benchmark generator contains three synthetic families:

- Uniform item sizes
- Weibull-like item sizes
- Bimodal item sizes

Training and holdout seeds are separated. Reported metrics include:

- mean bins used;
- excess bins over the volume lower bound;
- relative excess ratio;
- win/tie/loss counts against Best Fit;
- per-distribution generalization.

The lower bound is

```text
LB = ceil(sum(item_sizes) / bin_capacity)
```

and is used only as a reference bound, not as an online optimum.

## Experiment protocol

The repository now includes a leakage-resistant train/validation/generalization protocol, paired statistical comparison against Best Fit, and an OR-Library loader for the classic one-dimensional `binpack1.txt`–`binpack8.txt` format.

See [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) for the full reporting rules.

### Phase A — deterministic sanity check

Verify that:

- First Fit and Best Fit are correct;
- candidate expressions reproduce known behavior;
- the evaluator is deterministic under fixed seeds.

### Phase B — program search

Start from a small population of human-readable seeds and evolve candidates on the training suite.

### Phase C — generalization

Freeze the discovered heuristic and evaluate it on:

- unseen random seeds;
- larger item counts;
- shifted distributions;
- OR-Library instances where licensing/distribution permits local use.

### Phase D — statistical comparison

Compare the discovered program with Best Fit on exactly the same instances. Reports include mean/median paired bin deltas, wins/ties/losses, an exact two-sided sign test, and a deterministic 95% paired bootstrap confidence interval.

### Phase E — external OR-Library evaluation

Freeze the candidate before evaluating external OR-Library data. The offline best-known field is retained as a reference only; it is not interpreted as the optimum of the online arrival-order problem.

## Project layout

```text
src/llm_binpack/
    problem.py       # online bin-packing simulator
    baselines.py     # First Fit / Best Fit / Worst Fit
    expressions.py   # safe AST-based heuristic language
    benchmarks.py    # reproducible instance generators
    evaluator.py     # scoring and benchmark metadata
    paired.py        # exact sign test + paired bootstrap
    orlib.py         # OR-Library binpack parser
    protocol.py      # train/validation/generalization splits
    proposers.py     # deterministic + optional LLM proposers
    search.py        # evolutionary discovery loop
    cli.py           # command-line entry point

tests/
    test_problem.py
    test_expressions.py
    test_evaluator.py
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Run a deterministic discovery experiment:

```bash
llm-binpack discover --provider deterministic --generations 20 --population 24
```

Run with an OpenAI-backed proposer:

```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=...
llm-binpack discover --provider openai --model YOUR_MODEL --generations 20 --population 24
```

Evaluate a frozen candidate across train, validation, and generalization:

```bash
llm-binpack protocol --expression "-residual"
```

Run a paired comparison on an external OR-Library file:

```bash
llm-binpack compare --expression "-residual" --orlib data/binpack1.txt
```

The OpenAI integration uses the Responses API and reads only the model's textual output. The evaluator remains local and deterministic. External benchmark files are not redistributed by this repository.

## Example candidate language

Valid examples:

```text
-residual
-(residual ** 2)
-tightness + 0.10 * fill_ratio
-abs(residual) + 0.15 * item_ratio
-(residual ** 2) - 0.05 * open_bins
```

Invalid examples include imports, attribute access, loops, comprehensions, file access, network access, or arbitrary function calls.

## Frozen scale-generalization result

A frozen LLM-discovered heuristic was evaluated with no further tuning on OR-Library `binpack2`, `binpack3`, and `binpack4`:

```text
(2.0 + tightness) if residual < 0.25 * item else remaining / capacity
```

Compared with Best Fit, the mean paired bin-count deltas were:

- binpack2 (250 items): **-1.00 bins**, 15/3/2 wins/ties/losses
- binpack3 (500 items): **-2.30 bins**, 18/2/0
- binpack4 (1000 items): **-5.20 bins**, 20/0/0

See [experiments/orlib/FROZEN_SCALE_GENERALIZATION.md](experiments/orlib/FROZEN_SCALE_GENERALIZATION.md) for the full paired statistics and protocol.

## Success criteria

A discovered heuristic is interesting only if it satisfies more than one condition:

1. improves over strong classical baselines on the training distribution;
2. retains the improvement on held-out instances;
3. generalizes to different instance sizes or distributions;
4. remains compact enough to inspect;
5. survives paired statistical comparison.

Training-set wins alone are treated as overfitting, not discovery.

## References

- Romera-Paredes, B. et al. **Mathematical discoveries from program search with large language models.** *Nature* 625, 468–475 (2024). DOI: 10.1038/s41586-023-06924-6
- Google DeepMind FunSearch repository: https://github.com/google-deepmind/funsearch
- Martello, S. & Toth, P. *Knapsack Problems: Algorithms and Computer Implementations.* Wiley, 1990.

## Status

Initial research implementation under active development.
