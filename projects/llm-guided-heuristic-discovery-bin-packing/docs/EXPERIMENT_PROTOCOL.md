# Experiment Protocol

This document defines the evaluation discipline for heuristic discovery experiments in this repository.

The goal is to separate **search performance** from **generalization evidence**. A candidate that wins only on the instances used to evolve it is not treated as a discovered improvement.

## 1. Data partitions

The canonical synthetic protocol contains three deterministic, non-overlapping partitions.

| Split | Default size | Purpose |
| --- | ---: | --- |
| Train | 12 instances × 3 families × 200 items | Program search only |
| Validation | 12 × 3 × 200 items | Diagnostics and candidate/model selection |
| Generalization | 12 × 3 × 500 items | Final frozen-candidate report |

The random seeds for all three partitions are disjoint.

The generalization partition also changes problem scale from 200 to 500 arriving items. This is a simple scale-shift test; it is not claimed to cover every form of distribution shift.

## 2. Leakage rule

The search loop may evaluate candidates on **train only**.

Validation may be used to choose between already discovered candidates, tune search settings, or diagnose overfitting.

Once a final expression has been selected, freeze it before evaluating **generalization** and any external OR-Library file.

Do not repeatedly inspect generalization results and then revise the heuristic. Doing so converts the generalization set into another validation set.

## 3. Paired comparison

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

## 4. OR-Library external benchmark

The loader accepts the classic one-dimensional OR-Library `binpack1.txt` through `binpack8.txt` format:

```text
number_of_instances
instance_name
capacity item_count best_known
item_1
...
item_n
```

The third header field is stored as `reference_optimum`. It comes from the offline benchmark metadata and is **not** treated as an achievable optimum for the online arrival-order problem.

The repository does not redistribute OR-Library files. Download them from the original source when running external experiments:

- OR-Library index: https://people.brunel.ac.uk/~mastjjb/jeb/info.html
- Data directory: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/

## 5. Commands

Evaluate a frozen expression on the complete synthetic protocol:

```bash
llm-binpack protocol \
  --expression "(2.0 + fill_ratio) if residual < 0.15 * capacity else remaining"
```

Add an external OR-Library file:

```bash
llm-binpack protocol \
  --expression "(2.0 + fill_ratio) if residual < 0.15 * capacity else remaining" \
  --orlib data/binpack1.txt
```

Run a paired comparison on one split:

```bash
llm-binpack compare \
  --expression "-residual" \
  --suite validation
```

Run only on OR-Library:

```bash
llm-binpack compare \
  --expression "-residual" \
  --orlib data/binpack1.txt
```

## 6. Interpretation

A candidate should not be described as better than Best Fit merely because its training fitness is higher.

A stronger result has the following pattern:

1. improvement on train;
2. no collapse on validation;
3. negative paired mean delta on generalization;
4. bootstrap interval and win/loss pattern consistent with the effect;
5. similar behavior on external OR-Library instances;
6. compact, interpretable expression;
7. repeated discovery across independent search seeds.

This protocol is intentionally conservative. It is designed to make negative results informative and positive results harder to overstate.
