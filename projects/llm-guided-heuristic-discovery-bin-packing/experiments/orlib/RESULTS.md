# OR-Library Experiment — 2026-09-19

## Purpose

Compare a deterministic grammar search with a recorded LLM-guided proposal search under the same candidate budget.

The experiment downloads the original OR-Library data at runtime and does not redistribute the benchmark files.

## Protocol

- Dataset for search/selection: **binpack1** (20 instances, 120 items each)
- Train: `u120_00`–`u120_09`
- Validation: `u120_10`–`u120_14`
- Test: `u120_15`–`u120_19`
- External frozen-candidate generalization: **binpack2** (20 instances, 250 items each)
- Candidate budget: **48** proposals per method
- LLM: **GPT-5.6 Sol**
- Test and binpack2 were not used to construct generation 2.

## Selected heuristics

### Deterministic search

```text
tightness
```

Because `tightness = 1 - residual / capacity`, maximizing this score is equivalent to minimizing residual capacity. Therefore this selected rule is behaviorally equivalent to Best Fit.

### Recorded LLM search

```text
(2.0 + tightness) if residual < 0.25 * item else remaining / capacity
```

Interpretation: when placing the item would leave a residual gap smaller than 25% of the item size, prefer the tightest feasible bin; otherwise prefer the bin with the most remaining space. It is an interpretable conditional Best-Fit/Worst-Fit hybrid.

## Results

| Split | Method | Mean bins | Best Fit mean | Mean delta | W/T/L | Exact sign p | 95% bootstrap CI |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| binpack1 train (n=10) | Deterministic | 51.1 | 51.1 | 0.0 | 0/10/0 | 1.0000 | [0.00, 0.00] |
| binpack1 train (n=10) | LLM | 50.6 | 51.1 | -0.5 | 5/4/1 | 0.2188 | [-1.00, 0.00] |
| binpack1 validation (n=5) | LLM | 52.0 | 52.2 | -0.2 | 2/2/1 | 1.0000 | [-0.80, 0.40] |
| binpack1 test (n=5) | LLM | 53.4 | 53.2 | 0.2 | 1/2/2 | 1.0000 | [-0.40, 0.80] |
| **binpack2 external (n=20)** | **LLM** | **106.7** | **107.7** | **-1.0** | **15/3/2** | **0.002350** | **[-1.50, -0.45]** |

Delta is candidate bins minus Best Fit bins, so negative values favor the candidate.

On the external binpack2 suite, the mean gap to the OR-Library offline best-known reference decreased from **6.0** bins for Best Fit-equivalent deterministic search to **5.0** bins for the LLM-discovered rule.

## Interpretation

The binpack1 test subset is only five instances and does not support a claim of improvement: the selected LLM rule is slightly worse there (+0.2 bins on average).

The untouched binpack2 external suite is substantially more informative. The frozen LLM rule reduces mean bins from **107.7** to **106.7**, wins on 15 of 20 instances, and has a paired bootstrap interval that remains below zero.

This is evidence that the discovered conditional rule generalizes beyond the 120-item search distribution to 250-item instances. It is not evidence of universal superiority: the experiment covers one OR-Library class and one external scale.

## Reproduction

```bash
python scripts/orlib_experiment.py
```

The GitHub Actions workflow `.github/workflows/orlib-experiment.yml` runs the same experiment and uploads the exact JSON result as an artifact.
