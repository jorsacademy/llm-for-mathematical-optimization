# Frozen Scale Generalization — OR-Library binpack2–4

## Frozen heuristic

```text
(2.0 + tightness) if residual < 0.25 * item else remaining / capacity
```

This heuristic was selected before `binpack3` and `binpack4` were evaluated. The evaluation script contains no search, proposal generation, or tuning.

## Results vs Best Fit

| Dataset | Items / instance | LLM mean bins | Best Fit mean | Mean delta | W/T/L | Exact sign p | 95% paired bootstrap CI |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| binpack2 | 250 | 106.70 | 107.70 | -1.00 | 15/3/2 | 0.002350 | [-1.50, -0.45] |
| binpack3 | 500 | 209.70 | 212.00 | -2.30 | 18/2/0 | 0.00000763 | [-2.85, -1.75] |
| binpack4 | 1000 | 415.15 | 420.35 | -5.20 | 20/0/0 | 0.00000191 | [-5.95, -4.40] |

Delta is candidate bins minus Best Fit bins, so negative values favor the frozen heuristic.

## Scale effect

The absolute advantage over Best Fit increases with instance size:

- 250 items: **1.00 fewer bins**
- 500 items: **2.30 fewer bins**
- 1000 items: **5.20 fewer bins**

The win rate also rises from **15/20** on binpack2 to **18/20** on binpack3 and **20/20** on binpack4.

This is a strong scale-generalization result within the Falkenauer uniform class. It is still not a claim of universal superiority because binpack2–4 share the same broad benchmark family and capacity regime.

## Reproduction

```bash
python scripts/frozen_scale_generalization.py
```

The GitHub Actions workflow `.github/workflows/frozen-scale-generalization.yml` runs the same frozen evaluation and uploads the JSON artifact.
