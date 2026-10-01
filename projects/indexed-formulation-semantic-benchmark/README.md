# Indexed Formulation and Semantic Benchmark

Implemented bounded research baseline, v0.1, 2026-10-01.

A finite indexed LP/MILP intermediate representation expands named sets, parameter tables, foreach constraints, summations and previous-period references into solver matrices. Units are exponent maps over base dimensions; coefficient-times-variable dimensions and constraint/objective dimensions are checked. CSV ingestion retains exact file/row/value-column provenance. Lagged terms at the first period require explicit boundary omission.

The compiler does not execute generated Python. Undefined variables, incomplete parameter tables, duplicate expanded constraints, unsupported terms and inconsistent units are rejected. The HiGHS solution is independently checked for bounds, integrality and constraint residuals. An independently hand-built production LP verifies the model; missing inventory carryover is a deliberate semantic mutation.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest checks -v
python study.py --output local-results.json
```

API: compile_spec(spec), solve(spec), table(csv_path,axes,dimension), production_spec(periods). Nine local checks passed. Correct and missing-carryover models diverge for 2/3/5/8 periods; objective agreement with the independent reference is checked.

## Boundaries

This is an adjacent standalone compiler/benchmark module, NOT yet wired into the existing autoformulation CLI or a live LLM. It does not implement arbitrary algebra, nonlinear/stochastic formulations, MCTS candidate search, OPT-Engine adapters, or a proof of equivalence to natural-language requirements. The mutation suite is finite and cannot establish general semantic correctness. Only dimensions, not unit scale conversions, are represented. Expansion is limited to 5000 variables.

Primary reference: https://proceedings.mlr.press/v267/astorga25a.html

Independent implementation, not paper reproduction. Existing repository license applies. See VALIDATION.json and SOURCE_REPOSITORY.md.
