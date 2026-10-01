# Solver-Informed Policy Training

Implemented training-mechanics baseline with an optional local causal-LM adapter, v0.1, 2026-10-01.

A model scores eight restricted formulation completions: min/max objective, le/ge constraint and real/integer variable. Candidate probabilities are normalized from masked completion token log-likelihoods. The code compares an unchanged model, supervised fitting and sampled REINFORCE with a detached expected-reward baseline plus KL penalty. Tests verify finite gradients and actual parameter updates.

Rewards combine exact HiGHS solutions with known synthetic semantics and feasibility probes. A test deliberately uses two models with the same optimum but different variable domains: objective agreement alone must not receive semantic credit. Arbitrary generated code is never executed.

## Run the executed tiny-model benchmark

```bash
python -m pip install -r requirements.txt
python -m unittest checks -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python study.py --output local-results.json
```

Eight local checks passed. The actual executed backend is a small character-level GRU, NOT a pretrained large language model. Three seeded 50-update comparisons on 16 training tasks all remain at 12.5% held-out template accuracy in this short run. This is a negative learning result and a mechanics test, not evidence that solver-informed training improves modeling quality.

## Optional local Hugging Face backend

```bash
python -m pip install transformers
python study.py --local-model /path/to/local/checkpoint --output local-hf-results.json
```

The adapter uses local_files_only=True and trust_remote_code=False, performs gradient-based weight updates and saves a checkpoint. It was NOT executed in the current validation environment. No checkpoint was downloaded, no API was called, and no large-model training result is claimed. Hardware, checkpoint compatibility, memory use and a suitable learning-rate schedule must be validated separately before a meaningful pretrained-model study.

## Boundaries

One-variable synthetic MILPs and eight fixed DSL completions only. Not free-form model generation, general semantic verification, an SIRL reproduction, or a claim of LLM reasoning improvement. All methods solve syntactically valid candidates; solve completion is therefore deliberately separated from semantic correctness. Existing production autoformulation code is unchanged.

Primary research reference: https://arxiv.org/abs/2505.11792

Independent implementation. Existing repository license applies. Source fingerprints and scope are in VALIDATION.json.
