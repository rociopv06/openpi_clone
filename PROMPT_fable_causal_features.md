# Fable task — PI05 SAE causal feature experiments (LIBERO)

## Context
This is the PI05 SAE-interpretability project. We train sparse autoencoders (SAEs) on
PaliGemma activations inside the PI05 policy and interpret the resulting features on the
LIBERO manipulation benchmark. Prior finding: `lang_layer_12` (~67% through the language
model) has near-perfectly task-specific features — roughly one feature per LIBERO task,
60–65x fold enrichment — which raises a memorization-vs-generalization question. Current
active layer is `lang_layer_9` (experiment `pi05_libero_lang_layer_9_k32_exp4`).

Infra:
- SAE checkpoints: `/storage/project/r-agarg35-0/rvaldes6/sae_checkpoints`
  (exp4: 8k features; layers lang_layer_3/6/9/12/17, expert_layer_0, vision_proj)
- Interpretation results: `/storage/project/r-agarg35-0/rvaldes6/sae_interpretations/`
- `feature_task_enrichment.json` exists for all exp4 layers
- `visualize_top_examples.py` supports `--feature_ids` for targeted visualization
- PI05 checkpoint: `/storage/project/r-agarg35-0/rvaldes6/pi05_pytorch/model.safetensors`
- PACE/SLURM: account `gts-agarg35-ideasci23_dgx`; venv
  `/storage/project/r-agarg35-0/rvaldes6/openpi_clone/.venv/bin/python`

## Goal
Test whether task-specific SAE features decompose into **object features** vs
**action-primitive features**, and whether the policy composes them — i.e. move from
correlational interpretation to causal evidence.

## Experiments
Ordered pipeline: cheap read-only analysis first (Phase A) to identify and label candidate
features, then causal interventions (Phase C) building toward the payoff. Do them in order.

### Phase A — read-only feature characterization (do first)

#### 1. Feature sensitivity via prompt variation (read-only, no steering)
Run varied language prompts through the model and read off feature activations. Compare
e.g. `"place X in basket"` vs `"apple is in basket, move can to the right"`. Identify which
features fire, and how activations shift as the prompt changes. Use this to label features
as object-driven vs action-driven vs task-label.

#### 2. Decoder co-similarity histogram
For features that activate within the same LIBERO task, compute pairwise cosine similarity
of their SAE **decoder** vectors. Produce two overlaid histograms:
- same-task cosimilarity (within each task, pooled)
- all-tasks cosimilarity (global baseline over all feature pairs)
Question: do same-task features cluster more tightly than the global distribution? Report
the distributions, means, and a separation statistic.

### Phase C — causal interventions (builds toward the payoff)

#### 3. Steering (harness check)
Establish the steering harness works: steer with a known task feature on its own task prompt
and confirm the expected behavioral effect before doing anything subtler.

#### 4. Cross-task steering
Steer with a task feature while running a *different* task's prompt; record the behavioral
effect (does the policy drift toward the injected task?).

#### 5. Ablation
Ablate a feature from its own task prompt and record the effect on behavior/output.

#### 6. Compositionality causal test (KEY RESULT)
Dissociate object vs action features on the "put in basket" family:
- Remove the **"chocolate"** (object) feature → expect the policy CANNOT do
  chocolate-in-basket, but CAN still do can-in-basket.
- Remove the **"put in basket"** (action) feature → expect it CANNOT do chocolate-in-basket
  AND CANNOT do can-in-basket.
This is the crucial dissociation: object-specific features should be swappable while the
shared action primitive is preserved, and removing the action primitive should break the
whole family regardless of object.

## Deliverables
- The co-similarity histograms + summary stats.
- A table of candidate features labeled object / action / task-label with evidence.
- Steering + ablation results (behavioral outcomes) for the chocolate/can/basket cases.
- A short write-up of whether the compositionality hypothesis holds.

Start by locating the relevant feature IDs (chocolate, can, "put in basket") from the
existing enrichment/interpretation outputs before running any steering. Verify file/script
paths against the current tree — some notes may be out of date.
