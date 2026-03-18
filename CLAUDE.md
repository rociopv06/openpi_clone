# Project: PI0 SAE Interpretability on LIBERO

## Overview
Training and interpreting Sparse Autoencoders (SAEs) on PI0 (pi05), a robot manipulation policy = PaliGemma VLM + Gemma action expert. Goal: understand what features the model represents internally and how they causally influence actions.

## Cluster
- PACE ICE cluster, account: `gts-agarg35-ideasci23_dgx`
- Project dir: `/storage/project/r-agarg35-0/rvaldes6/`
- Python (venv, GPU jobs): `/storage/project/r-agarg35-0/rvaldes6/openpi_clone/.venv/bin/python`
- Python (conda, CPU jobs): `/storage/project/r-agarg35-0/rvaldes6/conda_envs/py311_openpi/bin/python`
- PI0 model: `/storage/project/r-agarg35-0/rvaldes6/pi05_pytorch/model.safetensors`
- SAE checkpoints: `/storage/project/r-agarg35-0/rvaldes6/sae_checkpoints/`
- Interpretations: `/storage/project/r-agarg35-0/rvaldes6/sae_interpretations/`
- PYTHONPATH: `$OPENPI_DIR:$OPENPI_DIR/src`

## Model Architecture
PI0 (pi05) has two transformer stacks:
- **PaliGemma prefix** (lang_layer_0–17): processes image + language tokens together. `inputs_embeds[0]` cached here.
- **Gemma action expert suffix** (expert_layer_0–17): processes state + noisy action tokens. `inputs_embeds[1]` cached here.
- **Vision projector** (vision_proj): SigLIP → language model bridge, 2048-dim, one forward hook per camera (fires 3× per forward pass for LIBERO — captures last camera).

## Token Structure (lang_layer_N prefix)
LIBERO uses 3 camera views (base + left wrist + right wrist zero-padded):
- Positions **0–767**: image tokens (3 cameras × 256 SigLIP patches each)
- Positions **768+**: language instruction tokens (max_token_len=200, padded)
- Total prefix seq_len ≈ 838 tokens
- Token type classification: `token_pos < 768` → image; `768 ≤ pos < 768+text_len` → text; else → padding
- 90% of top SAE activations are on image tokens, 10% on text tokens

## Trained SAEs
All use BatchTopKSAE, k=32, dict_size=8192, activation_dim=2048:

| Exp name | Hook point | Status |
|---|---|---|
| pi05_libero_lang_layer_3_k32_exp4 | lang_layer_3 | trained + interpreted |
| pi05_libero_lang_layer_6_k32_exp4 | lang_layer_6 | trained + interpreted |
| pi05_libero_lang_layer_9_k32_exp4 | lang_layer_9 | trained + interpreted |
| pi05_libero_lang_layer_12_k32_exp4 | lang_layer_12 | trained + interpreted |
| pi05_libero_lang_layer_17_k32_exp4 | lang_layer_17 | trained + interpreted |
| pi05_libero_expert_layer_0_k32_exp4 | expert_layer_0 | trained + interpreted |
| pi05_libero_vision_proj_k32_exp4 | vision_proj | trained + interpreted |

## Dataset: LIBERO (PI0 subset)
- 40 tasks (curated subset of LIBERO benchmark, NOT full 130 tasks), 1693 episodes, 273,465 frames
- Cache: `/storage/project/r-agarg35-0/rvaldes6/.cache/huggingface/lerobot/physical-intelligence/libero/`
- 2 real cameras in parquet (`image`, `wrist_image`); model expects 3 (right wrist zero-padded)
- Normalization: quantile norm `(x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0`
- Image matching uses 8-dim normalized proprio fingerprint (round to 4 decimal places)

## Key Scripts
- `interpret_sae.py` — collect top-k activating examples per SAE feature. **Fixed bug**: `token_idx // seq_len` (was `token_idx % batch_size`). Saves `token_pos` + `token_type`.
- `visualize_top_examples.py` — generate HTML visualizations (top-30 features × top-5 examples with images)
- `feature_task_enrichment.py` — hypergeometric task enrichment analysis per feature (proves semantic specificity)
- `feature_attribution_layer9.py` — GxA (gradient × activation) attribution. **OOM fix**: patches `embed_image` to run with `torch.no_grad()` so SigLIP is detached from gradient graph.
- `jacobian_feature_attribution.py` — Jacobian Scopes attribution (arxiv 2601.16407): `||∂(action_output)/∂h · d_i||₂` per feature, direct causal influence on robot actions.
- `compute_ms_and_label.py` — monosemanticity scores (DINOv2)
- `run_local_labels.py` — feature labeling with Llama-3.2-11B-Vision-Instruct (no Anthropic API needed)

## Key Findings
- **Metadata bug (fixed)**: old interpretations had wrong images because `token_idx % batch_size` assigned wrong proprio to most tokens. Only 1/50 tokens had correct metadata.
- **Feature 3718 (lang_layer_9, image token)**: fires 100% on image tokens, 92% on "put cream cheese in bowl", 60x enriched (p~10⁻⁷⁹) — genuine visual feature
- **Feature 2643 (lang_layer_9, text token)**: fires at fixed text position 773, 100% cream cheese bowl task — language feature detecting the word "cream cheese" in the instruction
- **Vision SAE cross-similarity**: decoder directions at vision_proj vs lang_layer_9 have mean max cosine ~0.085 — nearly orthogonal subspaces, Gemma layers do substantial representational reorganization
- **Task specificity**: model has distinct features for "cream cheese in bowl" vs "cream cheese in basket" vs "cream cheese + alphabet soup in basket"

## SLURM Jobs
- `pace_ice_reinterpret_all.sbatch` — array 0–5, all 6 lang/expert layers, n_batches=1000, top_k=50, mem=200G
- `pace_ice_reinterp_l9_fast.sbatch` — lang_layer_9 only, n_batches=300, 1.5h
- `pace_ice_interpret_vision.sbatch` — vision_proj interpretation
- `pace_ice_sae_vision.sbatch` — train vision SAE
- `pace_ice_attribution_all.sbatch` — GxA attribution array job
- `pace_ice_viz.sbatch` — regenerate HTML visualizations (CPU, conda env)

## Common Pitfalls
- `mem-per-gpu=80G` is NOT enough for interpret jobs at n_batches>200 — use `--mem=200G` instead (sample_metadata dict grows large)
- expert_layer_18 does NOT exist (model only has 0–17)
- Attribution OOM: SigLIP builds a huge gradient graph — always use the `embed_image` no-grad patch in attribution scripts
- WandB fails on cluster (no API key configured) but training still completes — ignore the wandb error
- Use venv for GPU jobs, conda env for CPU-only scripts (pandas/pyarrow available in conda)
- Do not use Chinese models (e.g. Qwen2-VL) for labeling
