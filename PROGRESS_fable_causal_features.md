# Progress — PI05 SAE causal feature experiments (LIBERO)

Updated: 2026-07-06 (Claude). Task spec: `PROMPT_fable_causal_features.md`.
Status: **Phases A and B complete. PAUSED for SAE-expert consult before next interventions.**

---

## TL;DR

- **Phase A (read-only):** the "one feature per task" story at lang_layer_12 dissolves into a clean
  compositional decomposition — **object-word features ⊕ action-template features ⊕ visual object
  detectors**. True task-label (combination) features are rare and weak. Same-task features are
  near-orthogonal in decoder space (no task "direction").
- **Checkpoint forensics:** the PyTorch checkpoint everything was built on turned out to be
  **π0.5-BASE, not the LIBERO-finetuned policy** (all 201 action-expert tensors differ from a fresh
  conversion of the official pi05_libero; trunk differs by a finetune delta). This *strengthens*
  Phase A — a never-LIBERO-finetuned trunk can't be memorizing LIBERO tasks — and it was fixed by
  re-converting the official weights (`pi05_pytorch_v2`, open-loop R² = 0.962 vs expert actions,
  per-dim corr 0.96–0.996; old checkpoint scored R² = −0.54).
- **SAE transfer gate:** the base-trained SAE transfers to the finetuned trunk — 13/14 selected
  features keep their variance labels (activations attenuate to ~60–70%). Feature 2348 blurred and
  was dropped.
- **Phase B (causal, mini: 7 conditions × 5 trials):** baselines healthy (5/5 chocolate, 4/5 tomato)
  — **but every feature intervention had zero behavioral effect** (object ablation 5/5, shared-action
  ablation 5/5 on both tasks, feature swap 5/5; gripper–object approach distances match baseline to
  the millimeter). Ablation math verified exact (re-encode → 0). A clean, interpretable negative.
- **Leading interpretation:** π0.5's action expert cross-attends to the trunk's **KV cache at all 18
  layers**; a layer-12 edit leaves the concept readable in layers 0–12. Single-layer feature ablation
  may be structurally leaky in this architecture. Alternatives: feature redundancy (we removed the top
  4–7 of dozens per concept; SAE runs ~35% attenuated on this trunk), or the features are a readout
  rather than a cause.

## Deliverables (all self-contained, shareable HTML)

| File (in `sae_interpretations/visualizations/`) | Contents |
|---|---|
| `phase_a_explorer.html` | Interactive Phase A: provenance banner (base-model discovery + why it strengthens the result), A1 prompt-grid scatter + per-feature 6×6 heatmaps under both frozen-image sets, frozen-images explainer, A2 co-similarity histograms with means + separation stats (d=0.17/AUC=0.548 layer 12; none at layer 9), top-14 selection criteria (concept → purity varO/varT ≥ 0.85 → strength/replication) and the base→finetuned transfer table (13/14 HOLD), full label table. |
| `phase_b_results.html` (= `phase_c_results.html`, kept in sync) | Phase B causal results: dissociation matrix with pre-registered predictions and ✓/✗ explained (grades the hypothesis, not the robot), exact prompts + token-position spec (text tokens 768+, image patches 0–767, every-replan application), per-condition cards with feature chips + approach distances, interpretation box (3 candidate explanations), and 5 open questions for the SAE expert. |

Other artifacts: `feature_labels_final.json` (label table + evidence),
`prompt_variation{,_v2}/` grids, `cosimilarity/` stats+plots, `steering_exp1_mini/`
(results JSON + videos), `PROMPT_fable_causal_features.md` (original spec).

## Key numbers

| Measurement | Value |
|---|---|
| Enriched features analysed (of 8192) | 500 — 397 image-token / 103 text-token |
| Chocolate object features (lang) | 6004 (varO .99), 4935, 4480, 3154 |
| Tomato object features (lang) | 3023 (varO .98–1.00), 4033 |
| Action features (pick-place-basket) | 6338, 2732, 2357, 4215, 7150 (varT .96–1.00); 3086 (put-basket); 612/3984 (push-right) |
| Visual chocolate / tomato features | 815, 7897, 4402 / 1919, 2029, 4739 (σ=0 across 37 prompts) |
| A2 same-task decoder cosine | μ=.022 vs global .012; d=.17, AUC=.548 (layer 12) |
| Old ckpt open-loop vs expert | R² = −0.54 (dim-2 corr −0.63) → diagnosed wrong training state |
| New ckpt (`pi05_pytorch_v2`) open-loop | R² = 0.962; per-dim corr .96–.996 |
| SAE transfer gate | 13/14 labels hold; 2348 dropped; acts ~60–70% of base |
| Phase B mini (job 10812923, 6.6 min H100) | baselines 5/5, 4/5; ALL interventions no effect (5/5 everywhere) |

## Infrastructure built (reusable)

- `src/openpi/interpretability/steering.py` — SteeringController: forward hooks on
  `paligemma.language_model.layers[i]` (inference uses the prefix-only HF path); modes
  `ablate` (h −= a_f(h)·d_f, exact) and `add` (h += α·d_f); positions text/image/all; hit counter.
- `serve_policy_steered.py` — websocket policy server, hot-reloads steering spec from a control
  file (one model load per sweep); `--checkpoint_dir` selectable.
- `examples/libero/steering_client.py` — LIBERO rollout client (py3.8 venv at
  `examples/libero/.venv`), per-condition JSON results, videos, **object-approach metric**
  (min gripper–object distance per episode + closest_object).
- `run_steering_conditions.py` + `pace_ice_steering.sbatch` — condition sweep driver (server once,
  conditions via control-file swap; EGL render smoke test first).
- Analysis: `prompt_variation_features.py` (6×6 grid + two-way variance decomposition),
  `feature_token_analysis.py` (exact token/patch attribution; **plain Pi0 prompt format —
  `discrete_state_input=False`, no Task:/State: scaffold**), `decoder_cosimilarity.py`,
  `test_openloop_actions.py` (checkpoint health), `test_joint_vs_cached.py` (inference-path A/B),
  `diff_checkpoints.py` (grouped tensor diff), `build_viz_data.py` / `build_phasec_viz_data.py`.

## Built and PARKED (not submitted, awaiting post-consult decision)

1. **Cross-layer ablation** (recommended next): project concept features out of every layer's KV /
   multi-layer SAE ablation — the decisive test between "architectural leak" and "readout".
   (Design discussed; not yet implemented.)
2. **Base-model approach-steering** — `experiments/steering_base_approach.json`: steer the base
   checkpoint (exact SAE match) on the chocolate env, metric = object approach, 5 conditions ready.
3. **ALOHA dictionary-stretch test** — FVU gate on ALOHA frames through the base trunk → action-verb
   feature firing with verb-matched prompts → visual-feature silence check. (Adapter to build.
   Object features won't transfer by construction — LIBERO-vocabulary dictionary.)



## Gotchas log (for future sessions)

- `pi05_libero` config: `discrete_state_input=False` → plain `<bos> text \n` prompts; and its
  `weight_loader` points at **pi05_base** (that's how the wrong checkpoint likely happened).
- Lang-layer forward hooks work at inference (prefix-only HF path) but NOT in the training joint path.
- Login node: ~a few GB memory cap (OOM-killed model loads), no GL; data loaders need
  `if __name__ == "__main__"` guard and `JAX_PLATFORMS=cpu` on CPU nodes.
- Home quota full → uv/python installs under project storage (`tools/`); scratchpad /tmp is wiped
  between sessions — keep templates + data in project storage.
- LIBERO client venv: py3.8 at `examples/libero/.venv`; `third_party/libero` needed a top-level
  `libero/__init__.py` touch + reinstall; MUJOCO_GL=egl on GPU nodes.

## SLURM ledger (all jobs this project, 2026-07-01 → 07-06)

| Job | What | Outcome |
|---|---|---|
| 10661830 | A1 prompt grid (v1) | failed fast — import path |
| 10662294 | A1 prompt grid (fixed), 2 image bases | ✅ 2.5 min/array |
| 10662551 | Phase B full (12 cond × 10) on OLD ckpt | ran fine; 0/10 everywhere → exposed bad ckpt |
| 10687749/10687872/10805047 | open-loop diagnostics (CPU) | ✅ after JAX_PLATFORMS + __main__ fixes: R²=−0.54 |
| 10805105 | joint-vs-cached inference A/B (CPU) | ✅ paths identical → ckpt at fault |
| 10805241 | re-convert official pi05_libero (CPU) | ✅ 1:46 |
| 10807212 | v2 diff + open-loop (CPU) | ✅ R²=0.962 |
| 10807230 | SAE-transfer gate on v2 (GPU 2:12) | ✅ 13/14 hold |
| 10812923 | **Phase B mini (7 cond × 5) on v2** | ✅ 6:36 — baselines healthy, interventions null |
