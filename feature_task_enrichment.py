"""
feature_task_enrichment.py
==========================
For each SAE feature, look up which LIBERO tasks its top-activating examples
came from, then compute task enrichment scores.

This is the causal interpretability proof: if feature X fires 90%+ of the time
on "put the cream cheese..." tasks, that feature is specifically detecting
cream cheese — not just correlated.

Uses a hypergeometric test (like gene ontology enrichment): given that task T
makes up X% of all frames, what's the probability of seeing Y% of a feature's
top examples come from task T by chance?

Output:
  feature_task_enrichment.json    per-feature top tasks + p-values
  feature_task_enrichment.html    sortable HTML table

Usage:
    python feature_task_enrichment.py \\
        --interp_dir /path/to/sae_interpretations/pi05_libero_lang_layer_9_k32_exp4 \\
        --top_n_features 200    # how many features to analyse (ranked by peak activation)
"""

import argparse
import json
from pathlib import Path
from collections import Counter
from scipy.stats import hypergeom
import numpy as np
import pyarrow.parquet as pq

LIBERO_CACHE = Path("/storage/project/r-agarg35-0/rvaldes6/.cache/huggingface/lerobot/physical-intelligence/libero")
NORM_STATS_PATH = Path("/storage/project/r-agarg35-0/rvaldes6/openpi_clone/assets/pi05_libero/physical-intelligence/libero/norm_stats.json")
MATCH_DECIMALS = 4


def load_tasks():
    tasks = {}
    with open(LIBERO_CACHE / "meta" / "tasks.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            tasks[obj["task_index"]] = obj["task"]
    return tasks


def load_quantile_stats():
    with open(NORM_STATS_PATH) as f:
        data = json.load(f)
    stats = data["norm_stats"]["state"]
    return np.array(stats["q01"], dtype=np.float32), np.array(stats["q99"], dtype=np.float32)


def quantile_normalize(state, q01, q99):
    return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def state_key(norm_state):
    return tuple(np.round(np.array(norm_state[:8], dtype=np.float64), MATCH_DECIMALS).tolist())


def build_parquet_lookup(q01, q99):
    """Build state_key -> task_index lookup."""
    lookup = {}
    task_frame_counts = Counter()
    data_dir = LIBERO_CACHE / "data"
    total = 0
    for chunk_dir in sorted(data_dir.iterdir()):
        for pf in sorted(chunk_dir.glob("episode_*.parquet")):
            table = pq.read_table(pf, columns=["state", "task_index"])
            states = np.array(table["state"].to_pylist(), dtype=np.float32)
            tk_idx = table["task_index"].to_pylist()
            norm_states = quantile_normalize(states, q01, q99)
            for i in range(len(table)):
                k = state_key(norm_states[i])
                lookup[k] = int(tk_idx[i])
                task_frame_counts[int(tk_idx[i])] += 1
            total += len(table)
    print(f"Built lookup: {len(lookup)} unique states from {total} frames", flush=True)
    return lookup, task_frame_counts, total


def read_all_features(json_path, feature_ids):
    """Stream JSON and return {feat_id: [examples]} for requested features."""
    results = {}
    target_keys = {f'"feature_{i}"' for i in feature_ids}
    collecting = False
    current_id = None
    depth = 0
    buf = []

    with open(json_path) as fh:
        for line in fh:
            s = line.strip()
            if not collecting:
                for tk in list(target_keys):
                    if tk in s:
                        current_id = int(tk.strip('"').split("_")[1])
                        collecting = True
                        bp = s.find("[")
                        if bp != -1:
                            rest = s[bp:]
                            buf = [rest]
                            depth = rest.count("[") - rest.count("]")
                        break
            else:
                buf.append(line)
                depth += line.count("[") - line.count("]")
                if depth <= 0:
                    raw = "".join(buf).strip().rstrip(",")
                    try:
                        results[current_id] = json.loads(raw)
                    except Exception:
                        pass
                    collecting = False
                    buf = []
                    depth = 0
                    target_keys.discard(f'"feature_{current_id}"')
                    if not target_keys:
                        break
    return results


def enrich(task_counts_in_feature, total_in_feature, task_frame_counts, total_frames):
    """
    For each task, compute hypergeometric p-value:
      population = total_frames
      successes in population = task_frame_counts[task]
      draws = total_in_feature (top examples)
      successes in draws = task_counts_in_feature[task]
    """
    results = []
    for task_idx, count in task_counts_in_feature.most_common():
        K = task_frame_counts[task_idx]   # successes in population
        n = total_in_feature              # draws
        k = count                         # successes in draws
        N = total_frames                  # population
        # P(X >= k) = survival function of hypergeometric
        pval = hypergeom.sf(k - 1, N, K, n)
        frac_feature = count / total_in_feature
        frac_baseline = K / N
        fold_enrichment = frac_feature / max(frac_baseline, 1e-9)
        results.append({
            "task_index": task_idx,
            "count": count,
            "frac_feature": frac_feature,
            "frac_baseline": frac_baseline,
            "fold_enrichment": fold_enrichment,
            "pval": pval,
        })
    results.sort(key=lambda x: x["pval"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interp_dir", required=True)
    parser.add_argument("--top_n_features", type=int, default=200)
    parser.add_argument("--min_examples", type=int, default=5,
                        help="Min matched examples needed to run enrichment for a feature")
    args = parser.parse_args()

    interp_dir = Path(args.interp_dir)

    print("Loading tasks...", flush=True)
    tasks = load_tasks()
    print(f"  {len(tasks)} tasks", flush=True)

    print("Loading norm stats...", flush=True)
    q01, q99 = load_quantile_stats()

    print("Building parquet lookup...", flush=True)
    lookup, task_frame_counts, total_frames = build_parquet_lookup(q01, q99)

    # Pick top features by peak activation
    import torch
    top_vals = torch.load(interp_dir / "top_activation_values.pt")  # [dict_size, top_k]
    peak = top_vals[:, 0]
    top_feature_ids = peak.argsort(descending=True)[:args.top_n_features].tolist()
    print(f"\nAnalysing top {args.top_n_features} features (by peak activation)", flush=True)

    # Stream JSON
    print("Streaming top_activating_examples.json...", flush=True)
    feature_data = read_all_features(interp_dir / "top_activating_examples.json", set(top_feature_ids))
    print(f"  Loaded {len(feature_data)} features", flush=True)

    # For each feature: map examples to tasks, run enrichment
    feature_enrichment = []
    n_matched_total = 0
    n_total = 0

    for feat_id in top_feature_ids:
        if feat_id not in feature_data:
            continue
        examples = feature_data[feat_id]
        task_counts = Counter()
        matched = 0

        for ex in examples:
            sp = ex.get("proprio_state", [])
            if not sp:
                continue
            k = state_key(sp[:8])
            if k in lookup:
                task_counts[lookup[k]] += 1
                matched += 1

        n_matched_total += matched
        n_total += len(examples)

        if matched < args.min_examples:
            continue

        enrichment = enrich(task_counts, matched, task_frame_counts, total_frames)

        top_task = enrichment[0] if enrichment else None
        feature_enrichment.append({
            "feature_id": feat_id,
            "peak_activation": float(peak[feat_id]),
            "n_examples": len(examples),
            "n_matched": matched,
            "top_task": {
                "task_index": top_task["task_index"],
                "task_name": tasks.get(top_task["task_index"], "?"),
                "count": top_task["count"],
                "frac_feature": top_task["frac_feature"],
                "frac_baseline": top_task["frac_baseline"],
                "fold_enrichment": top_task["fold_enrichment"],
                "pval": top_task["pval"],
            } if top_task else None,
            "all_tasks": [
                {**e, "task_name": tasks.get(e["task_index"], "?")}
                for e in enrichment[:5]
            ],
        })

    print(f"\nMatch rate: {n_matched_total}/{n_total} ({100*n_matched_total/max(n_total,1):.1f}%)", flush=True)

    # Sort by top-task fold enrichment
    feature_enrichment.sort(key=lambda x: -(x["top_task"]["fold_enrichment"] if x["top_task"] else 0))

    # Save JSON
    out_json = interp_dir / "feature_task_enrichment.json"
    with open(out_json, "w") as f:
        json.dump(feature_enrichment, f, indent=2)
    print(f"\nSaved: {out_json}", flush=True)

    # Print top 20
    print("\n" + "="*80, flush=True)
    print(f"TOP TASK-ENRICHED FEATURES", flush=True)
    print("="*80, flush=True)
    print(f"{'Feature':>8}  {'Peak':>7}  {'Fold':>6}  {'Frac':>6}  {'Base':>6}  {'p-val':>10}  Task", flush=True)
    print("-"*80, flush=True)
    for fe in feature_enrichment[:30]:
        tt = fe["top_task"]
        if tt:
            print(
                f"  {fe['feature_id']:6d}  {fe['peak_activation']:7.1f}  "
                f"{tt['fold_enrichment']:6.1f}x  "
                f"{tt['frac_feature']:5.1%}  {tt['frac_baseline']:5.1%}  "
                f"{tt['pval']:10.2e}  {tt['task_name'][:60]}",
                flush=True
            )

    # Find cream cheese features specifically
    print("\n" + "="*80, flush=True)
    print("CREAM CHEESE FEATURES", flush=True)
    print("="*80, flush=True)
    cream_cheese_task_ids = {ti for ti, t in tasks.items() if "cream cheese" in t.lower()}
    print(f"Cream cheese tasks: {[(ti, tasks[ti]) for ti in sorted(cream_cheese_task_ids)]}", flush=True)

    cc_features = []
    for fe in feature_enrichment:
        tt = fe["top_task"]
        if tt and tt["task_index"] in cream_cheese_task_ids:
            cc_features.append(fe)
    # Also check all_tasks for cream cheese
    for fe in feature_enrichment:
        if fe in cc_features:
            continue
        for t in fe["all_tasks"]:
            if t["task_index"] in cream_cheese_task_ids and t["fold_enrichment"] > 2.0:
                cc_features.append(fe)
                break
    cc_features.sort(key=lambda x: -(x["top_task"]["fold_enrichment"] if x["top_task"] and x["top_task"]["task_index"] in cream_cheese_task_ids else 0))

    if cc_features:
        for fe in cc_features[:10]:
            tt = fe["top_task"]
            print(f"  feature_{fe['feature_id']}  peak={fe['peak_activation']:.1f}  "
                  f"top_task='{tt['task_name']}'  "
                  f"frac={tt['frac_feature']:.1%} vs baseline {tt['frac_baseline']:.1%}  "
                  f"fold={tt['fold_enrichment']:.1f}x  p={tt['pval']:.2e}", flush=True)
    else:
        print("  No strongly enriched cream cheese features found in top activations.", flush=True)
        print("  (Try increasing --top_n_features)", flush=True)

    # Write HTML table
    rows = []
    for fe in feature_enrichment:
        tt = fe["top_task"]
        if not tt:
            continue
        # Highlight cream cheese
        cc = "background:#ffe0e0;" if tt["task_index"] in cream_cheese_task_ids else ""
        rows.append(
            f"<tr style='{cc}'>"
            f"<td>{fe['feature_id']}</td>"
            f"<td>{fe['peak_activation']:.1f}</td>"
            f"<td>{tt['fold_enrichment']:.1f}x</td>"
            f"<td>{tt['frac_feature']:.1%}</td>"
            f"<td>{tt['frac_baseline']:.1%}</td>"
            f"<td>{tt['pval']:.2e}</td>"
            f"<td>{tt['task_name']}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Feature Task Enrichment — {interp_dir.name}</title>
<style>
body{{font-family:monospace;font-size:13px;margin:20px}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
th{{background:#f0f0f0;cursor:pointer}}
tr:hover{{background:#f9f9f9}}
</style>
<script>
function sortTable(n){{
  var t=document.getElementById("t"),rows,sw=true,i,x,y,b;
  rows=Array.from(t.querySelectorAll("tr:not(:first-child)"));
  rows.sort(function(a,b){{
    x=a.cells[n].textContent;y=b.cells[n].textContent;
    return isNaN(parseFloat(x))?x.localeCompare(y):parseFloat(x)-parseFloat(y);
  }});
  rows.forEach(r=>t.appendChild(r));
}}
</script>
</head><body>
<h2>Feature Task Enrichment: {interp_dir.name}</h2>
<p>{len(feature_enrichment)} features analysed. <span style="background:#ffe0e0;padding:2px 6px">Pink = cream cheese task</span></p>
<table id="t">
<tr>
  <th onclick="sortTable(0)">Feature</th>
  <th onclick="sortTable(1)">Peak Act.</th>
  <th onclick="sortTable(2)">Fold Enrich.</th>
  <th onclick="sortTable(3)">Frac (feature)</th>
  <th onclick="sortTable(4)">Frac (baseline)</th>
  <th onclick="sortTable(5)">p-value</th>
  <th onclick="sortTable(6)">Top Task</th>
</tr>
{"".join(rows)}
</table></body></html>"""

    out_html = interp_dir / "feature_task_enrichment.html"
    out_html.write_text(html)
    print(f"Saved: {out_html}", flush=True)


if __name__ == "__main__":
    main()
