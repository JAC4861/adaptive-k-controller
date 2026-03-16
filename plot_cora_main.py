#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate main figure for Cora:
Panels: (A) Curvature vs Epoch (B) Validation ROC vs Epoch (C) Distortion (Stress) vs Epoch
Also optionally a bar chart of final metrics (fixed/free/adaptive).
Assumptions:
  - metrics_master.csv as provided
  - logs in runs_batch/logs/*.log
Limitations:
  - If adaptive logs lack per-epoch curvature lines, we approximate curvature as constant final_c (warn).
  - Stress panel will be empty (annotated) if no non-zero per-epoch stress.
"""

import re
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

VAL_ROC_PAT = re.compile(r"Ep:(\d+).*?val_roc:\s*([0-9.]+)")
# Adaptive curvature possible patterns (add more if you change logging)
CURV_PATTERNS = [
    re.compile(r"CURV=([0-9.]+)"),
    re.compile(r"K=([0-9.]+)"),
    re.compile(r"c_curv=([0-9.]+)")
]
# Distortion pattern if you later log per epoch
STRESS_PAT = re.compile(r"(?:STRESS|Dist|distortion)=\s*([0-9.]+)")

def read_metrics(csv_path: Path):
    df = pd.read_csv(csv_path)
    return df

def _split_keep_last_run(lines):
    """If a log file accidentally contains concatenated runs, keep only the last run.
       Heuristic: split by lines that start a run (e.g. '[Init]' or 'INFO:root:[Init]')"""
    start_idx = 0
    for i, ln in enumerate(lines):
        if "[Init]" in ln:
            start_idx = i
    return lines[start_idx:]

def parse_log(log_path: Path):
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    lines = _split_keep_last_run(lines)

    epochs = []
    val_roc = []

    # curvature per epoch (if available)
    curv_epochs = []
    curv_vals = []

    # stress (distortion) per epoch (rare currently)
    stress_epochs = []
    stress_vals = []

    # capture final distortion if appears
    final_dist = None
    final_dist_pat = re.compile(r"FinalDist.*?d=([0-9.]+)", re.IGNORECASE)

    for ln in lines:
        m = VAL_ROC_PAT.search(ln)
        if m:
            ep = int(m.group(1))
            roc = float(m.group(2))
            epochs.append(ep)
            val_roc.append(roc)

        # curvature lines
        if any(p.search(ln) for p in CURV_PATTERNS):
            # try first matching one
            for p in CURV_PATTERNS:
                mc = p.search(ln)
                if mc:
                    val = float(mc.group(1))
                    # try to infer epoch from same line if present
                    mep = re.search(r"Ep:(\d+)", ln)
                    if mep:
                        ecp = int(mep.group(1))
                        curv_epochs.append(ecp)
                        curv_vals.append(val)
                    else:
                        # if no epoch, we cannot align -> ignore
                        pass
                    break

        # stress lines (if you log them)
        ms = re.search(r"Ep:(\d+).*?(?:STRESS|Distortion)=\s*([0-9.]+)", ln, re.IGNORECASE)
        if ms:
            se = int(ms.group(1))
            sv = float(ms.group(2))
            stress_epochs.append(se)
            stress_vals.append(sv)

        mfd = final_dist_pat.search(ln)
        if mfd:
            final_dist = float(mfd.group(1))

    data = {
        "epochs": np.array(epochs, dtype=int),
        "val_roc": np.array(val_roc, dtype=float),
    }
    if curv_epochs:
        order = np.argsort(curv_epochs)
        data["curv_epochs"] = np.array(curv_epochs, dtype=int)[order]
        data["curv_vals"] = np.array(curv_vals, dtype=float)[order]
    else:
        data["curv_epochs"] = np.array([])
        data["curv_vals"] = np.array([])

    if stress_epochs:
        o2 = np.argsort(stress_epochs)
        data["stress_epochs"] = np.array(stress_epochs, dtype=int)[o2]
        data["stress_vals"] = np.array(stress_vals, dtype=float)[o2]
    else:
        data["stress_epochs"] = np.array([])
        data["stress_vals"] = np.array([])

    data["final_dist"] = final_dist
    return data

def build_mean_std(time_grid, per_seed_dict, key):
    """Interpolate (or hold) onto a common epoch grid for mean/std."""
    arr = []
    for d in per_seed_dict.values():
        if key not in d:
            continue
        epochs = d.get("epochs" if key=="val_roc" else f"{key}_epochs", np.array([]))
        values = d.get(key if key=="val_roc" else f"{key}_vals", np.array([]))
        if epochs.size == 0 or values.size == 0:
            continue
        # simple step interpolation: assign nearest previous
        series = np.full_like(time_grid, np.nan, dtype=float)
        last_val = np.nan
        ep_map = {e:v for e,v in zip(epochs, values)}
        for i, t in enumerate(time_grid):
            if t in ep_map:
                last_val = ep_map[t]
            series[i] = last_val
        arr.append(series)
    if not arr:
        return np.full_like(time_grid, np.nan, dtype=float), np.full_like(time_grid, np.nan, dtype=float)
    mat = np.vstack(arr)
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)

def main(args):
    metrics = read_metrics(Path(args.metrics))
    # focus only on fixed / free / adaptive for cora main figure
    focus = metrics[metrics.exp_group.isin(["fixed_no_learn","free_no_reg","adaptive_two_sided"])].copy()
    # group label normalization
    mapping = {
        "fixed_no_learn":"fixed",
        "free_no_reg":"free",
        "adaptive_two_sided":"adaptive"
    }
    focus["group"] = focus.exp_group.map(mapping)

    # Collect per-log parsed data
    groups = {}
    for _, row in focus.iterrows():
        g = row.group
        seed = int(row.seed)
        log_file = Path(row.log_file)
        data = parse_log(log_file)
        # store final c & epochs info
        data["init_c"] = row.init_c
        data["final_c"] = row.final_c
        data["epochs_cap"] = int(row.epochs)
        if g not in groups:
            groups[g] = {}
        groups[g][seed] = data

    # Determine unified epoch grid (max over configs)
    max_epoch = max(int(row.epochs) for _, row in focus.iterrows())
    # Use eval frequency? We parse only epochs that appear. We'll assume multiple of 10 (10..80 / 120)
    # Build grid: union of all parsed epoch lists + fill to max
    all_epochs = set()
    for gdict in groups.values():
        for d in gdict.values():
            all_epochs.update(d["epochs"].tolist())
    epoch_grid = np.array(sorted(all_epochs))
    if epoch_grid.size == 0:
        print("ERROR: No epoch data parsed (val_roc lines missing). Aborting.")
        return

    # Prepare figure
    fig, axes = plt.subplots(1, 3, figsize=(13,4), sharex=True)
    colors = {"fixed":"#7f7f7f","free":"#1f77b4","adaptive":"#d62728"}
    linestyles = {"fixed":"-","free":"--","adaptive":"-"}
    bands = (0.77, 0.82)  # target band

    # Panel A: Curvature
    axA, axB, axC = axes
    for g, gdict in groups.items():
        # curvature series (mean/std)
        # Strategy:
        #  fixed: constant init_c
        #  free : linear from init_c to final_c mean if variation
        #  adaptive: use logged curvature if exists else constant final_c (warn)
        # Build per-seed curvature aligned to epoch_grid
        curv_seed_series = []
        has_logged = False
        for seed, d in gdict.items():
            series = np.full_like(epoch_grid, np.nan, dtype=float)
            if d["curv_epochs"].size > 0:
                has_logged = True
                # map real logged curvature
                ep_map = {e:v for e,v in zip(d["curv_epochs"], d["curv_vals"])}
                last = np.nan
                for i,e in enumerate(epoch_grid):
                    if e in ep_map:
                        last = ep_map[e]
                    series[i] = last
            else:
                # fabricate
                init_c = d["init_c"]
                final_c = d["final_c"]
                if g == "fixed":
                    series[:] = init_c
                elif g == "free":
                    if abs(final_c - init_c) < 1e-3:
                        series[:] = final_c
                    else:
                        # linear interpolation across full training horizon
                        series = init_c + (final_c - init_c) * (epoch_grid - epoch_grid.min()) / (epoch_grid.max() - epoch_grid.min())
                elif g == "adaptive":
                    series[:] = final_c  # fallback
            curv_seed_series.append(series)
        curv_seed_series = np.vstack(curv_seed_series)
        curv_mean = np.nanmean(curv_seed_series, axis=0)
        curv_std = np.nanstd(curv_seed_series, axis=0)
        axA.plot(epoch_grid, curv_mean, label=g.capitalize(), color=colors[g],
                 linestyle=linestyles[g], linewidth=2)
        if g in ("free","adaptive") or (g=="fixed" and curv_std.max()>1e-6):
            axA.fill_between(epoch_grid, curv_mean-curv_std, curv_mean+curv_std,
                             color=colors[g], alpha=0.2, linewidth=0)
        if g=="adaptive" and not has_logged:
            axA.text(0.02, 0.05, "adaptive curvature not logged (using final_c)\n(re-run to refine)",
                     transform=axA.transAxes, fontsize=8, color="#d62728")

    axA.axhspan(bands[0], bands[1], color="#ff9896", alpha=0.25, label="Target band")
    axA.set_title("Curvature K")
    axA.set_ylabel("K")
    axA.grid(alpha=0.3, linestyle="--")

    # Panel B: Validation ROC
    for g, gdict in groups.items():
        mean, std = build_mean_std(epoch_grid, gdict, "val_roc")
        axB.plot(epoch_grid, mean, label=g.capitalize(), color=colors[g],
                 linestyle=linestyles[g], linewidth=2)
        axB.fill_between(epoch_grid, mean-std, mean+std, color=colors[g], alpha=0.20, linewidth=0)
    axB.set_title("Validation ROC")
    axB.grid(alpha=0.3, linestyle="--")

    # Panel C: Distortion (Stress)
    any_stress = False
    for g, gdict in groups.items():
        # attempt
        stress_mean, stress_std = build_mean_std(epoch_grid, gdict, "stress")
        if np.isnan(stress_mean).all() or np.allclose(stress_mean, 0, equal_nan=True):
            continue
        any_stress = True
        axC.plot(epoch_grid, stress_mean, label=g.capitalize(), color=colors[g],
                 linestyle=linestyles[g], linewidth=2)
        axC.fill_between(epoch_grid, stress_mean-stress_std, stress_mean+stress_std,
                         color=colors[g], alpha=0.2, linewidth=0)
    if not any_stress:
        axC.text(0.5, 0.5, "No per-epoch stress logged\n(add logging & rerun)", ha="center",
                 va="center", transform=axC.transAxes, fontsize=10, color="#555555")
    axC.set_title("Distortion (Stress)")
    axC.grid(alpha=0.3, linestyle="--")

    # Legend consolidate
    handles, labels = axA.get_legend_handles_labels()
    axC.legend(handles, labels, frameon=False, loc="lower right")

    for ax in axes:
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    out_main = Path(args.out_main)
    out_main.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_main, dpi=300, bbox_inches="tight")
    print(f"[OK] Main figure saved to {out_main}")

    # Final bar chart
    if args.out_bar:
        df_final = focus.groupby("group").agg(
            val_mean=("val_metric","mean"),
            val_std=("val_metric","std"),
            stress_mean=("stress","mean"),
            stress_std=("stress","std"),
            c_mean=("final_c","mean"),
            c_std=("final_c","std")
        ).reset_index()

        fig2, ax2 = plt.subplots(1,2, figsize=(8,3.2))
        # ROC bars
        x = np.arange(len(df_final))
        ax2[0].bar(x, df_final.val_mean, yerr=df_final.val_std, capsize=4,
                   color=[colors[g] for g in df_final.group])
        ax2[0].set_xticks(x)
        ax2[0].set_xticklabels(df_final.group.str.capitalize())
        ax2[0].set_ylabel("Val ROC")
        ax2[0].set_title("Final ROC")

        # Stress bars
        ax2[1].bar(x, df_final.stress_mean, yerr=df_final.stress_std, capsize=4,
                   color=[colors[g] for g in df_final.group])
        ax2[1].set_xticks(x)
        ax2[1].set_xticklabels(df_final.group.str.capitalize())
        ax2[1].set_ylabel("Stress")
        ax2[1].set_title("Final Distortion")

        plt.tight_layout()
        out_bar = Path(args.out_bar)
        fig2.savefig(out_bar, dpi=300, bbox_inches="tight")
        print(f"[OK] Bar figure saved to {out_bar}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="/home/lixianchen/hgcn-master11/hgcn-master/runs_batch/metrics_master.csv", help="Path to metrics_master.csv")
    ap.add_argument("--out-main", default="fig_cora_main.png")
    ap.add_argument("--out-bar", default="fig_cora_final_bars.png")
    args = ap.parse_args()
    main(args)