# %%
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

import json


def normalize_val(v):
    if isinstance(v, str):
        v = v.strip()
        v_list = v[1:-1]
        if "(" in v_list and v_list.endswith(")"):  # class-like repr
            return v_list.split("(")[0]
        if "(" in v and v.endswith(")"):  # class-like repr
            return v.split("(")[0]
        return v
    return v

# %%


def collect_metrics(
    log_dir: Path | str = "logs",
    label_params: List[str] | None = None,
    ignore_runs: List[str] | None = None,
):
    log_dir = Path(log_dir)
    train_pat = re.compile(
        r"Epoch (\d+).*?train_losses: \[([\d.eE+-]+)\].*?t_roc_auc': ([\d.eE+-]+), 't_pr_auc': ([\d.eE+-]+)"
    )
    val_pat = re.compile(
        r"Epoch (\d+).*?val_losses: \[([\d.eE+-]+)\].*?v_roc_auc': ([\d.eE+-]+), 'v_pr_auc': ([\d.eE+-]+)"
    )
    rows = []

    runs_labels = {}

    log_file = glob.glob(f"{log_dir}/**/*.out", recursive=True)[0]

    for log_file in glob.glob(f"{log_dir}/**/*.out", recursive=True):
        # run_name = log_file.split("/")[-1].replace(".out", "")
        log_file = Path(log_file)
        run_name = log_file.parent.name
        if ignore_runs:
            if run_name in ignore_runs:
                continue

        run_labels = []
        if label_params:
            run_params_file = log_file.parent / "params.json"
            with open(run_params_file) as f:
                run_params = json.load(f)
                runs_labels[run_name] = []
                for param in label_params:
                    runs_labels[run_name].append(
                        {param: normalize_val(run_params[param])})
                    run_labels.append(
                        {param: normalize_val(run_params[param])})

        epochs, train_loss, t_roc, t_pr, val_loss, v_roc, v_pr = [], [], [], [], [], [], []
        with open(log_file) as f:
            for line in f:
                m1 = train_pat.search(line)
                if m1:
                    epochs.append(int(m1.group(1)))
                    train_loss.append(float(m1.group(2)))
                    t_roc.append(float(m1.group(3)))
                    t_pr.append(float(m1.group(4)))
                    continue
                m2 = val_pat.search(line)
                if m2:
                    val_loss.append(float(m2.group(2)))
                    v_roc.append(float(m2.group(3)))
                    v_pr.append(float(m2.group(4)))
        if epochs:
            for i in range(len(epochs)):
                rows.append({
                    "run": run_name,
                    "epoch": epochs[i],
                    "train_loss": train_loss[i],
                    "t_roc_auc": t_roc[i],
                    "t_pr_auc": t_pr[i],
                    "val_loss": val_loss[i],
                    "v_roc_auc": v_roc[i],
                    "v_pr_auc": v_pr[i],
                    "run_labels": str(run_labels),
                })

    df = pd.DataFrame(rows)

    if df.empty:
        raise Exception("No metrics found")

    return df, runs_labels
