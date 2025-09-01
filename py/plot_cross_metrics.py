# %%
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# %%


def plot_cross_metrics(log_dir: Path | str = "logs", metric: str = "v_roc_auc"):
    log_dir = str(log_dir)
    train_pat = re.compile(
        r"Epoch (\d+).*?train_losses: \[([\d.eE+-]+)\].*?t_roc_auc': ([\d.eE+-]+), 't_pr_auc': ([\d.eE+-]+)"
    )
    val_pat = re.compile(
        r"Epoch (\d+).*?val_losses: \[([\d.eE+-]+)\].*?v_roc_auc': ([\d.eE+-]+), 'v_pr_auc': ([\d.eE+-]+)"
    )
    rows = []

    for log_file in glob.glob(f"{log_dir}/**/*.out", recursive=True):
        run_name = log_file.split("/")[-1].replace(".out", "")
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
                })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No metrics found.")
        return

    plt.figure(figsize=(10, 6))
    for run, subdf in df.groupby("run"):
        plt.plot(subdf["epoch"], subdf[metric], marker="o", label=run)

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} across runs")
    plt.legend()
    plt.tight_layout()
    plt.show()
