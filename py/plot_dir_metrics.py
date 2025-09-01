# %%
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# %%


def plot_dir_metrics(log_dir="logs"):
    train_pat = re.compile(
        r"Epoch (\d+).*?t_roc_auc': ([\d.eE+-]+).*?'t_pr_auc': ([\d.eE+-]+)")
    val_pat = re.compile(
        r"Epoch (\d+).*?v_roc_auc': ([\d.eE+-]+).*?'v_pr_auc': ([\d.eE+-]+)")
    data = []

    for log_file in glob.glob(f"{log_dir}/*.txt"):
        run_name = log_file.split("/")[-1].replace(".txt", "")
        epochs, v_roc, v_pr = [], [], []
        with open(log_file) as f:
            for line in f:
                m = val_pat.search(line)
                if m:
                    epochs.append(int(m.group(1)))
                    v_roc.append(float(m.group(2)))
                    v_pr.append(float(m.group(3)))
        if epochs:
            data.append({
                "run": run_name,
                "final_epoch": epochs[-1],
                "final_v_roc": v_roc[-1],
                "final_v_pr": v_pr[-1],
                "best_v_roc": max(v_roc),
                "best_v_pr": max(v_pr),
            })

    df = pd.DataFrame(data)

    # --- Barplot of best validation metrics per run
    df[["run", "best_v_roc", "best_v_pr"]].set_index("run").plot(
        kind="bar", figsize=(10, 5), rot=45)
    plt.ylabel("Score")
    plt.title("Best Validation Metrics per Run")
    plt.tight_layout()
    plt.show()

    # --- Scatterplot (ROC vs PR)
    plt.figure(figsize=(6, 6))
    plt.scatter(df["best_v_roc"], df["best_v_pr"], s=80)
    for _, row in df.iterrows():
        plt.text(row["best_v_roc"], row["best_v_pr"], row["run"], fontsize=8)
    plt.xlabel("Best Val ROC-AUC")
    plt.ylabel("Best Val PR-AUC")
    plt.title("Runs Comparison")
    plt.grid(True)
    plt.show()
