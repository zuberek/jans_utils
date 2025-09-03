# %%
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# %%


def plot_cross_metrics(
    df: pd.DataFrame,
    log_dir: Path | str,
    metric: str = "v_pr_auc",
):
    # compute last values
    # best_vals = (
    #     df.groupby("run")
    #     .apply(lambda g: g[metric].iloc[-1])
    #     .to_dict()
    # )
    best_vals = (
        df.groupby("run")[metric]
        .max()
        .to_dict()
    )

    runs_labels = df.groupby('run')['run_labels'].first()

    # sort runs by last metric (descending)
    sorted_runs = sorted(best_vals, key=best_vals.get, reverse=True)

    plt.figure(figsize=(10, 6))

    for run in sorted_runs:
        subdf = df[df["run"] == run]
        label = f"{run} ({runs_labels.loc[run]})"
        plt.plot(subdf["epoch"], subdf[metric], marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} across runs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # print(sorted_runs)
    # print([glob.glob(f"{log_dir / run}/**/*.out", recursive=True)[0]
    #       for run in sorted_runs])

    from IPython.display import HTML, display
    links = []
    for run in sorted_runs:
        out_file = glob.glob(f"{log_dir / run}/**/*.out", recursive=True)[0]
        links.append(f'<a href="{out_file}" target="_blank">{run}</a>')
    display(HTML(" | ".join(links)))
    # for run in sorted_runs:
    #     print(f"{run}: {best_vals[run]:.4f}")
