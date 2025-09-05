# %%
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import json

from .collect_metrics import extract_dataset_sizes, try_eval


# %%


def plot_cross_metrics(
    df: pd.DataFrame,
    log_dir: Path | str,
    metric: str = "v_pr_auc",
    plot_exp_params: list[str] | None = None,
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

    example_params_path = glob.glob(
        f"{log_dir}/**/params.json", recursive=True)[0]
    with open(example_params_path) as f:
        exp_params = json.load(f)
        exp_params = {k: try_eval(v) for k, v in exp_params.items()}

    # sort runs by last metric (descending)
    sorted_runs = sorted(best_vals, key=best_vals.get, reverse=True)

    # plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))

    for run in sorted_runs:
        subdf = df[df["run"] == run]
        label = f"{run} ({runs_labels.loc[run]})"
        plt.plot(subdf["epoch"], subdf[metric], marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{exp_params['experiment_name']} - {metric=}", fontsize=14)
    
    info_lines = []
    if (dsets_data := extract_dataset_sizes(glob.glob(f"{log_dir}/**/*.out")[0])):
        sizes_str = " | ".join(f"{k}: {v['count']}" for k, v in dsets_data.items())
        info_lines.append(f"Dsets sizes - {sizes_str}")

    if plot_exp_params is not None:
        params_str = " | ".join(
            f"{k}: {v}" for k, v in exp_params.items() if k in plot_exp_params
        )
        info_lines.append(f"Exp params - {params_str}")

    # Add box inside plot (bottom left here)
    if info_lines:
        ax.text(
            0.01, 0.99,             # relative axes coords
            "\n".join(info_lines),  # multi-line text
            transform=ax.transAxes,
            fontsize=9,
            va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.4, edgecolor="none", boxstyle="round,pad=0.3")
        )
    
    # if (dsets_data := extract_dataset_sizes(glob.glob(f"{log_dir}/**/*.out")[0])):
    #     sizes_str = " | ".join(f"{k}: {v['count']}" for k, v in dsets_data.items())
    #     plt.figtext(0.1, 0.86, f"Datasets    | {sizes_str}", fontsize=9)
    #     # plt.suptitle(f"Datasets | {sizes_str}", fontsize=9, y=0.83)

    # if plot_exp_params is not None:
    #     params_str = " | ".join(f"{k}: {v}" for k, v in exp_params.items() if k in plot_exp_params)
    #     plt.figtext(0.1, 0.83, f"Params    | {params_str}", fontsize=9)
        
    # plt.subplots_adjust(top=0.8)
    
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
