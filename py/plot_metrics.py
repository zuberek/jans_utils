import re
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import glob



def plot_metrics(
    df: pd.DataFrame,
    run: str,
    interactive=False,
    save=False,
):

    df = df[df.run == run]
    run_labels = df['run_labels'].iloc[0]

    if interactive:
        fig = go.Figure()

        # --- Loss curves (blue palette, solid lines, circles/triangles)
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["train_loss"],
            mode="lines+markers", name="Train Loss", yaxis="y1",
            line=dict(color="#1f77b4", dash="solid", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["val_loss"],
            mode="lines+markers", name="Val Loss", yaxis="y1",
            line=dict(color="#17becf", dash="solid", width=2),
        ))

        # --- AUC curves (warm/green palette, dashed/dotted lines, squares/diamonds)
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["t_roc_auc"],
            mode="lines+markers", name="Train ROC-AUC", yaxis="y2",
            line=dict(color="#d62728", dash="dash", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["v_roc_auc"],
            mode="lines+markers", name="Val ROC-AUC", yaxis="y2",
            line=dict(color="#ff7f0e", dash="dash", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["t_pr_auc"],
            mode="lines+markers", name="Train PR-AUC", yaxis="y2",
            line=dict(color="#2ca02c", dash="dot", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["v_pr_auc"],
            mode="lines+markers", name="Val PR-AUC", yaxis="y2",
            line=dict(color="#9467bd", dash="dot", width=2),
        ))

        # Layout with two y-axes
        fig.update_layout(
            title=f"Run {run}: {run_labels}",
            xaxis=dict(title="Epoch"),
            yaxis=dict(title="Loss", side="left"),
            yaxis2=dict(title="AUC", overlaying="y",
                        side="right", range=[0, 1]),
            legend=dict(x=0.01, y=0.99,
                        bordercolor="lightgrey", borderwidth=1),
            template="plotly_white"
        )

        fig.show()

        # Export to HTML if needed
        if save:
            fig.write_html("metrics.html")

    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # --- Loss curves (blue palette, solid lines, circles/triangles)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax1.plot(df["epoch"], df["train_loss"],
                 label="Train Loss", color="#1f77b4", linestyle="-",
                 linewidth=2, marker="o")
        ax1.plot(df["epoch"], df["val_loss"],
                 label="Val Loss", color="#17becf", linestyle="-",
                 linewidth=2, marker="o")
        ax1.legend(loc="upper left")

        # --- AUC curves (warm/green palette, dashed/dotted lines, squares/diamonds)
        ax2 = ax1.twinx()
        ax2.set_ylabel("AUC")

        ax2.plot(df["epoch"], df["t_roc_auc"],
                 label="Train ROC-AUC", color="#d62728", linestyle="--",
                 linewidth=2, marker="o")
        ax2.plot(df["epoch"], df["v_roc_auc"],
                 label="Val ROC-AUC", color="#ff7f0e", linestyle="--",
                 linewidth=2, marker="o")
        ax2.plot(df["epoch"], df["t_pr_auc"],
                 label="Train PR-AUC", color="#2ca02c", linestyle=":",
                 linewidth=2, marker="o")
        ax2.plot(df["epoch"], df["v_pr_auc"],
                 label="Val PR-AUC", color="#9467bd", linestyle=":",
                 linewidth=2, marker="o")
        ax2.legend(loc="upper right", handlelength=3)

        plt.title(f"Run {run}: {run_labels}")
        plt.tight_layout()
        plt.show()
