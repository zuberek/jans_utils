import re
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_metrics(
    log_file="training_log.txt",
    interactive=True,
    save=False,
):

    # regex patterns
    train_pat = re.compile(
        r"Epoch (\d+).*?train_losses: \[([\d.eE+-]+)\].*?t_roc_auc': ([\d.eE+-]+), 't_pr_auc': ([\d.eE+-]+)"
    )
    val_pat = re.compile(
        r"Epoch (\d+).*?val_losses: \[([\d.eE+-]+)\].*?v_roc_auc': ([\d.eE+-]+), 'v_pr_auc': ([\d.eE+-]+)"
    )

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

    df = pd.DataFrame({
        "epoch": epochs,
        "train_loss": train_loss,
        "t_roc_auc": t_roc,
        "t_pr_auc": t_pr,
        "val_loss": val_loss,
        "v_roc_auc": v_roc,
        "v_pr_auc": v_pr,
    })

    # print(df.head())

    if interactive:
        fig = go.Figure()

        # # Loss curves
        # fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"],
        #               mode="lines+markers", name="Train Loss", yaxis="y1"))
        # fig.add_trace(go.Scatter(
        #     x=df["epoch"], y=df["val_loss"], mode="lines+markers", name="Val Loss", yaxis="y1"))

        # # AUC curves
        # fig.add_trace(go.Scatter(x=df["epoch"], y=df["t_roc_auc"],
        #               mode="lines+markers", name="Train ROC-AUC", yaxis="y2"))
        # fig.add_trace(go.Scatter(x=df["epoch"], y=df["v_roc_auc"],
        #               mode="lines+markers", name="Val ROC-AUC", yaxis="y2"))
        # fig.add_trace(go.Scatter(x=df["epoch"], y=df["t_pr_auc"],
        #               mode="lines+markers", name="Train PR-AUC", yaxis="y2"))
        # fig.add_trace(go.Scatter(
        #     x=df["epoch"], y=df["v_pr_auc"], mode="lines+markers", name="Val PR-AUC", yaxis="y2"))

        # # Layout with two y-axes
        # fig.update_layout(
        #     title="Training & Validation Metrics",
        #     xaxis=dict(title="Epoch"),
        #     yaxis=dict(title="Loss", side="left"),
        #     yaxis2=dict(title="AUC", overlaying="y",
        #                 side="right", range=[0, 1]),
        #     legend=dict(x=0.01, y=0.99),
        #     template="plotly_white"
        # )

        # fig.show()

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
            title="Training & Validation Metrics",
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

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.plot(df["epoch"], df["train_loss"],
                 label="Train Loss", color="tab:blue")
        ax1.plot(df["epoch"], df["val_loss"],
                 label="Val Loss", color="tab:cyan")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.set_ylabel("AUC")
        ax2.plot(df["epoch"], df["t_roc_auc"],
                 label="Train ROC-AUC", color="tab:red")
        ax2.plot(df["epoch"], df["v_roc_auc"],
                 label="Val ROC-AUC", color="tab:orange")
        ax2.plot(df["epoch"], df["t_pr_auc"],
                 label="Train PR-AUC", color="tab:green")
        ax2.plot(df["epoch"], df["v_pr_auc"],
                 label="Val PR-AUC", color="tab:olive")
        ax2.legend(loc="lower right")

        plt.title("Training & Validation Metrics")
        plt.tight_layout()
        plt.show()
