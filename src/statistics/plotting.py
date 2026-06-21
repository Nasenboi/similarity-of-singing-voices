import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def plot_model_train_results(
    test_loss,
    train_loss,
    test_accuracy=None,
    train_accuracy=None,
    save_path: str = None,
    model_name: str = "Model",
):
    """Plot the training progress of a model

    Args:
        test_loss (iterable): The test loss
        train_loss (iterable): The train loss
        test_accuracy (iterable, optional): The test accuracy. Defaults to None.
        train_accuracy (iterable, optional): The train accuracy. Defaults to None.
        save_path (str, optional): An optional path to save the plot to. Defaults to None.
        model_name (str, optional): The models name. Defaults to "Model".
    """
    fig, ax1 = plt.subplots()
    ax1.plot(test_loss, label="Test Loss", color="red")
    ax1.plot(train_loss, label="Train Loss", color="orange", linestyle=":")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    if test_accuracy is not None or train_accuracy is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy", color="black")
        ax2.tick_params(axis="y", labelcolor="black")

        if test_accuracy is not None:
            ax2.plot(test_accuracy, label="Test Accuracy", color="green")
        if train_accuracy is not None:
            ax2.plot(train_accuracy, label="Train Accuracy", color="blue", linestyle=":")

    plt.title(f"{model_name} Training Progress")
    lines1, labels1 = ax1.get_legend_handles_labels()
    if test_accuracy is not None or train_accuracy is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_scores(
    x,
    y,
    title: str = "Accuracy Scores",
    xlabel: str = "Accuracy (%)",
    ylabel: str = "Features",
    save_path: str = None,
    random_chance: float = None,
    target_feature: float = None,
    legend_loc: str = "best",
):
    """Plots any scores as a bar plot

    Args:
        x (iterable): the scores
        y (iterable): the labels
        title (str, optional): Plot title. Defaults to "Accuracy Scores".
        xlabel (str, optional): X axis label. Defaults to "Accuracy (%)".
        ylabel (str, optional): Y axis label_. Defaults to "Features".
        save_path (str, optional): Optional path to save the model to. Defaults to None.
        random_chance (float, optional): Optional random chance line. Defaults to None.
        target_feature (float, optional): Optional human base line. Defaults to None.
        legend_loc (string, optional): Optional location of the legend. Defaults to "best".
    """
    x, y = list(x), list(y)
    plt.barh(y=y, width=x)
    for i, v in enumerate(x):
        plt.text(0.01, i, f"{v:.3f}", va="center", ha="left")
    if random_chance is not None:
        plt.axvline(
            x=random_chance,
            linestyle=":",
            color="red",
            alpha=1.0,
            label=f"Random chance = {random_chance:.3f}",
        )

    if target_feature is not None:
        plt.axvline(
            x=target_feature,
            linestyle=":",
            color="green",
            alpha=1.0,
            label=f"Human Baseline = {target_feature:.3f}",
        )

    if random_chance is not None or target_feature is not None:
        plt.legend(loc=legend_loc)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


def plot_correlation_bar(
    feature_df,
    target_feature,
    top_x: int = 15,
    title: str = "Feature Correlations with Human Baseline",
    xlabel: str = "Pearson r",
    save_path: str = None,
):
    baseline = np.asarray(target_feature, dtype=float)
    records = []
    for col in feature_df.columns:
        vals = np.asarray(feature_df[col], dtype=float)
        mask = ~(np.isnan(vals) | np.isnan(baseline))
        if mask.sum() < 3:
            continue
        r, p = stats.pearsonr(vals[mask], baseline[mask])
        records.append({"feature": col, "r": r, "p_value": p, "n": mask.sum()})

    corr_df = pd.DataFrame(records).assign(abs_r=lambda d: d["r"].abs()).nlargest(top_x, "abs_r").sort_values("r")

    colors = ["#C44E52" if r < 0 else "#4C72B0" for r in corr_df["r"]]

    fig, ax = plt.subplots(figsize=(7, max(4, top_x * 0.45)))

    bars = ax.barh(
        y=corr_df["feature"],
        width=corr_df["r"],
        color=colors,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    for bar, (_, row) in zip(bars, corr_df.iterrows()):
        stars = (
            "***"
            if row["p_value"] < 0.001
            else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        )
        if stars:
            x_pos = row["r"] + (0.01 if row["r"] >= 0 else -0.01)
            ha = "left" if row["r"] >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, stars, va="center", ha=ha, fontsize=8, color="#333333")

    ax.axvline(0, color="black", linewidth=0.8, zorder=4)

    ax.text(
        0.98,
        0.02,
        "* p<.05   ** p<.01   *** p<.001",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_correlation_scatter(
    x,
    y,
    title: str = "Feature Correlation",
    xlabel: str = "Human Baseline",
    ylabel: str = "Features",
    save_path: str = None,
    legend_loc: str = "best",
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    r, p_value = stats.pearsonr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        x,
        y,
        color="#4C72B0",
        alpha=0.75,
        edgecolors="white",
        linewidths=0.5,
        s=60,
        zorder=3,
        label="Data points",
    )

    ax.plot(
        x_line,
        y_line,
        color="#C44E52",
        linewidth=1.8,
        label=f"r = {r:.3f}  (p = {p_value:.2e})",
        zorder=4,
    )

    n = len(x)
    x_mean = x.mean()
    se = np.sqrt(
        np.sum((y - (slope * x + intercept)) ** 2)
        / (n - 2)
        * (1 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2))
    )
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ax.fill_between(
        x_line,
        y_line - t_crit * se,
        y_line + t_crit * se,
        color="#C44E52",
        alpha=0.12,
        label="95 % CI",
        zorder=2,
    )

    stats_text = f"Pearson r = {r:.3f}\np-value  = {p_value:.2e}\nn        = {n}"
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
    )

    ax.legend(loc=legend_loc)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()
