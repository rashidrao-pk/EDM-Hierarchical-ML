from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300,
                         "font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "legend.fontsize": 9})


def correlation_heatmap(df: pd.DataFrame, path: Path) -> None:
    corr = df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True,
                annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Pearson correlation matrix")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def effects_plot(df: pd.DataFrame, target: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    sns.lineplot(data=df, x="I", y=target, hue="E", style="D", markers=True,
                 errorbar=None, ax=axes[0])
    axes[0].set_title(f"{target}: operating-level interactions")
    sns.pointplot(data=df, x="E", y=target, hue="D", errorbar=None,
                  dodge=0.25, ax=axes[1])
    axes[1].set_title(f"{target}: categorical main effects")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def observed_vs_predicted(pred: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.3, 4.0))
    ax.scatter(pred["observed"], pred["predicted"], s=38, edgecolor="black", linewidth=0.4)
    lo = min(pred["observed"].min(), pred["predicted"].min())
    hi = max(pred["observed"].max(), pred["predicted"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
    ax.set(xlabel="Observed", ylabel="Out-of-fold prediction", title=title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

