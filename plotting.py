"""Plot functions exposed to model-generated analysis code."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_corr_matrix(
    corr,
    pval,
    out_path=None,
    xticklabels=None,
    yticklabels=None,
    title=None,
    alpha=0.05,
    vmax=0.40,
    figsize=(8, 7),
    reference_pval=None,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    corr = np.asarray(corr)
    pval = np.asarray(pval)

    if corr.ndim != 2 or pval.ndim != 2:
        raise ValueError("corr and pval must be 2D arrays.")

    if corr.shape != pval.shape:
        raise ValueError("corr and pval must have the same shape.")

    if reference_pval is not None:
        reference_pval = np.asarray(reference_pval)
        if reference_pval.shape != corr.shape:
            raise ValueError("reference_pval must have the same shape as corr.")

    fig, ax = plt.subplots(figsize=figsize)

    cmap = sns.light_palette("#3b7eb1", as_cmap=True)

    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.08)

    # Gray background
    sns.heatmap(
        np.zeros_like(corr),
        cmap=ListedColormap(["#ebedf0"]),
        cbar=False,
        ax=ax,
    )

    # Significant correlations
    sns.heatmap(
        np.abs(corr),
        vmin=0,
        vmax=vmax,
        mask=(pval >= alpha),
        cmap=cmap,
        linecolor="0.95",
        linewidths=1,
        cbar=True,
        cbar_ax=cbar_ax,
        ax=ax,
    )

    # Mark correlations significant in reference
    # but not significant in current matrix
    if reference_pval is not None:
        lost_significance = (reference_pval < alpha) & (pval >= alpha)

        for i, j in np.argwhere(lost_significance):
            ax.text(
                j + 0.5,
                i + 0.5,
                "x",
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold",
            )

    if title is not None:
        ax.set_title(title, fontsize=20, pad=12)

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, fontsize=14, rotation=90)

    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontsize=14, rotation=0)

    cbar = ax.collections[-1].colorbar
    cbar.set_ticks(np.linspace(0, vmax, 5))
    cbar.set_label("Abs. Spearman Correlation", fontsize=16, labelpad=8)

    plt.tight_layout()

    if out_path is not None:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fig.savefig(
            out_path,
            dpi=180,
            bbox_inches="tight",
            transparent=True,
        )

    return fig, ax
