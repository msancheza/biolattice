import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import config

# Minimalist palette (neutral background + soft accents per channel)
_STYLE = {
    "fig_bg": "#e6e8ed",
    "card_face": "#ffffff",
    "card_edge": "#c5cdd9",
    "title": "#1e293b",
    "subtitle": "#64748b",
    "cmaps": ["gray", "magma", "cividis"],
    "tags": [
        {"code": "C1", "label": "Hybrid Str", "bg": "#ccfbf1", "edge": "#2dd4bf", "fg": "#0f766e"},
        {"code": "C2", "label": "Denoised Het", "bg": "#fef3c7", "edge": "#f59e0b", "fg": "#b45309"},
        {"code": "C3", "label": "Aligned Kin", "bg": "#e0e7ff", "edge": "#818cf8", "fg": "#3730a3"},
    ],
}


def visualize_micro_cube(tensor_path, show=True):
    data_obj = torch.load(tensor_path, map_location="cpu")
    cube = data_obj['tensor']
    slice_idx = config.MICRO_CUBE_SIZE // 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), facecolor=_STYLE["fig_bg"])

    fig.suptitle(
        "Multi-modal Micro-cube · Middle Axial Slice",
        fontsize=15,
        fontweight=600,
        color=_STYLE["title"],
        y=0.97,
    )

    channel_titles = [
        "Channel 1 · hybrid (max + avg)",
        "Channel 2 · denoised heterogeneity",
        "Channel 3 · aligned wash-in (post − pre)",
    ]

    for i, ax in enumerate(axes):
        slice_2d = cube[i, slice_idx, :, :].numpy()
        tag = _STYLE["tags"][i]

        im = ax.imshow(slice_2d, cmap=_STYLE["cmaps"][i], interpolation="nearest")
        ax.axis("off")

        # Top tag (pill)
        ax.text(
            0.04,
            0.96,
            f'{tag["code"]}\n{tag["label"]}',
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight=600,
            color=tag["fg"],
            linespacing=1.15,
            bbox={
                "boxstyle": "round,pad=0.45,rounding_size=0.15",
                "facecolor": tag["bg"],
                "edgecolor": tag["edge"],
                "linewidth": 1,
                "alpha": 0.92,
            },
        )

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.82)
        cb.outline.set_edgecolor(_STYLE["card_edge"])
        cb.ax.tick_params(colors=_STYLE["subtitle"], labelsize=8)

        # Bottom card (panel footer)
        ax.text(
            0.5,
            -0.065,
            channel_titles[i],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color=_STYLE["title"],
            bbox={
                "boxstyle": "round,pad=0.5,rounding_size=0.2",
                "facecolor": _STYLE["card_face"],
                "edgecolor": _STYLE["card_edge"],
                "linewidth": 0.9,
                "alpha": 0.98,
            },
        )

    plt.subplots_adjust(top=0.88, bottom=0.18, left=0.05, right=0.96, wspace=0.28)

    # Card-like frames behind each panel
    for ax in axes:
        pos = ax.get_position()
        pad_x, pad_y = 0.01, 0.015
        extra_bottom = 0.055
        card = FancyBboxPatch(
            (pos.x0 - pad_x, pos.y0 - pad_y - extra_bottom),
            pos.width + 2 * pad_x,
            pos.height + 2 * pad_y + extra_bottom,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            transform=fig.transFigure,
            facecolor=_STYLE["card_face"],
            edgecolor=_STYLE["card_edge"],
            linewidth=0.85,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(card)
        ax.set_zorder(2)

    if show:
        plt.show()
    return fig

# Usage: visualize_micro_cube('datasets/micro_cubes/Breast_MRI_001_lattice.pt')
