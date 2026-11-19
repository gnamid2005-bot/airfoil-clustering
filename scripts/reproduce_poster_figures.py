"""Recreate the poster figures from saved, inspectable research artifacts.

This script is deliberately lightweight.  It redraws figures from the saved
CSV/NPY/NPZ artifacts and does not rerun registration or refit PCA/GMM models.
The two more involved geometry figures are delegated to their original,
validated plotting scripts included in this repository.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-airfoil-clustering-github")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "generated"
SELECTED_Q = {8: 27, 9: 14, 10: 17, 11: 15}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finish(fig: plt.Figure, name: str, dpi: int = 240) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def airfoil_representation() -> None:
    path = ROOT / "data/airfoils/processed_gabriel/global_mean_gabriel.npy"
    contour = np.load(path)
    if contour.shape != (200, 2):
        raise ValueError(f"Expected a (200, 2) contour, got {contour.shape}")

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    upper, lower = contour[:100], contour[100:]
    ax.plot(upper[:, 0], upper[:, 1], color="#2563EB", lw=1.2)
    ax.scatter(upper[:, 0], upper[:, 1], color="#2563EB", s=11, label="Upper surface: 100 points")
    ax.plot(lower[:, 0], lower[:, 1], color="#D97706", lw=1.2)
    ax.scatter(lower[:, 0], lower[:, 1], color="#D97706", s=11, label="Lower surface: 100 points")
    ax.scatter([contour[0, 0]], [contour[0, 1]], marker="s", s=45, color="#111827", zorder=5)
    ax.scatter([contour[100, 0]], [contour[100, 1]], marker="D", s=45, color="#111827", zorder=5)
    ax.annotate("Trailing edge / index 0", contour[0], xytext=(-115, 22), textcoords="offset points", arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.annotate("Leading edge / index 100", contour[100], xytext=(24, -12), textcoords="offset points", arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.set_title("One Airfoil Represented by 200 Corresponding 2D Points")
    ax.set_xlabel("x / chord")
    ax.set_ylabel("y / chord")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.margins(x=0.05, y=0.30)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    finish(fig, "01_airfoil_200_corresponding_points.png")


def kmeans_inertia() -> None:
    rows = read_csv(ROOT / "outputs/airfoil_kmeans_stability/kmeans_inertia_summary.csv")
    k = np.asarray([int(r["k"]) for r in rows])
    mean = np.asarray([float(r["mean_inertia"]) for r in rows])
    std = np.asarray([float(r["std_inertia"]) for r in rows])

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.fill_between(k, mean - std, mean + std, color="#FDE68A", alpha=0.55, label="mean ± standard deviation")
    ax.plot(k, mean, "o-", color="#B45309", linewidth=2.0, label="mean inertia")
    ax.set(xlabel="Number of clusters K", ylabel="Mean K-means inertia", title="K-means Inertia vs Number of Clusters")
    ax.set_xticks(k)
    ax.grid(True, alpha=0.22, color="#9CA3AF")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    finish(fig, "02_kmeans_inertia_vs_k.png")


def kmeans_template_deformation() -> None:
    rows = read_csv(ROOT / "outputs/gabriel_modal_kmeans_registration/tables/registration_summary_k2_k9.csv")
    k = np.asarray([int(r["K"]) for r in rows])
    energy = np.asarray([float(r["dirichlet_mean"]) for r in rows])
    energy_se = np.asarray([float(r["dirichlet_se"]) for r in rows])
    magnitude = np.asarray([float(r["svf_magnitude_mean"]) for r in rows])
    magnitude_se = np.asarray([float(r["svf_magnitude_se"]) for r in rows])

    fig, left = plt.subplots(figsize=(9.4, 5.2))
    right = left.twinx()
    energy_artist = left.errorbar(k, energy, yerr=energy_se, color="#1F4E79", marker="o", capsize=4, lw=2, label="Dirichlet energy")
    magnitude_artist = right.errorbar(k, magnitude, yerr=magnitude_se, color="#C45A18", marker="s", capsize=4, lw=2, label="SVF magnitude")
    left.set(xlabel="Number of registration templates K", ylabel="Average physical Dirichlet energy", xticks=k)
    right.set_ylabel("Average SVF magnitude")
    left.set_title("K-means Template Registration: Deformation vs Number of Templates")
    left.grid(axis="y", alpha=0.22)
    left.spines["top"].set_visible(False)
    right.spines["top"].set_visible(False)
    left.legend([energy_artist, magnitude_artist], ["Dirichlet energy", "SVF magnitude"], frameon=False, loc="upper right")
    finish(fig, "03_kmeans_template_deformation.png")


def preferred_gmm_components() -> None:
    rows = read_csv(ROOT / "outputs/gabriel_globalmean_field_pca_gmm/gmm/preferred_k_by_q.csv")
    q = np.asarray([int(r["q"]) for r in rows])
    curves = (
        ("preferred_k_bic", "BIC", "#2563EB", "o"),
        ("preferred_k_icl", "ICL", "#D97706", "s"),
        ("preferred_k_heldout_ll", "Held-out LL", "#6B7280", "^"),
    )
    fig, ax = plt.subplots(figsize=(11, 5.7))
    for field, label, color, marker in curves:
        ax.plot(q, [int(r[field]) for r in rows], label=label, color=color, marker=marker, ms=3, lw=1.4)
    ax.set(xlabel="PCA dimension q", ylabel="Preferred GMM component count K", title="Preferred Component Count Across Registration-Field PCA Dimensions")
    ax.set_xticks(range(0, 101, 10))
    ax.set_yticks(range(1, 21))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    finish(fig, "04_preferred_k_field_q1_q100.png", dpi=300)


def physical_deformation_cost() -> None:
    rows = read_csv(ROOT / "outputs/gabriel_big_registration_k2_k7_full_leftmost/tables/combined_registration_summary_k2_k11.csv")
    k = [int(r["K"]) for r in rows]
    q = [int(r["q"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    settings = (
        (axes[0], "dirichlet_mean", "dirichlet_se", "Average Physical Dirichlet Energy", "Physical Dirichlet energy"),
        (axes[1], "svf_mean", "svf_se", "Average SVF Magnitude", "SVF mean magnitude"),
    )
    for ax, value_key, se_key, title, ylabel in settings:
        values = [float(r[value_key]) for r in rows]
        errors = [float(r[se_key]) for r in rows]
        ax.errorbar(k, values, yerr=errors, marker="o", capsize=3)
        for x, y, dim in zip(k, values, q):
            ax.annotate(f"q={dim}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
        ax.set(title=title, xlabel="Number of templates K", ylabel=ylabel, xticks=k)
        ax.grid(alpha=0.25)
    fig.suptitle("K=2–7: Full-Range Leftmost Rule; K=8–11: Previous Post-Spike Rule", fontsize=10)
    finish(fig, "05_dirichlet_svf_k2_k11.png", dpi=280)


def adjacent_q_stability() -> None:
    rows = read_csv(ROOT / "outputs/gabriel_task2_assignment_consistency_refit/pca_svf/adjacent_q_ari.csv")
    for k in SELECTED_Q:
        selected = sorted(
            (r for r in rows if int(r["K"]) == k and int(r["lag"]) == 1),
            key=lambda r: int(r["q"]),
        )
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot([int(r["q"]) for r in selected], [float(r["ARI"]) for r in selected], color="#2563EB", lw=1.5)
        ax.axvline(SELECTED_Q[k], color="#DC2626", ls="--", label=f"selected q={SELECTED_Q[k]}")
        ax.set(xlabel="PCA dimension q", ylabel="ARI", title=f"Adjacent-q PCA-SVF Consistency, K={k}", ylim=(-0.02, 1.02))
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
        finish(fig, f"06_adjacent_q_ari_K{k:02d}.png")


def original_geometry_figures() -> None:
    scripts = (
        ROOT / "scripts/plot_gabriel_airfoil_clusters_k1_k11.py",
        ROOT / "scripts/revise_gabriel_sankey_lineage_colours.py",
    )
    for script in scripts:
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

    copies = {
        ROOT / "outputs/gabriel_airfoil_cluster_shapes_k1_k11/figures/overview/airfoil_cluster_means_K01_K11_poster_overview.png": OUT / "07_airfoil_cluster_means_K01_K11.png",
        ROOT / "outputs/gabriel_cross_k_cluster_evolution/figures/cluster_evolution_sankey_lineage_colours.png": OUT / "08_cluster_evolution_sankey.png",
    }
    for source, destination in copies.items():
        shutil.copy2(source, destination)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    airfoil_representation()
    kmeans_inertia()
    kmeans_template_deformation()
    preferred_gmm_components()
    physical_deformation_cost()
    adjacent_q_stability()
    original_geometry_figures()
    print(f"Created poster figures in {OUT}")


if __name__ == "__main__":
    main()
