import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, studentized_range
import matplotlib.pyplot as plt
from critdd import Diagram

# -------------------------
# Example results: datasets × algorithms
# -------------------------
results_pedestrian = pd.DataFrame({
    "yolo8":      [0.82, 0.75, 0.90, 0.88, 0.83, 0.80],
    "yolo11":  [0.79, 0.77, 0.88, 0.86, 0.85, 0.81],
    "yolo12":      [0.70, 0.72, 0.75, 0.73, 0.71, 0.69],
    "RT-DETR":[0.68, 0.70, 0.72, 0.69, 0.67, 0.65],
})

results_vulnerable = pd.DataFrame({
    "yolo8":      [0.82, 0.75, 0.90, 0.88, 0.83, 0.80],
    "yolo11":  [0.79, 0.77, 0.88, 0.86, 0.85, 0.81],
    "yolo12":      [0.70, 0.72, 0.75, 0.73, 0.71, 0.69],
    "RT-DETR":[0.68, 0.70, 0.72, 0.69, 0.67, 0.65],
})

results_all = pd.read_csv('runs/detect/mAP50_comparison.csv', index_col=0)
print(results_all.head())


# Ranks per dataset
ranks = results_all.rank(axis=1, ascending=False)
avg_ranks = ranks.mean().values
labels = results_all.columns.tolist()

# Friedman test
stat, p = friedmanchisquare(*[results_all[algo] for algo in results_all.columns])
print(f"Friedman test: chi2={stat:.3f}, p={p}")

# Nemenyi posthoc test (matrix of p-values)
nemenyi = sp.posthoc_nemenyi_friedman(results_all.values)
nemenyi.columns = results_all.columns
nemenyi.index = results_all.columns
print("\nNemenyi test p-values:\n", nemenyi)

# -------------------------
# Compute Critical Difference
# -------------------------
k = len(labels)         # number of algorithms
N = results_all.shape[0]    # number of datasets
alpha = 0.05

# Studentized range quantile (q_alpha)
q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)

# Critical Difference
CD = q_alpha * np.sqrt(k * (k+1) / (6 * N))
print(f"\nCritical Difference (CD) = {CD:.3f}")

# -------------------------
# CD Diagram
# -------------------------
def plot_cd_diagram(avg_ranks, names, cd=None, width=6, textspace=1.5, title="Critical Difference Diagram"):
    k = len(avg_ranks)
    order = np.argsort(avg_ranks)
    avg_ranks = np.array(avg_ranks)[order]
    names = np.array(names)[order]

    fig, ax = plt.subplots(figsize=(width, 2))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 2)
    ax.set_title(title, fontsize=12)

    # Axis for ranks
    ax.hlines(1, 0.5, k + 0.5, color="black")
    for rank, name in zip(avg_ranks, names):
        ax.plot(rank, 1, "o", markersize=6, color="black")
        ax.text(rank, 1.05, name, rotation=60, ha="center", va="bottom")

        # Draw CD line from each point
        if cd is not None and name != names[-1]:
            ax.hlines(0.9, rank, rank + cd, lw=1.5, color="red")
            ax.vlines([rank, rank + cd], 0.85, 0.95, color="red")


    # Draw CD bar if available
    if cd is not None:
        start = np.min(avg_ranks)
        ax.hlines(1.2, start, start + cd, lw=2, color="black")
        ax.vlines([start, start + cd], 1.15, 1.25, color="black")
        ax.text(start + cd, 1.3, f"CD={cd:.4f}", ha="center", va="bottom")

    ax.axis("off")
    plt.savefig("cd_diagram.png", dpi=300, bbox_inches="tight")

# Plot
plot_cd_diagram(avg_ranks, labels, cd=CD)


# create a CD diagram from the Pandas DataFrame
diagram = Diagram(
    results_all.to_numpy(),
    treatment_names = results_all.columns,
    maximize_outcome = True
)

# inspect average ranks and groups of statistically indistinguishable treatments
diagram.average_ranks # the average rank of each treatment
diagram.get_groups(alpha=.05, adjustment="holm")

# export the diagram to a file
diagram.to_file(
    "example.tex",
    alpha = .05,
    adjustment = "holm",
    reverse_x = True,
    axis_options = {"title": "critdd"},
)
