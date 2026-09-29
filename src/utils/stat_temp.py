import numpy as np
import pandas as pd
import scikit_posthocs as sp
# import plotly.figure_factory as ff
from scipy.stats import friedmanchisquare

# Simulated mAP scores for 4 algorithms across 4 datasets
# Rows: datasets, Columns: algorithms
map_scores = pd.read_csv('runs/detect/mAP50_comparison.csv', index_col=0)

print(map_scores.head())

# Perform Friedman test
stat, p = friedmanchisquare(*[map_scores[col] for col in map_scores.columns])

# If significant, perform Nemenyi post-hoc test
if p < 0.01:
    nemenyi_results = sp.posthoc_nemenyi_friedman(map_scores.values)
    nemenyi_results.columns = map_scores.columns
    nemenyi_results.index = map_scores.columns
else:
    nemenyi_results = "No significant differences found."

# # Create Critical Difference diagram
# import Orange


# # Calculate average ranks
# ranks = map_scores.rank(axis=1, ascending=False)
# avg_ranks = ranks.mean().values
# names = map_scores.columns.tolist()

# # Generate CD diagram
# cd = Orange.evaluation.compute_CD(avg_ranks, len(map_scores), test='nemenyi')
# fig = graph_ranks(avg_ranks, names, cd=cd, width=6)
# fig.write_image("cd_diagram.png")
# fig.write_json("cd_diagram.json")

# Output results
print(f"Friedman test statistic: {stat}, p-value: {p}\n")
print("Nemenyi post-hoc test results:")
print(nemenyi_results)
