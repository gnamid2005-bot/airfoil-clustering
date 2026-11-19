# Gabriel PCA-SVF clusters: evolution from K=8 to K=11

**Technical summary.** The evidence is most consistent with **H2 — hierarchical refinement/splitting**. Across adjacent transitions, dominant-continuation retention averages 77.5%; mean dominant-successor mean-shape RMS displacement averages 0.00587 chord units. ARI is not a movement percentage: it evaluates agreement of all sample pairs and can fall markedly when one large family is split.

## Overlap and movement evidence

| Transition | ARI | AMI | NMI | dominant retained | non-retained | Hungarian matched |
|---|---:|---:|---:|---:|---:|---:|
| K8→K9 | 0.477 | 0.574 | 0.578 | 68.8% | 31.2% | 68.8% |
| K9→K10 | 0.792 | 0.773 | 0.776 | 85.6% | 14.4% | 85.6% |
| K10→K11 | 0.664 | 0.692 | 0.696 | 78.1% | 21.9% | 78.1% |

Dominant retention asks whether each airfoil follows its old cluster's largest-overlap successor. Hungarian matching is stricter and can label legitimate one-to-many splits as unmatched.

## Physical mean shapes remain close along dominant paths

Across adjacent comparisons the average cluster-level RMS change is 0.00587 chord units. See `mean_shape_evolution.png` and the per-transition mean-shape tables for cluster-specific values and maxima.

## Splits, merges, and moved geometries

The heuristic classification uses raw overlap first: stable means retention ≥80% and secondary share <15%; split means secondary share ≥15%; possible merge means a new cluster's second-largest source contributes ≥20%. These labels are descriptive, not ground truth. 13 major movement figures show the actual airfoils on dominant versus secondary paths, with subgroup means computed from all subgroup members.

Airfoil-level dominant-path summary: 942 very stable, 387 involved in one non-dominant step, and 323 involved in repeated non-dominant steps.

## Scope and definitions

N=1652 canonical Gabriel uniformly resampled 200-point contours. Assignments are the saved PCA-SVF GMM solutions K8/q27, K9/q14, K10/q17, and K11/q15. Correspondence is inferred from contingency overlap, never raw-label equality. Mean-shape distance is RMS pointwise Euclidean distance in chord-normalized coordinates.

## Robustness and limitations

All contingency totals, row normalizations, marginal cluster sizes, sample identities, and saved template means were checked. The split/merge thresholds are explicit heuristics; conclusions should be read together with raw matrices and shape plots. The analysis is descriptive and does not establish clustering truth.

## Recommended next step

Review the largest split panels with Gabriel and relate the visible subtypes to domain-relevant airfoil properties before attaching semantic names to clusters.

## Further question

Do the largest secondary paths remain stable under nearby q values for the same K? That sensitivity analysis is outside this task and would require separately approved existing-model outputs.
