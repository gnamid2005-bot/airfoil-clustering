# Code map

The repository preserves more code than is needed for a quick visual review. The files below are the shortest path through the main research workflow.

## Data acquisition and preprocessing

- `src/download_airfoils.py`: downloads and safely extracts the UIUC coordinate archive.
- `src/airfoil_data.py`: parses, normalizes, and resamples the earlier direct-coordinate representation.
- `src/gabriel_pinned_registration.py`: Gabriel-format parsing, pinned resampling, contour validation, SVF registration, and diagnostics.
- `src/run_gabriel_pinned_registration_dev.py`: constructs and validates the 1,652-contour Gabriel dataset.
- `references/pinned_registration.py`: retained reference implementation used by the source-faithfulness checks.

## K-means analysis

- `src/run_airfoil_kmeans.py`: direct-coordinate K-means workflow.
- `src/run_airfoil_kmeans_stability.py`: multi-seed K-means stability and inertia figures for the 1,654-sample dataset.
- `src/run_gabriel_pinned_kmeans_r200.py`: repeated K-means fits on the 1,652 Gabriel representation.
- `src/analyse_airfoil_kmeans_modal_assignments_r200.py`: modal-partition analysis.
- `src/run_gabriel_modal_kmeans_registration.py`: registration to cluster-mean K-means templates and deformation diagnostics.

## Registration-field PCA and GMM analysis

- `scripts/run_gabriel_globalmean_field_pca_gmm.py`: global-mean registration coefficients, training-only PCA, GMM grid, BIC, ICL, and held-out likelihood.
- `scripts/run_gabriel_big_registration_k2_k7_full_leftmost.py`: selected `K=2–7` GMM-template branches.
- `scripts/run_gabriel_big_registration_k8_k11.py`: selected `K=8–11` GMM-template branches.
- `scripts/coordinator_gabriel_big_registration_k2_k7_full_leftmost.py`: combines the branch summaries and creates the physical-cost figure.
- `scripts/run_gabriel_task2_assignment_consistency_refit.py`: fixed-`K` assignment stability across PCA dimensions.

## Shape interpretation and poster figures

- `scripts/plot_gabriel_airfoil_clusters_k1_k11.py`: cluster mean geometry across `K=1–11`.
- `scripts/analyze_gabriel_cross_k_cluster_evolution.py`: contingency and movement analysis across `K=8–11`.
- `scripts/revise_gabriel_sankey_lineage_colours.py`: final alluvial/Sankey rendering.
- `scripts/reproduce_poster_figures.py`: lightweight entry point for recreating the poster figures from saved artifacts.

## Additional research code

The remaining files in `src/` and `scripts/` are preserved because they document validation, sensitivity analysis, alternative representations, earlier experiments, and the development path that led to the final poster. They are not all required to reproduce the displayed figures.

