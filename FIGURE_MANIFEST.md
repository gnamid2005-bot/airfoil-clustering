# Poster figure manifest

This table maps each quantitative or geometry figure in the poster to the retained image, its plotting code, and its direct inputs.

| ID | Figure | Plotting code | Direct saved inputs | Provenance status |
|---|---|---|---|---|
| 01 | One airfoil as 200 corresponding points | `scripts/reproduce_poster_figures.py` | `data/airfoils/processed_gabriel/global_mean_gabriel.npy` | Recreated for this repository because the poster-layout source was not found |
| 02 | K-means inertia vs `K` | `src/run_airfoil_kmeans_stability.py`; lightweight redraw in `scripts/reproduce_poster_figures.py` | `outputs/airfoil_kmeans_stability/kmeans_inertia_summary.csv` | Exact saved poster image retained |
| 03 | K-means registration deformation vs number of templates | `src/run_gabriel_modal_kmeans_registration.py`; combined redraw in `scripts/reproduce_poster_figures.py` | `outputs/gabriel_modal_kmeans_registration/tables/registration_summary_k2_k9.csv` | Exact separate energy/SVF source plots retained; combined poster chart recreated |
| 04 | Preferred GMM component count across PCA dimensions | `scripts/run_gabriel_globalmean_field_pca_gmm.py` | `outputs/gabriel_globalmean_field_pca_gmm/gmm/preferred_k_by_q.csv` | Exact saved poster image retained |
| 05 | Physical deformation cost, `K=2–11` | `scripts/coordinator_gabriel_big_registration_k2_k7_full_leftmost.py` | `outputs/gabriel_big_registration_k2_k7_full_leftmost/tables/combined_registration_summary_k2_k11.csv` | Exact saved poster image retained |
| 06 | Adjacent-`q` ARI for `K=8–11` | `scripts/run_gabriel_task2_assignment_consistency_refit.py`; lightweight redraw in `scripts/reproduce_poster_figures.py` | `outputs/gabriel_task2_assignment_consistency_refit/pca_svf/adjacent_q_ari.csv` | All four exact saved poster images retained |
| 07 | Airfoil cluster means, `K=1–11` | `scripts/plot_gabriel_airfoil_clusters_k1_k11.py` | Gabriel contours plus `K=2–11` assignment arrays and templates | Exact saved poster image retained |
| 08 | Cluster evolution alluvial/Sankey diagram | `scripts/analyze_gabriel_cross_k_cluster_evolution.py` and `scripts/revise_gabriel_sankey_lineage_colours.py` | `K=8–11` labels and contingency tables | Exact saved poster image retained |

The images are collected in `figures/poster/`. Newly generated versions are written to `figures/generated/`.

Mathematical formulas, arrows, headings, and explanatory boxes in the poster are layout elements rather than data figures and therefore do not have separate plotting scripts.

