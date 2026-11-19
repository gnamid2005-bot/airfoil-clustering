# Gabriel single-global-mean field PCA/GMM report

## Result

Validation: **PASS**. Valid registrations: 1651; invalid/failed: 1.

The PCA in this experiment was performed on registration-derived vector-field/B-spline coefficients relative to a single common mean, not on raw airfoil shape coordinates.

## Exact method

- Dataset: `data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz`; SHA-256 `f357610c338b7a820c1f09522052d29951e9e62e41d3faffc61de19621995e33`; N=1652; 200 points per contour.
- Global mean: exact pointwise arithmetic mean of all 1652 eligible Gabriel pinned contours.
- Registration: `gabriel_pinned_registration.optimise_registration`; direction global mean → individual target.
- Registration configuration: `{"grid_n": 30, "grid_xlim": [-0.05, 1.05], "grid_ylim": [-0.35, 0.35], "lambda_init": 0.1, "line_search": "strong_wolfe", "max_lbfgs_iterations_per_cycle": 100, "n_cycles": 8, "regularisation_fraction": 0.2, "rk_steps": 5, "tolerance_change": 1e-14, "tolerance_grad": 1e-14}`.
- Coefficients: `(30,30,2)`, C-order flattened to D=1800; axes are control-x, control-y, component [x,y].
- Split: seed 20260813 produced the fixed original 1322/330 partition. Invalid target 513 belonged to held-out and was excluded, so PCA/GMM used 1322 training / 329 held-out valid samples. Both original and valid-used index arrays are saved; PCA was fitted on training only.
- PCA: sklearn PCA, n_components=100, svd_solver=full, centered, no whitening.
- GMM: full covariance, n_init=5, reg_covar=1e-07, max_iter=500, tol=1e-05, init_params=kmeans, random_state=20260813.

## Preferred-K sequences (q=1..100)

- BIC: `2,6,5,5,7,11,9,12,14,16,16,14,13,9,11,11,10,9,11,9,9,9,10,10,10,10,8,11,11,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3`
- ICL: `1,2,2,5,4,11,9,12,14,16,16,14,13,9,11,11,10,9,11,9,9,9,10,10,10,10,8,11,11,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3`
- Held-out LL: `3,9,10,12,14,14,17,17,17,16,16,8,8,6,10,7,4,4,4,4,10,4,7,4,4,4,4,4,4,2,4,2,2,3,3,3,3,3,3,3,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3`

BIC/ICL agreement rate: 96.0%.

Plateaus and leftmost q for K=2..9 are in the tables listed below. No downstream template registration was launched.

Major stable BIC/ICL plateaus are K=8 at q=30–50, K=5 at q=51–67, K=4 at q=68–88, and K=3 at q=89–100. Held-out LL stabilises at K=3 from q=43–100. BIC and ICL differ only at q=1–5 and agree for q=6–100.

Leftmost selections for K=2..9 are: BIC `{2:1, 3:89, 4:68, 5:3, 6:2, 7:5, 8:27, 9:7}`; ICL `{2:2, 3:89, 4:5, 5:4, 6:never, 7:never, 8:27, 9:7}`.

## Outputs

- `coefficients/registration_coefficients.npz`
- `gmm/full_model_selection_grid.csv`
- `gmm/preferred_k_by_q.csv`
- `tables/pca_diagnostics.csv`
- `tables/preferred_k_plateaus.csv`
- `tables/leftmost_q_k2_k9.csv`
- `splits/*_original_1652.npy` and `splits/*_valid_used.npy`
- `figures/preferred_k_field_q1_q100.png` and `.pdf`
- `validation.json`

## Qualitative comparison

The previous pinned-shape figure is methodologically comparable only at the PCA/GMM selection layer; it used flattened coordinates. In that old figure BIC/ICL settled mainly at K=2 from about q=46 onward and held-out LL at K=2 from q=33 onward. The new field-based curves instead pass through long K=8, K=5 and K=4 plateaus before ending at K=3, while held-out LL settles at K=3 from q=43. This is a qualitative representational difference, not evidence that either result is better.
