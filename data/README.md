# Data provenance and contents

## Source

The raw coordinate archive comes from the UIUC Airfoil Data Site Selig-format collection:

`https://m-selig.web.engr.illinois.edu/ads/archives/coord_seligFmt.zip`

The repository owner confirmed that the data and project materials may be included in this public repository. The original archive is preserved at `airfoils/raw/coord_seligFmt.zip`, and the extracted directory contains 1,665 `.dat` files.

`src/download_airfoils.py` records the download URL and performs path-safe ZIP extraction.

## Processed datasets

### Earlier direct-coordinate representation

`airfoils/processed/airfoils_n100_seed0.npz`

- Shape array: `(1654, 200, 2)`
- 100 points on each surface
- Shared cosine-spaced chord coordinates
- Used by the direct-coordinate K-means stability and inertia analysis

The associated preprocessing functions are in `src/airfoil_data.py`.

### Gabriel pinned representation

`airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz`

- Contour array: `(1652, 200, 2)`
- Pinned trailing- and leading-edge correspondence
- Used by the main registration, PCA/GMM, cluster-shape, and cross-`K` analyses

The Gabriel construction inspected 1,665 raw files. The saved development report records that 1,657 passed the initial scalar filter and 1,652 remained after the no-repair geometry validation. The exact parser, validation rules, exclusions, and registration implementation are in `src/gabriel_pinned_registration.py` and `src/run_gabriel_pinned_registration_dev.py`.

## Other retained data artifacts

- `global_mean_gabriel.npy` and `.npz`: pointwise global mean of the 1,652 Gabriel contours.
- `airfoils_n100_seed0_max40.npz`: small development subset retained from the original workspace.
- `outputs/gabriel_globalmean_field_pca_gmm/coefficients/registration_coefficients.npz`: the 22 MB coefficient matrix used by the saved PCA/GMM and assignment-stability analyses.
- Assignment arrays, templates, split indices, and summary tables under `outputs/`.

## Two sample universes

The 1,654-sample and 1,652-sample datasets were created by different preprocessing and validation procedures. There are 1,644 filenames shared uniquely between them, with additional filenames present only in one representation. They are retained as distinct research inputs rather than silently merged.

## Integrity

SHA-256 hashes for the retained files are recorded in `checksums.sha256`. Recompute them from the repository root with:

```bash
find data -type f -not -name checksums.sha256 -exec shasum -a 256 {} \;
```

