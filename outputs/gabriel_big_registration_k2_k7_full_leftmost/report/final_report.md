# Gabriel big registration K=2–7 — full-range leftmost extension

Validation: **PASS**.

No second-stage PCA/GMM was run.

## Branches

| K | q | completed | valid | invalid | failed | mean Dirichlet ± SE | mean SVF magnitude ± SE | wall hours |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 1652 | 1652 | 0 | 0 | 0.021544866 ± 0.00129 | 0.017273023 ± 0.00026 | 0.979 |
| 3 | 89 | 1652 | 1652 | 0 | 0 | 0.013865755 ± 0.000933 | 0.013639424 ± 0.000224 | 0.992 |
| 4 | 5 | 1652 | 1651 | 1 | 0 | 0.022688572 ± 0.00144 | 0.015844487 ± 0.000284 | 0.919 |
| 5 | 3 | 1652 | 1652 | 0 | 0 | 0.018861169 ± 0.00184 | 0.012915727 ± 0.000308 | 0.902 |
| 6 | 2 | 1652 | 1652 | 0 | 0 | 0.017296166 ± 0.00182 | 0.011644447 ± 0.000302 | 0.871 |
| 7 | 5 | 1652 | 1651 | 1 | 0 | 0.017591329 ± 0.00112 | 0.013741235 ± 0.00026 | 0.871 |

Each of the 1,652 targets in each branch was registered from its assigned pointwise cluster-mean Gabriel contour. Six ordered `(1652,1800)` B-spline coefficient matrices are ready for the separate next-stage PCA/GMM task.
