# Gabriel big registration K=8–11 — first execution stage

Validation: **PASS**.

No second-stage PCA/GMM was run.

## Branches

| K | q | completed | valid | invalid | failed | mean Dirichlet ± SE | mean SVF magnitude ± SE | wall hours |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 27 | 1652 | 1652 | 0 | 0 | 0.0068436701 ± 0.000762 | 0.0090511362 ± 0.000179 | 0.918 |
| 9 | 14 | 1652 | 1652 | 0 | 0 | 0.0070040629 ± 0.000427 | 0.0098945751 ± 0.000174 | 0.858 |
| 10 | 17 | 1652 | 1652 | 0 | 0 | 0.0071894323 ± 0.000721 | 0.0097966492 ± 0.000186 | 0.862 |
| 11 | 15 | 1652 | 1652 | 0 | 0 | 0.0070811841 ± 0.000835 | 0.0094118942 ± 0.000188 | 0.865 |

Each of the 1,652 targets in each branch was registered from its assigned pointwise cluster-mean Gabriel contour. Four ordered `(1652,1800)` B-spline coefficient matrices are ready for the separate next-stage PCA/GMM task.
