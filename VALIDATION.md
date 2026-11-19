# Repository validation

This file records checks performed while assembling the public repository. It distinguishes figure-level verification from full end-to-end reproducibility.

## Status

**PASS for the curated data package and lightweight poster-figure workflow.**

The following commands completed successfully on 19 November 2026:

```bash
python3 scripts/reproduce_poster_figures.py
python3 -m compileall -q src scripts tests
python3 -m unittest discover -s tests -v
shasum -a 256 -c data/checksums.sha256
shasum -a 256 -c ARTIFACTS.sha256
```

The figure command was run twice after visual inspection and correction of the two newly reconstructed charts. It regenerated all expected files in `figures/generated/`. The original cluster-mean and Sankey plotting programs each reported `PASS`. All three retained unit tests passed.

## Included checks

- Poster figures were traced to saved image files and plotting programs in the original research workspace.
- Raw and processed data were copied without modifying the originals.
- The two distinct sample universes (`N=1654` and `N=1652`) are documented rather than combined.
- Direct plot inputs, assignments, templates, PCA/GMM tables, configuration files, and validation reports were retained.
- Large per-airfoil registration checkpoint directories were excluded because they are intermediate caches and are not read by the poster plotting code.
- SHA-256 manifests are included for data and repository artifacts.
- A text-file scan found no credentials, private keys, access tokens, email addresses, or remaining user-specific absolute paths.
- No retained file is larger than 90 MB; the largest retained scientific artifact is the global-mean registration coefficient matrix at approximately 22 MB.

## Reproducibility boundary

Running `scripts/reproduce_poster_figures.py` tests the lightweight figure workflow only. It does not establish that the expensive registration, PCA, and GMM experiments reproduce from scratch on another machine. Those experiment programs and their saved audits are included, but a clean end-to-end rerun has not been claimed.

