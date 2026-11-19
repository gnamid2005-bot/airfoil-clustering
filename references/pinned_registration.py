"""Fixed-correspondence airfoil registration -- a minimal, self-contained demonstration.

A shape is a closed contour of M points with BOTH edges pinned: index 0 at the trailing edge,
index M/2 at the leading edge, and M/2 points at uniform arclength along each surface (upper
first, matching the Selig file traversal). Equal per-surface budgets mean camber cannot slide
the leading edge to a different index. Because every airfoil is parametrised the SAME way,
index i is a correspondence: point i of the source is matched to point i of the target, and the
data term needs no nearest-neighbour search of any kind:

    loss(v) = (1/M) sum_i | Phi_v(x_i) - y_i |^2  +  lambda * ||grad v||^2

Phi_v = exp(v) is a diffeomorphism of the plane: the flow (RK4-integrated) of a stationary
velocity field v laid out on a tensor-product cubic B-spline lattice; the trainable parameters
are the lattice coefficients. The finite-difference Sobolev semi-norm on those coefficients
keeps v smooth. Optimisation is staged LBFGS: cycle 1 runs at lambda_init, every later cycle
re-pins lambda so the regulariser is REG_FRACTION of the current attachment (capped at
lambda_init) -- strong smoothing first, then the leash loosens in proportion to the remaining
misfit.

Default task: register the DATASET MEAN onto the airfoil farthest (corresponded L2) from it --
the template-to-sample direction used in atlas construction. --source/--target take 'mean', an
index, or a .dat name.

Expects the UIUC Selig files in coord_seligFmt/ next to this script.
Dependencies: numpy, torch, matplotlib.

Run:  python pinned_registration.py                       # mean -> farthest airfoil
      python pinned_registration.py --target goe398.dat   # mean -> a chosen airfoil
      python pinned_registration.py --source naca0012.dat --target goe398.dat
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)          # ops here are tiny; thread dispatch costs more than it buys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "coord_seligFmt")
OUTDIR = os.path.join(HERE, "scratch")
EXCLUDE_TOKENS = ("flap", "main-flap", "slat", "30p-30n")

N_PTS = 200                       # contour points per airfoil (M)
GRID_N = 30                       # B-spline control lattice (GRID_N x GRID_N x 2 coefficients)
GRID_XLIM = (-0.05, 1.05)         # lattice extent, chord units
GRID_YLIM = (-0.35, 0.35)
RK_STEPS = 5                     # RK4 substeps integrating the flow over t in [0, 1]
LAMBDA_SMOOTH = 1e-1              # initial smoothness weight (and the cap for later cycles)
REG_FRACTION = 0.2                # later cycles re-pin lambda to this fraction of the attachment
N_CYCLES, ITERS_PER_CYCLE = 8, 100


# --- data: Selig files -> TE/LE-pinned closed loops (equal upper/lower point counts) ------------
def _resample_open(seg, n, include_end=False):
    """Open polyline -> n points at uniform arclength (endpoint excluded unless include_end)."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))])
    s_new = np.linspace(0.0, s[-1], n if include_end else n + 1)[: n]
    return np.stack([np.interp(s_new, s, seg[:, 0]), np.interp(s_new, s, seg[:, 1])], axis=1)


def resample_pinned(coords, n):
    """Closed contour -> n points with index 0 = TE and index n/2 = LE exactly: the contour is
    split at the leading edge (the min-x vertex; Selig order runs TE -> upper -> LE -> lower),
    and each surface gets n/2 uniform-arclength points. Equal budgets per surface keep the LE at
    the same index regardless of camber."""
    assert n % 2 == 0, "n must be even (n/2 points per surface)"
    le = int(np.argmin(coords[:, 0]))
    upper = coords[: le + 1]                              # TE -> LE (upper surface)
    lower = np.vstack([coords[le:], coords[:1]])          # LE -> TE (lower surface, loop closed)
    return np.vstack([_resample_open(upper, n // 2), _resample_open(lower, n // 2)])


def load_airfoils(n_pts=N_PTS):
    """All valid UIUC airfoils as (names, (N, n_pts, 2)); chord-normalised to [0, 1].
    Parsed once, then served from an npz cache beside the script (~0.1s instead of ~30s)."""
    cache = os.path.join(HERE, f"airfoils_cache_{n_pts}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return list(d["names"]), d["contours"]
    names, out = [], []
    for path in sorted(glob.glob(os.path.join(SRC_DIR, "*.dat"))):
        fname = os.path.basename(path)
        if any(tok in fname.lower() for tok in EXCLUDE_TOKENS):
            continue
        with open(path) as f:
            f.readline()
            rows = [(float(p[0]), float(p[1])) for p in
                    (line.split() for line in f) if len(p) == 2]
        try:
            coords = np.asarray(rows, dtype=np.float64)
        except ValueError:
            continue
        if len(coords) < 20 or len(coords) > 5000 or np.abs(coords).max() > 1e3:
            continue
        chord = coords[:, 0].max() - coords[:, 0].min()
        if chord < 0.5 or chord <= 0:
            continue
        coords = (coords - [coords[:, 0].min(), 0.0]) / chord
        if np.abs(coords[:, 1]).max() > 0.5:
            continue
        names.append(fname)
        out.append(resample_pinned(coords, n_pts))
    contours = np.stack(out)
    np.savez(cache, names=np.array(names), contours=contours)
    return names, contours


# --- the warp: B-spline velocity field, exponentiated by RK4 ------------------------------------
def cubic_bspline_1d(t):
    """Cardinal cubic B-spline: partition of unity, support [-2, 2], C^2."""
    a = torch.abs(t)
    inner = (2.0 / 3.0) - a * a + 0.5 * a * a * a
    outer = ((2.0 - a) ** 3) / 6.0
    return torch.where(a < 1.0, inner, torch.where(a < 2.0, outer, torch.zeros_like(t)))


def _local_basis(q, ctrl):
    """Compact support: B3 is zero outside [-2, 2], so each query touches only 4 consecutive
    basis functions per axis. Returns their first index [N] and weights [N, 4]; any candidate
    outside the true support gets weight 0 from B3 itself, so clamping is exact."""
    u = (q - ctrl[0]) / (ctrl[1] - ctrl[0])               # continuous lattice coordinate
    i0 = (torch.floor(u).long() - 1).clamp(0, len(ctrl) - 4)
    idx = i0[:, None] + torch.arange(4)
    return i0, cubic_bspline_1d(u[:, None] - idx)


def velocity(query_xy, coeffs, ctrl_xs, ctrl_ys):
    """v(x) = sum_ij coeffs[i,j] * B((x - x_i)/dx) * B((y - y_j)/dy)   -- [N,2] -> [N,2].
    Evaluated over the supporting 4x4 coefficient patch per point (identical to the dense sum)."""
    ix, bx = _local_basis(query_xy[:, 0], ctrl_xs)
    iy, by = _local_basis(query_xy[:, 1], ctrl_ys)
    patch = coeffs[(ix[:, None] + torch.arange(4))[:, :, None],
                   (iy[:, None] + torch.arange(4))[:, None, :]]           # [N, 4, 4, 2]
    return torch.einsum("ni,nj,nijd->nd", bx, by, patch)


def flow(x0, coeffs, ctrl_xs, ctrl_ys, n_steps=RK_STEPS):
    """Phi(x0) = exp(v)(x0): integrate x' = v(x) over t in [0,1] with RK4."""
    x, dt = x0, 1.0 / n_steps
    for _ in range(n_steps):
        k1 = velocity(x, coeffs, ctrl_xs, ctrl_ys)
        k2 = velocity(x + 0.5 * dt * k1, coeffs, ctrl_xs, ctrl_ys)
        k3 = velocity(x + 0.5 * dt * k2, coeffs, ctrl_xs, ctrl_ys)
        k4 = velocity(x + dt * k3, coeffs, ctrl_xs, ctrl_ys)
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return x


def smoothness(coeffs):
    """Finite-difference Sobolev semi-norm on the lattice ~ ||grad v||^2_L2."""
    return ((coeffs[1:] - coeffs[:-1]) ** 2).sum() + ((coeffs[:, 1:] - coeffs[:, :-1]) ** 2).sum()


# --- staged LBFGS -------------------------------------------------------------------------------
def optimise(att, coeffs):
    print(f"=== Staged LBFGS ({N_CYCLES} cycles x {ITERS_PER_CYCLE} iters; "
          f"reg = {REG_FRACTION:.0%} of attachment, capped at {LAMBDA_SMOOTH:g}) ===")
    for cyc in range(N_CYCLES):
        with torch.no_grad():
            att_now, smooth_now = float(att(coeffs)), float(smoothness(coeffs))
        lam = LAMBDA_SMOOTH if cyc == 0 else \
            min(REG_FRACTION * att_now / max(smooth_now, 1e-12), LAMBDA_SMOOTH)
        print(f"--- cycle {cyc+1}/{N_CYCLES}: att={att_now:.4e} (rms={att_now**0.5:.4e}) "
              f"-> lambda={lam:.4e} ---")
        opt = torch.optim.LBFGS([coeffs], max_iter=ITERS_PER_CYCLE, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-14, tolerance_change=1e-14)
        n = [0]

        def closure():
            opt.zero_grad()
            a = att(coeffs)
            loss = a + lam * smoothness(coeffs)
            loss.backward()
            n[0] += 1
            if n[0] % 25 == 0 or n[0] == 1:
                print(f"  [LBFGS {n[0]:03d}] loss={loss.item():.4e}  att={a.item():.4e} "
                      f"(rms={a.item()**0.5:.4e})")
            return loss

        opt.step(closure)


# --- main ---------------------------------------------------------------------------------------
def resolve(spec, names, contours, mean):
    if spec == "mean":
        return "mean", mean
    if spec.lstrip("-").isdigit():
        return names[int(spec) % len(names)], contours[int(spec) % len(names)]
    hits = [i for i, nm in enumerate(names) if nm in (spec, spec + ".dat")]
    if not hits:
        raise SystemExit(f"airfoil '{spec}' not found")
    return names[hits[0]], contours[hits[0]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="mean", help="'mean' (default), index, or name")
    p.add_argument("--target", default=None,
                   help="index or name (default: the airfoil farthest from the mean)")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    names, contours = load_airfoils()
    mean = contours.mean(0)                               # the correspondence mean
    print(f"{len(names)} airfoils, {N_PTS} pts each")

    src_name, src = resolve(args.source, names, contours, mean)
    if args.target is None:
        d = np.sqrt(((contours - mean[None]) ** 2).sum(-1).mean(1))       # corresponded L2 rms
        tgt_name, tgt = names[int(d.argmax())], contours[int(d.argmax())]
        print(f"target = farthest from mean: {tgt_name} (rms {d.max():.4f} chord; "
              f"median {np.median(d):.4f})")
    else:
        tgt_name, tgt = resolve(args.target, names, contours, mean)
    print(f"registering {src_name} -> {tgt_name}")

    X_src = torch.from_numpy(np.ascontiguousarray(src))
    X_tgt = torch.from_numpy(np.ascontiguousarray(tgt))
    ctrl_xs = torch.from_numpy(np.linspace(*GRID_XLIM, GRID_N))
    ctrl_ys = torch.from_numpy(np.linspace(*GRID_YLIM, GRID_N))
    coeffs = torch.zeros((GRID_N, GRID_N, 2), requires_grad=True)

    def att(c):
        return ((flow(X_src, c, ctrl_xs, ctrl_ys) - X_tgt) ** 2).sum(1).mean()

    rms0 = float(att(coeffs)) ** 0.5
    print(f"initial rms = {rms0:.5f} (chord units)")
    optimise(att, coeffs)
    with torch.no_grad():
        D = flow(X_src, coeffs, ctrl_xs, ctrl_ys).numpy()
    rms1 = float(np.sqrt(((D - tgt) ** 2).sum(1).mean()))
    print(f"\nfinal rms = {rms0:.5f} -> {rms1:.5f}  [{rms0/max(rms1,1e-30):.0f}x closer]")

    os.makedirs(OUTDIR, exist_ok=True)
    tag = f"{src_name.replace('.dat', '')}__{tgt_name.replace('.dat', '')}"
    np.savez(os.path.join(OUTDIR, f"pinned_reg_{tag}.npz"),
             coeffs=coeffs.detach().numpy(), ctrl_xs=ctrl_xs.numpy(), ctrl_ys=ctrl_ys.numpy(),
             rk_steps=RK_STEPS, src_name=src_name, tgt_name=tgt_name,
             src=src, tgt=tgt, deformed=D, rms_before=rms0, rms_after=rms1)
    print(f"saved -> {OUTDIR}/pinned_reg_{tag}.npz")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    for ax, C, ttl, rms in ((axes[0], src, "BEFORE", rms0), (axes[1], D, "AFTER", rms1)):
        ax.plot(np.append(tgt[:, 0], tgt[0, 0]), np.append(tgt[:, 1], tgt[0, 1]), c="#2b6cb0", lw=1)
        ax.plot(np.append(C[:, 0], C[0, 0]), np.append(C[:, 1], C[0, 1]), c="#c1466b", lw=1)
        for i in range(0, N_PTS, N_PTS // 25):            # correspondence whiskers
            ax.plot([C[i, 0], tgt[i, 0]], [C[i, 1], tgt[i, 1]], c="0.75", lw=0.4)
        ax.set_aspect("equal")
        ax.set_title(f"{ttl}: {src_name} (red) vs {tgt_name} (blue)   rms = {rms:.5f}", fontsize=9)
    fig.suptitle("fixed-correspondence airfoil registration", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"pinned_reg_{tag}.png"), dpi=110)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
