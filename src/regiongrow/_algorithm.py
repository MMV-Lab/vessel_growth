"""Priority-queue region growing with gradient-flux-weighted cost.

Combines three literature-based stopping criteria for robust vessel
boundary detection:

1. **Geodesic cost accumulation** (Fast Marching, Sethian 1996)
   Each voxel has a base traversal cost = 1 / (eps + g(x)) where
   g(x) = exp(-beta |grad I|^2 / kappa^2) is the edge indicator.
   Growth proceeds cheapest-first via a min-heap; accumulated cost
   rises steeply at edges, giving a natural stop.

2. **Gradient flux as soft cost modifier** (inspired by Vasilevskiy &
   Siddiqi 2002)
   flux = grad I . n_outward.   Negative flux means intensity
   decreases outward (wall->background).  Instead of a hard reject, the
   base cost is *multiplied* by (1 + w * max(0, -cos theta)^2).
   This allows growth through wrinkled surfaces where flux is
   transiently negative, while strongly penalising sustained boundary
   crossings where cos theta ~ -1.

3. **Adaptive region statistics** (Confidence Connected / ITK;
   Pohle & Toennies 2001)
   Running mean mu and std sigma maintained with Welford's algorithm.
   Candidates far below the region mean are rejected.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, generate_binary_structure, binary_dilation
from skimage.filters import threshold_otsu, threshold_triangle, threshold_li
import heapq


# ──────────────────────────── helpers ────────────────────────────────────── #

_OFFSETS_6 = np.array(
    [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
    dtype=np.int32,
)


def _precompute_edge_map(image, sigma):
    """Return edge indicator *g*, gradient vector field, magnitude, and kappa.

    Uses a Lorentzian (rational) edge indicator for better dynamic range
    than the exponential form:

        g(x) = 1 / (1 + (|grad I(x)| / kappa)^2)

    kappa is auto-calibrated as the 90th percentile of |grad I|, which
    separates moderate texture/noise from real structural edges.

    The Lorentzian smoothly decreases from 1 (no gradient) to ~0 (strong
    edge) without the saturation problem of the exponential, preserving
    cost discrimination across the full range of edge strengths.
    """
    smoothed = gaussian_filter(image, sigma=sigma)
    grad = np.array(np.gradient(smoothed))          # (3, Z, Y, X)
    grad_mag = np.sqrt(np.sum(grad ** 2, axis=0))

    nonzero = grad_mag[grad_mag > 0]
    kappa = float(np.percentile(nonzero, 90)) if nonzero.size > 0 else 1.0
    kappa = max(kappa, 1e-10)

    g = 1.0 / (1.0 + (grad_mag / kappa) ** 2)
    return g, grad, grad_mag, kappa


def _seed_boundary(seed_mask):
    """6-connected dilation of *seed_mask* minus the seed itself."""
    struct = generate_binary_structure(3, 1)
    dilated = binary_dilation(seed_mask, structure=struct)
    return np.where(dilated & ~seed_mask)


def compute_upper_threshold(image, method):
    """Compute an upper intensity hard-stop threshold.

    Parameters
    ----------
    image : ndarray
        3-D image (any dtype; converted to float64 internally).
    method : str
        One of ``'otsu'``, ``'triangle'``, ``'li'``, ``'p90'``, ``'p95'``.

    Returns
    -------
    float
        Threshold value above which voxels are rejected during growing.
    """
    arr = np.asarray(image, dtype=np.float64)
    try:
        if method == "otsu":
            return float(threshold_otsu(arr))
        if method == "triangle":
            return float(threshold_triangle(arr))
        if method == "li":
            return float(threshold_li(arr))
        if method == "p90":
            return float(np.percentile(arr, 90))
        if method == "p95":
            return float(np.percentile(arr, 95))
        raise ValueError(f"Unknown threshold method: {method!r}")
    except Exception:
        # Fallback: 95th percentile is always computable
        return float(np.percentile(arr, 95))


# ──────────────────────── main entry point ──────────────────────────────── #

def region_grow(
    image,
    seed_mask,
    start_point,
    end_point,
    # ── edge / speed ──
    sigma=2.0,
    # ── stopping gates ──
    cost_budget=None,           # max accumulated cost (auto if None)
    flux_weight=15.0,           # soft flux penalty weight
    intensity_tolerance=3.0,    # reject if intensity < mu - N*sigma
    upper_threshold=None,        # hard stop: reject if intensity > this
    # ── geometry ──
    margin=5.0,
    # ── visualisation ──
    yield_every=500,
):
    """Priority-queue region growing for 3-D vessel segmentation.

    Parameters
    ----------
    image : ndarray (Z, Y, X)
        3-D fluorescence image.
    seed_mask : bool ndarray
        Initial region (user brush stroke along vessel centre).
    start_point, end_point : array-like (z, y, x)
        Vessel start and end coordinates for length constraint.
    sigma : float
        Gaussian smoothing sigma before gradient computation.
        Larger -> ignores small surface wrinkles.
    cost_budget : float or None
        Maximum accumulated geodesic cost a voxel may have to be accepted.
        If *None*, auto-calibrated from the image edge indicator.
    flux_weight : float
        Soft flux penalty weight.  When the normalised gradient flux
        (cos theta) is negative, the local traversal cost is multiplied
        by  ``1 + flux_weight * max(0, -cos_theta)^2``.
        Higher -> stronger penalty at boundary crossings but still allows
        growth through transiently negative-flux wrinkle regions.
    intensity_tolerance : float
        Reject a candidate whose intensity is more than this many standard
        deviations below the adaptive region mean.
    margin : float
        Extra voxel leeway beyond start / end planes.
    yield_every : int
        Yield (step, mask) every *N* accepted voxels for animation.

    Yields
    ------
    (int, ndarray) -- step counter and current boolean mask.
    """

    image = np.asarray(image, dtype=np.float64)
    seed_mask = np.asarray(seed_mask, dtype=bool)

    shape = image.shape

    # ── 1. Pre-compute edge indicator & gradient vector field ───────────
    g, grad, grad_mag, kappa = _precompute_edge_map(image, sigma)

    epsilon = 0.01
    local_cost = 1.0 / (epsilon + g)        # high at edges, ~1 in smooth

    # ── 2. Auto-calibrate cost budget ───────────────────────────────────
    #
    # Strategy: the cost budget should allow traversing the entire vessel
    # cross-section but not much further.  We estimate this as:
    #   budget = (typical edge cost) * (generous radial distance factor)
    # Using the 95th percentile of local_cost ensures we account for
    # real edges, and a multiplier of 50 allows ~50 high-cost steps
    # (traversal through the wall) plus many more smooth-cost steps.
    if cost_budget is None:
        p95 = float(np.percentile(local_cost, 95))
        p50 = float(np.median(local_cost))
        cost_budget = p95 * 30 + p50 * 50

    # ── 3. Axis constraint ──────────────────────────────────────────────
    start = np.asarray(start_point, dtype=np.float64)
    end = np.asarray(end_point, dtype=np.float64)
    axis = end - start
    axis_len = float(np.linalg.norm(axis))
    axis_dir = axis / axis_len if axis_len > 0 else None

    # ── 4. Region mask & Welford running statistics ─────────────────────
    region = seed_mask.copy()
    seed_vals = image[seed_mask]
    n_acc = int(seed_vals.size)
    mu = float(np.mean(seed_vals))
    m2 = float(np.sum((seed_vals - mu) ** 2))

    # ── 5. Accumulated-cost map & min-heap ──────────────────────────────
    acc_cost = np.full(shape, np.inf)
    acc_cost[seed_mask] = 0.0

    heap: list = []
    bz, by, bx = _seed_boundary(seed_mask)
    for i in range(len(bz)):
        z, y, x = int(bz[i]), int(by[i]), int(bx[i])
        c = float(local_cost[z, y, x])
        acc_cost[z, y, x] = c
        heapq.heappush(heap, (c, z, y, x))

    # ── 6. Yield initial state ──────────────────────────────────────────
    yield 0, region.copy()
    step = 0
    last_yielded = 0

    # ── 7. Main loop ─────────────────────────────────────────────
    while heap:
        cost_val, z, y, x = heapq.heappop(heap)

        if region[z, y, x]:
            continue
        if cost_val > acc_cost[z, y, x]:
            continue

        # gate A: geodesic cost budget
        if cost_val > cost_budget:
            break

        step += 1

        # gate B: length constraint
        if axis_dir is not None:
            proj = float(np.dot(
                np.array([z, y, x], dtype=np.float64) - start, axis_dir
            ))
            if proj < -margin or proj > axis_len + margin:
                continue

        # gate C: adaptive intensity
        std = np.sqrt(m2 / max(n_acc, 1)) if n_acc > 1 else 0.0
        val = image[z, y, x]
        if std > 0 and val < mu - intensity_tolerance * std:
            continue

        # gate D: upper threshold hard stop
        if upper_threshold is not None and val > upper_threshold:
            continue

        # ── ACCEPT ──
        region[z, y, x] = True

        # Welford update
        n_acc += 1
        delta = val - mu
        mu += delta / n_acc
        delta2 = val - mu
        m2 += delta * delta2

        # ── Compute flux-weighted cost for neighbours ──
        # Outward normal from segmented-neighbour centroid
        cz_s, cy_s, cx_s, n_seg = 0.0, 0.0, 0.0, 0
        for off in _OFFSETS_6:
            nz, ny, nx = z + off[0], y + off[1], x + off[2]
            if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                if region[nz, ny, nx]:
                    cz_s += nz; cy_s += ny; cx_s += nx
                    n_seg += 1

        for off in _OFFSETS_6:
            nz = z + int(off[0]); ny = y + int(off[1]); nx = x + int(off[2])
            if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                if not region[nz, ny, nx]:
                    base_inc = float(local_cost[nz, ny, nx])

                    # Flux-based soft cost multiplier
                    flux_mult = 1.0
                    if n_seg > 0:
                        dz = nz - cz_s / n_seg
                        dy = ny - cy_s / n_seg
                        dx = nx - cx_s / n_seg
                        norm = np.sqrt(dz * dz + dy * dy + dx * dx)
                        if norm > 1e-10:
                            nz_o = dz / norm; ny_o = dy / norm; nx_o = dx / norm
                            gz = float(grad[0, nz, ny, nx])
                            gy = float(grad[1, nz, ny, nx])
                            gx = float(grad[2, nz, ny, nx])
                            gm = float(grad_mag[nz, ny, nx])
                            if gm > 1e-10:
                                cos_theta = (gz * nz_o + gy * ny_o + gx * nx_o) / gm
                                if cos_theta < 0:
                                    flux_mult = 1.0 + flux_weight * cos_theta * cos_theta

                    nc = cost_val + base_inc * flux_mult
                    if nc < acc_cost[nz, ny, nx]:
                        acc_cost[nz, ny, nx] = nc
                        heapq.heappush(heap, (nc, nz, ny, nx))

        # Yield for animation
        if step % yield_every == 0:
            yield step, region.copy()
            last_yielded = step

    # Always yield final state
    if step != last_yielded:
        yield step, region.copy()
