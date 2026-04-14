"""3-D morphological geodesic active contour for vessel segmentation.

Uses the Morphological Geodesic Active Contour (MGAC) algorithm
(Márquez-Neila et al. 2014):

  - A tube is initialised around the user-drawn seed centerline using
    the Euclidean distance transform.
  - The tube is evolved toward the vessel boundary via morphological
    dilation/erosion guided by the inverse-Gaussian-gradient (IGG) edge
    image.  The IGG gives a speed function that is ~1 in smooth regions
    and ~0 at strong edges, so the contour accelerates away from the
    seed and decelerates (and stops) at the vessel wall.
  - A balloon term (small positive value) keeps the contour inflating
    through weak-edge interior regions.
  - A length constraint clips the evolving mask to the start→end axis,
    mirroring the plain region-growing mode.

Reference:
  P. Márquez-Neila, L. Baumela, and L. Álvarez,
  "A morphological approach to curvature-based evolution of curves and
  surfaces", IEEE TPAMI 2014.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import (
    morphological_geodesic_active_contour,
    inverse_gaussian_gradient,
)


# ─────────────────────────── helpers ─────────────────────────────────────── #

def _init_tube(seed_mask, radius):
    """Dilate *seed_mask* by *radius* voxels (EDT) → initial boolean tube."""
    dist = distance_transform_edt(~seed_mask.astype(bool))
    return dist <= radius


def _length_mask(shape, start, end, margin):
    """Return a boolean volume: True for voxels inside [−margin, len+margin]
    projected along the start→end axis."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    axis = end - start
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 1e-10:
        return np.ones(shape, dtype=bool)
    axis_dir = axis / axis_len

    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    coords = np.stack([
        zz.ravel() - start[0],
        yy.ravel() - start[1],
        xx.ravel() - start[2],
    ], axis=1)                      # (N, 3)
    proj = coords @ axis_dir        # (N,)
    mask = (proj >= -margin) & (proj <= axis_len + margin)
    return mask.reshape(shape)


# ─────────────────────────── main entry point ────────────────────────────── #

def active_contour_grow(
    image,
    seed_mask,
    start_point,
    end_point,
    # ── initialisation ──
    radius=10.0,
    # ── edge image ──
    sigma=2.0,
    # ── MGAC parameters ──
    balloon=0.5,        # >0 = inflate; helps cross weak-edge interior regions
    smoothing=1,        # morphological smoothing steps per iteration
    total_iter=200,     # total evolution iterations
    # ── animation ──
    yield_every=5,      # iterations between display updates
    # ── geometry ──
    margin=5.0,
):
    """3-D morphological geodesic active contour iterator.

    Initialises a tube around *seed_mask* and evolves it toward the
    vessel boundary using the MGAC algorithm.

    Parameters
    ----------
    image : ndarray (Z, Y, X)
        3-D fluorescence image.
    seed_mask : bool ndarray
        User's centerline brush stroke.
    start_point, end_point : array-like (z, y, x)
        Vessel endpoints for the length constraint.
    radius : float
        Initial tube radius in voxels around the seed centerline.
    sigma : float
        Gaussian sigma for the inverse-Gaussian-gradient edge image.
        Larger → ignores fine surface texture.
    balloon : float
        Balloon (inflation) coefficient.  Positive values drive the
        contour outward through smooth interior; should be small enough
        that edge stopping still works (0.3–1.0 is typical).
    smoothing : int
        Number of morphological smoothing iterations per evolution step.
        Higher → smoother final surface.
    total_iter : int
        Total number of MGAC iterations.
    yield_every : int
        Iterations between animation-frame yields (larger = faster but
        coarser animation).
    margin : float
        Extra margin beyond start/end planes along the vessel axis.

    Yields
    ------
    (int, ndarray bool)
        Iteration counter and current boolean segmentation mask.
    """
    image = np.asarray(image, dtype=np.float64)
    seed_mask = np.asarray(seed_mask, dtype=bool)
    shape = image.shape

    # Normalise to [0, 1] — required by inverse_gaussian_gradient
    imin, imax = image.min(), image.max()
    img_norm = (
        (image - imin) / (imax - imin) if imax > imin else np.zeros_like(image)
    )

    # Build gradient edge image:
    #   gimage(x) ≈ 1 / (1 + alpha * |grad(Gauss(I))|^2)
    # High (~1) in smooth regions, low (~0) at strong edges.
    gimage = inverse_gaussian_gradient(img_norm, alpha=100.0, sigma=sigma)

    # Length constraint mask
    lmask = _length_mask(
        shape,
        np.asarray(start_point, dtype=np.float64),
        np.asarray(end_point, dtype=np.float64),
        margin,
    )

    # Initialise tube and clip to length constraint
    ls = _init_tube(seed_mask, radius) & lmask

    yield 0, ls.copy()

    steps_done = 0
    outer_steps = max(1, (total_iter + yield_every - 1) // yield_every)

    for _ in range(outer_steps):
        iters_this = min(yield_every, total_iter - steps_done)
        if iters_this <= 0:
            break

        ls = morphological_geodesic_active_contour(
            gimage,
            num_iter=iters_this,
            init_level_set=ls,
            balloon=balloon,
            smoothing=smoothing,
        )

        # Re-apply length constraint after each evolution step
        ls = np.asarray(ls, dtype=bool) & lmask

        steps_done += iters_this
        yield steps_done, ls.copy()

        if steps_done >= total_iter:
            break
