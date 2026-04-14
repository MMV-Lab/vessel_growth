"""Sample data generator: synthetic hollow vessel with wrinkled walls."""

import numpy as np


def make_sample_vessel():
    """Generate a 3-D synthetic hollow vessel for testing the plugin.

    The vessel is a curved tube with bright, wrinkled walls, a dark hollow
    interior, and dark background — matching the morphology this plugin is
    designed for.

    Returns a list of napari layer-data tuples (image, seed labels, and
    start/end points) so the user can immediately test the widget.
    """
    shape = (80, 160, 160)
    rng = np.random.default_rng(42)

    # Coordinate grids
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    zz = zz.astype(np.float64)
    yy = yy.astype(np.float64)
    xx = xx.astype(np.float64)

    # Vessel centerline: gentle curve through the volume
    # Center goes along z, with a sinusoidal offset in y and x
    t = zz / shape[0]  # 0..1 along z
    center_y = shape[1] / 2 + 15 * np.sin(2 * np.pi * t)
    center_x = shape[2] / 2 + 10 * np.cos(2 * np.pi * t)

    dy = yy - center_y
    dx = xx - center_x
    r = np.sqrt(dy ** 2 + dx ** 2)  # radial distance from axis
    angle = np.arctan2(dy, dx)

    # Wall radii with angular wrinkles
    wrinkle = 2.5 * np.sin(6 * angle + 0.15 * zz)
    inner_radius = 12.0 + wrinkle
    outer_radius = 25.0 + wrinkle * 0.6

    # Vessel wall mask
    wall = (r >= inner_radius) & (r <= outer_radius)

    # Build image
    image = np.zeros(shape, dtype=np.float32)
    image[wall] = 0.85
    # Smooth wall intensity variation
    image[wall] += rng.uniform(-0.1, 0.1, size=int(wall.sum())).astype(np.float32)
    # Background / interior noise
    image += rng.normal(0, 0.04, size=shape).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)

    # --- Seed labels: a thin brush stroke along the centerline ---
    seed = np.zeros(shape, dtype=np.int32)
    for z in range(5, shape[0] - 5):
        frac = z / shape[0]
        cy = int(round(shape[1] / 2 + 15 * np.sin(2 * np.pi * frac)))
        cx = int(round(shape[2] / 2 + 10 * np.cos(2 * np.pi * frac)))
        seed[z, cy - 1 : cy + 2, cx - 1 : cx + 2] = 1

    # --- Start / end points ---
    frac_s = 5 / shape[0]
    frac_e = (shape[0] - 6) / shape[0]
    start = np.array(
        [
            5,
            shape[1] / 2 + 15 * np.sin(2 * np.pi * frac_s),
            shape[2] / 2 + 10 * np.cos(2 * np.pi * frac_s),
        ]
    )
    end = np.array(
        [
            shape[0] - 6,
            shape[1] / 2 + 15 * np.sin(2 * np.pi * frac_e),
            shape[2] / 2 + 10 * np.cos(2 * np.pi * frac_e),
        ]
    )
    points = np.stack([start, end])

    return [
        (image, {"name": "Vessel Image"}, "image"),
        (seed, {"name": "Vessel Seed", "opacity": 0.4}, "labels"),
        (
            points,
            {"name": "Start/End Points", "size": 5, "face_color": "magenta"},
            "points",
        ),
    ]
