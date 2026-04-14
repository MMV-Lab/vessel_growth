# Region Grow - 3D Vessel Segmentation for napari

A [napari](https://napari.org) plugin for interactive segmentation of hollow
vessels in 3D fluorescence microscopy.

The plugin provides two complementary modes:

- Plain region growing (priority-queue geodesic growth with stopping gates)
- 3D active contour (Morphological Geodesic Active Contour, MGAC)

## What this is for

- Hollow vessels with bright walls, darker lumen, and darker background
- Surfaces with folds/wrinkles where simple thresholding leaks
- Users who want interactive seed-and-run segmentation in napari

## Install

```bash
pip install -e .
```

## Essential first-run workflow

1. Open napari and load a 3D image.
2. Open the widget: Plugins -> Region Grow Vessel Segmentation.
3. Create a seed layer and paint a thin centerline stroke.
4. Create a points layer and place exactly two points: start then end.
5. Pick a mode tab: Plain Region Growing or 3D Active Contour.
6. Keep defaults for first run, then click Run.
7. Optional for large images: set Preprocess downsample > 1, create the downsampled image, then draw seed and points on that downsampled layer.
8. Review the first segmentation result.
9. Optional post-processing: use Upsample Result to Original Size (if downsampled) and/or Morphological Post-Processing (Dilation/Erosion with selectable ball radius).

## Method overview

### Plain region growing

The plain mode is a min-heap front propagation method with multiple stopping
criteria:

1. Geodesic edge cost.
   Local cost is derived from an edge indicator based on image gradients, so crossing strong edges becomes expensive.
2. Priority-queue expansion.
   Voxels are accepted in increasing accumulated cost order (Dijkstra/Fast Marching style propagation).
3. Flux penalty.
   Outward gradient flux is used as a soft penalty to discourage wall-to-background leakage while tolerating local wall roughness.
4. Adaptive intensity gate.
   Running region statistics reject candidates that fall too far below the current region intensity model.
5. Length constraint.
   Growth is clipped along the user-defined start-to-end vessel axis plus a margin.

### 3D active contour (MGAC)

The active contour mode initializes a tube around the seed centerline and
evolves it with Morphological Geodesic Active Contours on an inverse-gradient
edge image. A balloon force controls outward/inward bias, smoothing controls
surface regularity, and the same length constraint clips the final extent.

### Shared post-processing

After either segmentation mode:

- Upsampling restores full resolution when preprocessing downsampling was used.
- Morphological Dilation/Erosion (ball radius in voxels) refines mask shape.
- In anisotropic datasets, a common correction is one Erosion with radius 1
   to remove slight extra thickness along Z while preserving XY quality.

## Practical parameter tips

### Plain mode

- Smoothing sigma: start at 2.0; increase for noisy images.
- Flux penalty: increase if leakage occurs; decrease if growth stalls too early.
- Intensity tolerance: increase if true vessel voxels are being rejected.
- Cost budget: keep auto first; increase only when growth stops prematurely.

### Active contour mode

- Initial radius: start near half the vessel diameter.
- Sigma: 1.5 to 3.0 is a good default range for most datasets.
- Balloon:
  0.1 to 0.3 for thin vessels or strong edges;
  0.5 to 1.0 for weak edges or smoother interiors.
- Smoothing steps: 1 to 2 for most cases; increase for smoother boundaries.

### Morphological post-processing

- Dilation (radius 1 to 2) can fill tiny gaps or connect close fragments.
- Erosion (radius 1 to 2) can remove thin protrusions or boundary noise.
- For anisotropic voxel spacing, start with Erosion radius 1 to clean mild
   Z-direction over-segmentation (often one-voxel too thick in Z).
- Use larger radii cautiously because topology changes quickly in 3D.

## References

1. Sethian JA. A Fast Marching Level Set Method for Monotonically Advancing Fronts. Proc Natl Acad Sci U S A. 1996;93(4):1591-1595.
2. Dijkstra EW. A note on two problems in connexion with graphs. Numerische Mathematik. 1959;1:269-271.
3. Vasilevskiy A, Siddiqi K. Flux maximizing geometric flows. IEEE Trans Pattern Anal Mach Intell. 2002;24(12):1565-1578.
4. Marquez-Neila P, Baumela L, Alvarez L. A morphological approach to curvature-based evolution of curves and surfaces. IEEE Trans Pattern Anal Mach Intell. 2014;36(1):2-17.
5. Welford BP. Note on a method for calculating corrected sums of squares and products. Technometrics. 1962;4(3):419-420.

## License

BSD-3-Clause
