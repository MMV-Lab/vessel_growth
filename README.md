# Region Grow - 3D Vessel Segmentation for napari

A [napari](https://napari.org) plugin for interactive segmentation of hollow
vessels in 3D fluorescence microscopy.

The plugin uses one workflow for every vessel segment (main trunk or side branch):

- **3D active contour (MGAC)** (default) or **plain region growing** along a user-drawn **polyline** (two or more ordered points in a points layer).
- Each **Grow** writes a preview to **Branch Segmentation**; **Merge** unions that preview into the selected **Segmentation mask** labels layer (often **Segmentation Result**). That mask may be **empty** for the first segment—no separate seed labels layer or two-point-only “start/end” layer.
- Optional **Blocker mask** (labels on the image grid): painted foreground blocks **both** MGAC and Plain growth — useful on whole-organ scans to stop leakage where vessels meet the organ boundary. Use **New Blocker** or choose any matching labels layer in **Blocker mask (optional)**. Optional **Upsample** / **morphology** run after you have a mask on the working grid (merge at least once so the plugin tracks a labels layer for post-processing).

## What this is for

- Hollow vessels with bright walls, darker lumen, and darker background
- Surfaces with folds/wrinkles where simple thresholding leaks
- Users who want interactive seed-and-run segmentation in napari

## Install

Requires **Python ≥ 3.11** and a working **napari** with a Qt backend (e.g. `pip install "napari[all]"`). One install provides the napari plugin, OME-Zarr I/O, preprocessing CLIs, and image → OME-Zarr conversion (OME-TIFF, TIFF, PNG, JPEG, and other [BioIO reader plugins](https://bioio-devs.github.io/bioio/OVERVIEW.html)):

```bash
pip install -e .
```

For development with tests:

```bash
pip install -e ".[test]"
```

Console scripts (also available after install):

- `regiongrow-convert-to-ome-zarr` — any BioIO-readable image → multiscale OME-Zarr
- `regiongrow-preprocess-zarr` — contrast stretch / downsample on OME-Zarr
- `regiongrow-preprocess-ome-tiff` — contrast stretch / downsample on OME-TIFF

## OME-Zarr and large volumes

- **Preprocess before napari:** run contrast stretch (and optional downsample) on disk with **`regiongrow-preprocess-zarr`** or **`regiongrow-preprocess-ome-tiff`**, then open the output in napari. The napari widget does not include in-memory preprocessing.
- **Which reader to pick:** to load an image **and** its saved segmentation together, choose this plugin’s **Read OME-Zarr + segmentation labels** reader for `.ome.zarr` directories. To load only the raw image (no labels), use **[napari-ome-zarr](https://github.com/ome/napari-ome-zarr)**. Both may appear in napari’s open dialog; this plugin’s reader only claims paths ending in `.ome.zarr`.
- This plugin also ships an **OME-TIFF** reader for `*.ome.tif` / `*.ome.tiff` (loads the first series; multi-channel/time stacks use index 0 and emit a warning).
- **RAM vs chunks:** Zarr chunking helps on disk and in napari’s viewer. **Compute Branch** can crop to a polyline ROI (on by default under **Layers**) and optionally cache the pyramid level in session RAM. Grow uses float32. Uncheck ROI to process the full level (slower). If the region is still too large, use a coarser pyramid level or run **`regiongrow-preprocess-zarr`** first.
- Under **Layers**, use **Pyramid level** when the image is multiscale. **Enable multiscale rendering in 2D** (on by default) lets napari switch pyramid levels as you zoom; turn it off to lock the canvas to the selected level (zoom without loading finer data). **Adapt Z step to pyramid level** (on by default) widens the napari Z slider step on coarse levels so each keypress shows a new slice instead of identical subsampled planes (2D with multiscale off, or 3D). **Post-processing → Upsample** zooms a mask to the **finest** resolution when the working grid is coarser than finest (e.g. after **Grow** on a pyramid level).
- **OME-Zarr labels layout:** the raw image pyramid lives at the store root (`0/`, `1/`, …). Segmentations are separate NGFF **labels** groups under **`labels/<name>/`** — each group has its own pyramid arrays (`labels/segmentation/0`, `labels/segmentation/1`, …) and metadata. The store root lists label names in **`labels`** (zarr attrs). This plugin uses **`segmentation`** (first manual save), **`segmentation_v2`**, … (each manual **New version** save), and **`segmentation_autosave`** (overwrite checkpoint after **Merge**). Pick the **`.ome.zarr` root** in load/save dialogs — not a folder inside `labels/`.
- **Saving segmentation:** under the separate **Saving** section, **Save target** = **New version** (default, always creates the next `segmentation_vN`), **Overwrite autosave**, or **Overwrite existing version…** (pick a group). **Save resolution** = working pyramid level or full finest. **Autosave** after **Merge** only touches `segmentation_autosave`.
- **Load saved segmentation…** (under **Layers**) asks which group to load when you select the **`.ome.zarr` root** or **`labels/`**; select **`labels/<name>/`** directly to load that group without the picker. Loading runs in the background and resamples large finest-resolution saves down to your current pyramid level without loading the full volume into RAM.

### Example CLI

```bash
regiongrow-convert-to-ome-zarr volume.ome.tif -o volume.ome.zarr --levels 6
regiongrow-convert-to-ome-zarr stack.tif -o stack.ome.zarr --voxel-size 2.0,0.65,0.65
regiongrow-preprocess-zarr volume.ome.zarr volume_stretched.ome.zarr --stretch --no-downsample
```

## Step-by-step guide

1. Open napari and load a 3D image (single channel, shape Z×Y×X). For **OME-Zarr**, install **napari-ome-zarr** and open the `.ome.zarr` directory.
2. Open the widget: **Plugins → Region Grow Vessel Segmentation**.
3. Select the image in **Layers → Image**. The plugin does **not** add an empty labels layer until your first **Grow** or **Merge** when no labels layer exists on that grid (then it creates **Segmentation Result**).
4. Under **Segmentation**, use **Segmentation mask** to choose the labels layer that receives **Merge** and provides context for **Grow** (e.g. **Segmentation Result**). Use **New Mask** to add another empty labels layer on the grid. An **empty** mask is valid for the **first** segment: **Grow** ORs it with the growing preview.
5. **New BranchPoints Layer** adds a new points layer (**BranchPoints_2**, …). **Reset branch points** clears the layer currently selected in **Branch points layer** (not only the default **BranchPoints** name). Optionally use **Blocker mask (optional)** / **New Blocker** to paint walls on the same pyramid grid as the image (whole-organ leakage).
6. Add **at least two** points in **click order** along one vessel segment. Open **Segmentation parameters** for **Seed tube radius** and **Fill with** (**3D Active Contour** is the default; switch to Plain if needed). Defaults: MGAC **Corridor length margin** 0.15, tube radius 60, smoothing γ 0, 85 iterations; Plain **Length margin** 0. **Compute Branch** runs MGAC/Plain in a **local ROI** around the polyline (padding ≈ 2× tube radius in Z, 1.5× in XY, plus corridor margin). Writes to **Draft_Branch**; if that layer already held a preview and you did not reset or merge, the old mask is renamed to **Draft_Branch (1)**, **(2)**, … and a fresh preview layer is used. Hover the **Segmentation**, **Post-processing**, or **Saving** section titles for workflow notes.
7. When satisfied, **Merge Branch**. Clear or reset points, then repeat for the next segment. Merged masks are **autosaved in the background** to `labels/segmentation_autosave` when the source `.ome.zarr` path is known (~2.5 s after merge).
8. **Reset branch preview** clears **Branch Segmentation** only. To clear a labels mask, edit or delete the layer in napari.
9. For large volumes: pick a coarser **Pyramid level** under **Layers**. Run **`regiongrow-preprocess-zarr`** (or **`regiongrow-preprocess-ome-tiff`**) before opening napari if you need contrast stretch on disk (see example below). **Post-processing → Upsample** and morphology need a merged mask on the working grid; the plugin tracks the last layer you merged into (or **Segmentation Result** if present). Use **Saving** to export labels to OME-Zarr.

**Archiving:** the old **Run** path that renamed **Segmentation Result** to **Result_v*** before each run has been removed. Rename or duplicate layers in napari if you want snapshots.

### GIF capture (growth animation)

Under **Visualization**, enable **Capture animation (GIF)** to record the viewer during each **Grow**. By default, **Skeletal Preview** (the polyline tube before propagation) stays **hidden** in the layer list; show it in napari if you want that overlay.

**One GIF per Grow:** leave **Combine branch grows in one GIF** unchecked. After a successful grow you are prompted for a save path; encoding runs in the background.

**One GIF for several branches:** check **Combine branch grows in one GIF (commit on Merge)**. Each grow is still recorded, but frames are appended to the combined clip only when you click **Merge**. **Reset branch preview** discards the last grow’s recording without merging. Starting a new **Grow** clears any unmerged recording from the previous attempt. When finished, use **Save combined GIF…** (encoding clears the in-memory combined buffer after a successful save).

Playback length follows the number of captured frames and **GIF playback FPS**. **Frame subsample (N)** keeps every *N*-th displayed step (use with **Animate growth** and the Plain / MGAC step controls). **Max frames** caps frames per grow segment. **Capture region** is viewer canvas or full napari window; **Frame scale** shrinks frames before encoding. **GIF canvas width / height** control the pixel size of every frame in the saved GIF (letterboxing): leave both at **Auto** to use the largest width and height seen in that save (enough for a single grow); for **combined GIFs**, set both to the same fixed size (e.g. 960×720) so segments from different window layouts still stack. If **Animate growth** is off but capture is on, the plugin still uses the same step intervals as when animation is on so the recording is usable.

## Technical notes (short)

1. **Coordinates:** Points use `data_to_world` on the points layer and `world_to_data` on the image layer so grids match after contrast-stretched images or layer translation.
2. **Branch seed:** `skimage.draw.line_nd` (or a fallback) rasterizes segment between consecutive knots; `scipy.ndimage.distance_transform_edt` builds the same style of tube as main MGAC. **Branch MGAC** clips evolution with a **polyline corridor** (EDT envelope around the full centerline), not only the straight segment between first and last point, so curved branches are not eroded to an empty mask. The morphological **edge shrink** step would otherwise delete a thin polyline tube in a handful of iterations; branch runs therefore use a **balloon-first warmup** (inflate along the edge image with a low speed threshold, no shrink term), then the full MGAC loop.
3. **MGAC smoothing γ (steps):** skimage morphological curvature (`_curvop`) runs **after balloon+edge in every MGAC iteration**, not once at the end. Values ``> 0`` regularize but **thin** narrow masks each iteration; the UI default is **0** (tune upward per dataset). After each MGAC chunk a **binary closing** (one iteration, physical ball) fills typical **1-voxel stripe / checkerboard** gaps from the discrete edge update.
4. **Blockers:** Choose a labels layer (or **none**) in **Blocker mask (optional)**. Foreground voxels block **Plain** priority-queue growth (non-traversable). For **MGAC**, they zero the edge speed map, are cleared from the level set each iteration, and are stripped from the initial seed before the first displayed frame.
5. **Grow ROI:** When **Crop Compute Branch to polyline ROI** is checked (default), only a bounding box around the polyline is loaded (padding ≈ **2×** seed tube radius in **Z**, **1.5×** in **XY**, plus corridor margin and MGAC σ halo). Optional **Cache pyramid level in session** stores each level in RAM after the first full read (float32).

## Method overview

### Plain region growing

The plain mode is a min-heap front propagation method with multiple stopping
criteria:

1. Edge-weighted local cost.
   Local cost is derived from an edge indicator based on image gradients, so crossing strong edges becomes expensive.
2. Priority-queue expansion.
   Voxels are accepted in increasing accumulated cost order (Dijkstra-style, cheapest-first).
3. Flux penalty.
   Outward gradient flux is used as a soft penalty to discourage wall-to-background leakage while tolerating local wall roughness.
4. Adaptive intensity gate.
   Running region statistics reject candidates that fall too far below the current region intensity model.
5. Length constraint.
   Growth is clipped along the vessel axis implied by the polyline (chord-based margin), plus a margin.

### 3D active contour (MGAC)

The active contour mode initializes a tube around the seed centerline and
evolves it with Morphological Geodesic Active Contours on an inverse-gradient
edge image. A balloon force controls outward/inward bias. **Smoothing steps**
apply skimage morphological curvature **once per outer iteration** (after
balloon and edge updates), not as a final post-process—non-zero smoothing tends
to thin narrow tubes over many iterations. The length constraint clips extent
(polyline **corridor** for MGAC; plain mode uses the same polyline for statistics and axis margin).

### Shared post-processing

After you have merged labels on the working grid:

- Upsampling restores full resolution when the working grid is coarser than the finest image (e.g. **Grow** on a pyramid level).
- Morphological Dilation/Erosion (ball radius in voxels) refines mask shape.
- In anisotropic datasets, a common correction is one Erosion with radius 1
   to remove slight extra thickness along Z while preserving XY quality.

Export to OME-Zarr labels is under **Saving** (separate from upsample/morphology).

## Practical parameter tips

### Plain mode (parameters under **Segmentation → Plain parameters**)

- Use **Blocker mask** if plain growth leaks outside the organ (same optional layer as MGAC).
- Smoothing sigma: start at 2.0; increase for noisy images.
- Flux penalty: increase if leakage occurs; decrease if growth stalls too early.
- Intensity tolerance: increase if true vessel voxels are being rejected.
- Cost budget: keep auto first; increase only when growth stops prematurely.

### Active contour mode (MGAC in **Segmentation parameters**) — default fill method

- Optional **Blocker mask** (same grid / pyramid level as the image): painted voxels block MGAC (see **Segmentation → Blocker mask**).
- **Seed tube radius**: default **60** voxels (finest-isotropic); reduce for very thin vessels.
- **Smoothing γ (steps):** default **0**; increase for stronger per-iteration smoothing (also thins narrow tubes).
- **Total iterations:** default **85**. **Early stop** (MGAC parameters slider): stop after *N* consecutive display updates with an unchanged mask; **0** = run all iterations. Default **2**; increase if growth stops too soon.
- Sigma is a **physical** Gaussian scale (same units as voxel spacing). The UI default **10** suits typical finest-resolution µm spacing; lower it (≈1.5–3.0) for small physical voxels or coarse pyramid levels where 10 over-smooths.
- Balloon: 0.1 to 0.3 for thin vessels or strong edges; 0.5 to 1.0 for weak edges or smoother interiors.

### Morphological post-processing

- Dilation (radius 1 to 2) can fill tiny gaps or connect close fragments.
- Erosion (radius 1 to 2) can remove thin protrusions or boundary noise.
- For anisotropic voxel spacing, start with Erosion radius 1 to clean mild
   Z-direction over-segmentation (often one-voxel too thick in Z).
- Use larger radii cautiously because topology changes quickly in 3D.

## Troubleshooting

- **"seed_mask is empty (no seed voxels to grow from)":** the seed tube did not cover any voxel — increase **Seed tube radius**, check the polyline lies on the vessel, or confirm a **Blocker mask** is not covering the seed.
- **"start_point and end_point coincide…":** place at least two distinct branch points (a double-click on the same spot produces a zero-length axis).
- **"image contains NaN/Inf…":** the working pyramid level has non-finite values; reload, pick another level, or re-run contrast stretch on disk with `regiongrow-preprocess-zarr`.
- **"… does not match any image pyramid level":** the mask grid differs from every image level. Pick the matching **Pyramid level** under **Layers**, or save at **finest** resolution.
- **"… is not an .ome.zarr store" / "store not found":** select the store **root** (`mydata.ome.zarr`) in save/load dialogs, not a folder inside `labels/`. The saver never creates a new store.
- **Out-of-memory on Grow:** enable the **ROI** crop (default), use a **coarser pyramid level**, or preprocess/downsample on disk first. Grow refuses to start past the RAM budget and tells you the estimate.
- **GIF capture stops mid-grow:** capture is RAM-capped (~1.5 GB of frames); reduce **capture scale** or increase **subsample** to cover the whole grow.

## Vessel Morphometrics Viewer

A second widget in the same plugin/install (no separate setup) for browsing the output of a
[vessel_analysis_3d](https://github.com/MMV-Lab/vessel_analysis_3d) run: skeleton, segments,
branch/end points, and per-segment statistics, all in napari.

### Running it

Open napari, then **Plugins → Vessel Morphometrics Viewer**, then **Load results folder...**
and select a `vessel_analysis_3d` run's timestamped output folder (selecting its parent
`output_dir` also works -- it finds the most recent run inside it).

### Features

- **Real skeleton-path segments**, colored by any stat via the *Color by* dropdown (diameter,
  length, straightness, volume, surface area, ...).
- **Live "Current selection" summary**, recomputed on every filter/checkbox change from just
  the currently-shown segments -- separate from the *Summary* panel, which always reflects the
  full, unfiltered run.
- **Segment filters**: min/max length, min/max diameter, per-row checkboxes, and *Hide
  boundary-clipped segments* (on by default) for vessels `vessel_analysis_3d` flagged as cut
  off by the volume edge.
- **Branch angle visualization**: a straight branch-point-to-endpoint reference layer alongside
  the arc, showing exactly what geometry each angle was measured from.
- Every section (Summary, Current selection, Layers, Segment filter, Precise diameter along
  branch) is collapsible.
- Click a row in the segment table to highlight that segment in the viewer; *Select all* /
  *Deselect all* to toggle every row at once.

### Known limitations

- Segment paths only render in napari's 3D view. This is inherent napari behavior, not a bug:
  a `Shapes` path spanning multiple Z slices isn't drawn in 2D slice view unless every vertex
  falls in the current slice, which a traced skeleton path essentially never does. The
  segmentation and skeleton image layers stay visible in 2D regardless.
- The *Precise diameter (stepped)* label layer is the slowest part of any filter change
  (rebuilt from scratch every time, ~1.6s on a real 534-segment run) -- not yet optimized; see
  `_build_precise_diameter_labels` in `_morphometrics_widget.py`.
- Only what a given `vessel_analysis_3d` run actually exported is shown -- a results folder
  from an older run without `*_Segment_Diameter_Profiles.csv` falls back to a straight-line
  approximation for the segments layer instead of the real traced path.

## References

1. Dijkstra EW. A note on two problems in connexion with graphs. Numerische Mathematik. 1959;1:269-271.
2. Vasilevskiy A, Siddiqi K. Flux maximizing geometric flows. IEEE Trans Pattern Anal Mach Intell. 2002;24(12):1565-1578.
3. Marquez-Neila P, Baumela L, Alvarez L. A morphological approach to curvature-based evolution of curves and surfaces. IEEE Trans Pattern Anal Mach Intell. 2014;36(1):2-17.
4. Welford BP. Note on a method for calculating corrected sums of squares and products. Technometrics. 1962;4(3):419-420.

## License

BSD-3-Clause
