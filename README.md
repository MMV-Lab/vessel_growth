# Hollow vessel branch segmentation for napari

Interactive **semi-automatic** segmentation of **hollow vessels** in 3D fluorescence microscopy. You place ordered points along a branch; the plugin grows a 3D mask along that path and you **merge** it into a master labels layer, branch by branch.

The plugin uses one workflow for every vessel segment (main trunk or side branch):

- **3D active contour (MGAC)** (default) or **plain region growing** along a user-drawn **polyline** (two or more ordered points in a points layer).
- Each **Compute Branch** writes a preview to **`Draft_Branch`**; **Merge Branch** unions that preview into the selected **Segmentation mask** labels layer (default name **`Segmentation`**). That mask may be **empty** for the first segment—no separate seed labels layer or two-point-only “start/end” layer.

---

## What this is for

- Bright vessel walls, darker lumen, darker background  
- Folds and wrinkles where global thresholding leaks  
- Large volumes (OME-Zarr pyramids) where you need **human guidance**, not fully automatic whole-organ segmentation  

---

## Install

**Python ≥ 3.11**, napari with a Qt backend (e.g. `pip install "napari[all]"`).

```bash
git clone https://github.com/MMV-Lab/napari-hollow-vessel-segmentation.git
cd napari-hollow-vessel-segmentation
pip install -e .
```

**Console scripts:** `holvesseg-convert-to-ome-zarr`, `holvesseg-preprocess-zarr`, `holvesseg-preprocess-ome-tiff` (see [OME-Zarr and large volumes](#ome-zarr-and-large-volumes)). Open the dock: **Plugins → Hollow Vessel Branch Segmentation**.

---

## OME-Zarr and large volumes

### What Zarr and OME-Zarr are

**Zarr** stores large arrays as **chunked**, **compressed** files on disk. Tools read **regions** (chunks), not necessarily the entire volume at once — essential for whole-organ 3D data.

**OME-Zarr** (NGFF) is the microscopy standard built on Zarr: a **multiscale pyramid** of the same field of view (`0/` = finest, `1/`, `2/`, … = coarser), plus **voxel spacing** in JSON metadata. Datasets usually live in a folder named **`something.ome.zarr`**.

### Prepare data on disk (CLI)

Typical pipeline: **convert** stacks to OME-Zarr, **optionally preprocess** (contrast and/or coarser voxels on disk), then open the store in napari. The **dock widget does not run these steps in memory** — preprocessing keeps RAM use bounded by reading/writing **Zarr chunks** (or TIFF tiles) on disk.


| CLI | Role |
|-----|------|
| `holvesseg-convert-to-ome-zarr` | Build a multiscale pyramid from TIFF, OME-TIFF, etc. |
| `holvesseg-preprocess-zarr` | Stream-read an `.ome.zarr`, optionally mean-downsample and/or contrast-stretch, write a **new** `.ome.zarr` |
| `holvesseg-preprocess-ome-tiff` | Same preprocessing ideas on a single OME-TIFF; convert to Zarr separately |

<details>
<summary><strong>Preprocessing</strong> — what it does and CLI flags</summary>

**What preprocessing is for**

- **Contrast stretch** — maps intensity into a display-friendly range (default output **uint8**). Useful when raw 16-bit stacks look flat in napari or when MGAC edge terms need stronger local contrast. Stretch is **off unless you pass `--stretch`**.
- **Mean downsample** — integer factors in **Z** and/or **XY** shrink the **finest** resolution before coarser pyramid levels are **rebuilt** from that result (`holvesseg-preprocess-zarr` keeps the same NGFF multiscale layout as the input unless you use `--finest-only`).
- **Why a separate CLI** — whole-organ volumes do not fit in RAM; these tools never materialize the full volume at once.

**Workflow notes**

- Output paths must **not exist** unless you pass **`--overwrite`** (Zarr: directory removed and recreated; TIFF: file replaced).
- **`--no-downsample`** — skip downsampling even if `--downsample-z` / `--downsample-xy` are &gt; 1 (common with **`--stretch`** only).
- Downsampling runs only when **`--no-downsample` is absent** and at least one of **`--downsample-z`** or **`--downsample-xy`** is **&gt; 1** (defaults for both are **1** = no shrink).
- After **`holvesseg-preprocess-ome-tiff`**, run **`holvesseg-convert-to-ome-zarr`** on the new TIFF to get a pyramid for napari.

#### `holvesseg-preprocess-zarr`

```text
holvesseg-preprocess-zarr INPUT.ome.zarr OUTPUT.ome.zarr [options]
```

| Flag | Default | Meaning |
|------|---------|---------|
| **`--overwrite`** | off | Delete existing **OUTPUT** and write a new store. |
| **`--downsample-z`** | `1` | Mean downsample factor along **Z** (integer ≥ 1). |
| **`--downsample-xy`** | `1` | Mean downsample factor in **Y** and **X** (same factor for both). |
| **`--no-downsample`** | off | Do not downsample; ignore downsample factors. |
| **`--stretch`** | off | Apply contrast stretch to the finest level (required to enable stretch). |
| **`--stretch-mode`** | `percentile` | `percentile` — limits from global histogram; `fixed` — use **`--fixed-min`** / **`--fixed-max`**. |
| **`--percentile-low`** | `1.0` | Lower percentile for `percentile` mode (0–100). |
| **`--percentile-high`** | `99.0` | Upper percentile for `percentile` mode. |
| **`--fixed-min`** | `0.0` | Background / low clip in `fixed` mode (input intensity units). |
| **`--fixed-max`** | `255.0` | Vessel / high clip in `fixed` mode. |
| **`--out-dtype`** | `uint8` | Output integer dtype after stretch: **`uint8`** or **`uint16`**. |
| **`--finest-only`** | off | Write **only** the finest dataset (no coarser NGFF levels). Legacy / special cases; normal use keeps the full pyramid. |

#### `holvesseg-preprocess-ome-tiff`

Same core flags as **`holvesseg-preprocess-zarr`**, plus TIFF write options. **`--finest-only`** is accepted for CLI parity but **has no effect** on a single-resolution TIFF.

```text
holvesseg-preprocess-ome-tiff INPUT.ome.tif OUTPUT.ome.tif [options]
```

| Flag | Default | Meaning |
|------|---------|---------|
| *(shared)* | — | **`--overwrite`**, **`--downsample-z`**, **`--downsample-xy`**, **`--no-downsample`**, **`--stretch`**, **`--stretch-mode`**, **`--percentile-low`**, **`--percentile-high`**, **`--fixed-min`**, **`--fixed-max`**, **`--out-dtype`** — same as Zarr table above. |
| **`--compression`** | `zlib` | Lossless TIFF compression: **`none`**, **`zlib`**, **`lzw`**, or **`zstd`**. |
| **`--compression-level`** | *(codec default)* | **zlib**: 0–9 (default 6). **zstd**: 1–22 (default 3). Ignored for **`none`** / **`lzw`**. |
| **`--no-predictor`** | off | Disable horizontal differencing (slightly larger files). |
| **`--tile`** | *(auto)* | Optional 3D tile shape for writes, e.g. **`128,128,128`** (Z,Y,X). |

#### `holvesseg-convert-to-ome-zarr` (related)

Not preprocessing, but usually the **first** step. Optional flags:

| Flag | Default | Meaning |
|------|---------|---------|
| **`-o` / `--output`** | `<input_stem>.ome.zarr` | Output directory. |
| **`--image-name`** | from metadata / stem | NGFF image name in metadata. |
| **`--voxel-size`** | from metadata | **`Z,Y,X`** physical size if missing (e.g. **`2.0,0.65,0.65`**). |
| **`--voxel-unit`** | `micrometer` | Unit string for spatial axes. |
| **`--no-downsample-z`** | off | Build pyramid with **XY-only** coarser levels (no extra **Z** halving). |
| **`--levels`** | `3` | Number of pyramid resolutions (level 0 = full); stops when size reaches 1. |
| **`--chunk-target-mib`** | `16` | Target decoded chunk size in MiB. |
| **`--chunks`** | *(auto)* | Fixed chunk shape **`Z,Y,X`** (overrides **`--chunk-target-mib`**). |
| **`--zarr-format`** | `2` | **`2`** = NGFF 0.4 / napari-friendly; **`3`** = NGFF 0.5. |

#### `Examples`

```bash
# 1) Build a multiscale store from TIFF / OME-TIFF / other BioIO formats
holvesseg-convert-to-ome-zarr volume.ome.tif -o volume.ome.zarr --levels 6
holvesseg-convert-to-ome-zarr stack.tif -o stack.ome.zarr --voxel-size 2.0,0.65,0.65

# 2) Optional: new store with contrast stretch only (same voxel grid)
holvesseg-preprocess-zarr volume.ome.zarr volume_stretched.ome.zarr --stretch --no-downsample

# 3) Optional: downsample + stretch (rebuilds pyramid from new finest level)
holvesseg-preprocess-zarr volume.ome.zarr volume_ds.ome.zarr --stretch --downsample-z 2 --downsample-xy 2

# Alternative path: preprocess OME-TIFF first, then convert
holvesseg-preprocess-ome-tiff in.ome.tif out_stretched.ome.tif --stretch --overwrite
holvesseg-convert-to-ome-zarr out_stretched.ome.tif -o out_stretched.ome.zarr
```

</details>

### How this plugin uses an OME-Zarr store

- **Image** — pyramid at the store root (`0/`, `1/`, …). Open with reader **Read OME-Zarr image** (image only; no labels loaded automatically).  
- **Segmentations** — separate NGFF **labels** groups under **`labels/<name>/`**, each with its own pyramid. The raw image is never overwritten.  
- **Names** — manual save: **`segmentation`**, **`segmentation_v2`**, …; **Merge** autosave (when path is known): **`segmentation_autosave`**.  
- **Dialogs** — select the **`.ome.zarr` root**, not an inner `labels/` folder (unless you deliberately load one group via **Load saved segmentation…**).

**Also supported:** OME-TIFF reader (`*.ome.tif` / `*.ome.tiff`); **Load saved segmentation from OME-Zarr…** under **Layers** resamples labels to your current pyramid level when possible. **[napari-ome-zarr](https://github.com/ome/napari-ome-zarr)** can open the same stores — pick the reader that fits your workflow.

### Layers section — pyramid level, RAM, and performance

These controls live under **Layers** in the plugin dock. They govern **how much data** is loaded for **Compute Branch**, not how napari draws the image in general.

| Control | Default | Purpose |
|---------|---------|---------|
| **Pyramid level** | (user choice) | **You choose** the resolution for grow, masks, branch points, and merge. Level **0** = finest. |
| **Crop Compute Branch to polyline ROI** | On | Loads only a bounding box around the branch (plus margin). Faster and less RAM. Turn **off** to process the full level at once (slower). |
| **Cache pyramid level in session** | On | After the first read, keeps the working level in RAM (float32) for faster repeats. |
| **Adapt Z step to pyramid level** | On | On coarse levels, widens the napari **Z** slider step so each step shows a new slice. |
| **Enable multiscale rendering in 2D** | **Off** | See below. |

**Why pick the pyramid level yourself (and leave auto multiscale off)?**

- **3D viewing:** napari does not stream the volume in the same chunked way as on-disk Zarr for all interactions — at a given time you effectively work with **one pyramid level in memory**. You should stay on a level your machine can handle; coarser levels are the main way to stay within RAM.  
- **2D viewing:** your GPU may cope with the finest level, but **automatic pyramid switching while zooming** would change the grid under the image **without** the plugin automatically re-aligning branch points, labels, and grow parameters. Keeping segmentation and algorithms on a **fixed working level** avoids that mismatch. Real-time re-sampling and re-running growth on every zoom level would be **slow and error-prone**.  
- **Recommendation:** leave **Enable multiscale rendering in 2D** **off** for segmentation work. Turn it **on** only for **visual QC** (browsing intensity at different zoom levels) while accepting that grow/merge stay tied to the **Pyramid level** you selected in the dock.

**RAM, chunks, ROI, and seed tube size**

- **Zarr chunking** helps on **disk** and when napari reads OME-Zarr; **Compute Branch** still materializes a **local float32** region (whole level or ROI).  
- If grow runs out of memory: keep **ROI crop** on, use a **coarser Pyramid level**, or **`holvesseg-preprocess-zarr`** to downsample on disk first.  
- A **seed tube radius** that is **too small** can **under-segment** bends and sharp edges — the corridor simply does not cover the lumen. **Fix:** increase **Seed tube radius**, add more branch points on tortuous paths, and/or turn **off ROI crop** if the bounding box cuts off vessel context at corners.  
- **Uncheck ROI** only when you accept the higher RAM cost of loading a larger subvolume or full level.

---

## Workflow

**Coarse first, detail later:** sketch the vessel tree on a **coarse Pyramid level**. Move to finer levels when you want more detail or before final export — **upsampling is optional**, not a required step between every stage.

1. **Setup** — Open **Plugins → Hollow Vessel Branch Segmentation**, then load a 3D volume in napari (OME-Zarr or other).
2. **Coarse level** — **Layers → Pyramid level** → choose a **coarse** level (not finest).
3. **Image & mask** — Select **Image**. Choose or create **Segmentation mask** (empty is OK for the first branch).
4. **(Optional) Blockers** — **New Blocker** if growth leaks at organ boundaries.
5. **One branch** — Add branch points (click order) → **Compute Branch** → edit **`Draft_Branch`** if needed → **Merge Branch**.
6. **More branches at this level?** — **Reset branch points** and **repeat step 5**. With **CleanUp** checked (default), **Merge Branch** already clears archived **`Draft_Branch (N)`** layers and extra **`BranchPoints_*`** layers — you usually just place the next polyline.
7. **Finer detail (optional)** — Switch to a **finer Pyramid level** and grow more branches, and/or use **optional Post-processing** upsampling or morphology if you need a finer grid or cleaner boundaries.
8. **Export** — **Saving → Save segmentation** when satisfied.

---

## Step-by-step (detailed)

1. Open napari. Open the widget: **Plugins → Hollow Vessel Branch Segmentation**.
2. Load a 3D image (single channel, shape Z×Y×X). For **OME-Zarr**, install **napari-ome-zarr** and open the `.ome.zarr` directory. The plugin checks the selected image when you open data or pick a layer: multi-channel Zarr (several `C:0`, `C:1`, … layers or more than three dimensions) shows a warning and disables **Compute Branch** until you use a single-channel volume.
3. Select the image in **Layers → Image**. Selecting an image creates an empty **`Segmentation`** mask on that grid when none exists yet (you can also use **New Mask**).
4. Under **Segmentation**, use **Segmentation mask** to choose the labels layer that receives **Merge Branch** and supplies context during grow (union with the preview). Use **New Mask** for additional empty masks. An **empty** mask is valid for the **first** segment.
5. **New BranchPoints Layer** adds a new points layer (**BranchPoints_2**, …). **Reset branch points** clears the layer currently selected in **Branch points layer** (not only the default **BranchPoints** name). Optionally use **Blocker mask (optional)** / **New Blocker** to paint walls on the same pyramid grid as the image (whole-organ leakage).
6. Add **at least two** points in **click order** along one vessel segment. **Branch point placement:** the polyline is only as faithful as the points you place — **higher tortuosity** (curves, bends, S-shapes) needs **more points** along the centerline so the corridor follows the vessel. For **long branches**, split the vessel into **several shorter grows** (reset points or use **New BranchPoints Layer** between segments) rather than one span from end to end; this helps especially when **diameter changes** a lot along the branch (tapering, bulges, junctions), because seed tube radius and corridor margin are uniform per grow. Open **Segmentation parameters** for **Seed tube radius** and **Fill with** (**3D Active Contour** is the default; switch to Plain if needed). See [Default parameters](#default-parameters) for MGAC and Plain defaults. **Compute Branch** runs MGAC/Plain in a **local ROI** around the polyline (padding ≈ 2× tube radius in Z, 1.5× in XY, plus corridor margin). Writes to **`Draft_Branch`**; if that layer already held a preview and you did not reset or merge, the old mask is renamed to **`Draft_Branch (1)`**, **`(2)`**, … and a fresh preview layer is used. Hover the **Segmentation**, **Post-processing**, or **Saving** section titles for workflow notes.
7. When satisfied, **Merge Branch**. Clear or reset points, then repeat for the next segment. Merged masks are **autosaved in the background** to `labels/segmentation_autosave` when the source `.ome.zarr` path is known (~2.5 s after merge).
8. **Reset Branch** clears **`Draft_Branch`** only (branch points stay). To clear a labels mask, edit or delete the layer in napari.
9. When the coarse pass is complete, move to a **finer Pyramid level** and repeat branch placement and merging for detail, or use **Post-processing → Upsample** / morphology on the working grid (optional). For contrast stretch on disk before opening napari, use the CLI under **OME-Zarr and large volumes → Preprocessing**. Use **Saving** to export labels to OME-Zarr.

---

## Algorithms

### 3D active contour (MGAC) — default

The active contour mode initializes a tube around the seed centerline and evolves it with Morphological Geodesic Active Contours on an inverse-gradient edge image. A balloon force controls outward/inward bias. **Smoothing steps** apply skimage morphological curvature **once per outer iteration** (after balloon and edge updates), not as a final post-process—non-zero smoothing tends to thin narrow tubes over many iterations. MGAC clips evolution to a polyline **corridor**; plain region growing uses a separate **length constraint** along the polyline axis (see Plain region growing).

1. **Seed tube** — polyline between consecutive points; distance transform builds a tube.  
2. **Polyline corridor** — evolution clipped to an envelope around the **full** polyline (curved branches).  
3. **Balloon-first warmup** — avoids losing a thin tube in early shrink steps.  
4. **MGAC loop** — morphological geodesic active contours on an inverse-gradient edge image; balloon force; optional **smoothing γ** each iteration (default **0**).  
5. **Post-chunk closing** — fills typical 1-voxel stripe gaps.

### Plain region growing

Priority-queue expansion with gradient edge cost, flux penalty, adaptive intensity gate, and length constraint along the polyline axis.

1. **Edge-weighted local cost** — Local cost is derived from an edge indicator based on image gradients, so crossing strong edges becomes expensive.
2. **Priority-queue expansion** — Voxels are accepted in increasing accumulated cost order (Dijkstra-style, cheapest-first).
3. **Flux penalty** — Outward gradient flux is used as a soft penalty to discourage wall-to-background leakage while tolerating local wall roughness.
4. **Adaptive intensity gate** — Running region statistics reject candidates that fall too far below the current region intensity model.
5. **Length constraint** — Growth is clipped along the vessel axis implied by the polyline (chord-based margin), plus a margin.

### Blockers

**Blocker_Mask** foreground blocks Plain and zeros MGAC edge speed; cleared from the contour each iteration.

### Coordinates

Branch points and growth share the image grid at the selected **Pyramid level** (spacing-aware tube radius).

---

## Default parameters

Widget defaults at startup (**Seed tube radius** uses finest-isotropic units × min spacing).

### Layers / grow

| Setting | Default |
|---------|---------|
| Crop Compute Branch to polyline ROI | On |
| Cache pyramid level in session | On |
| Enable multiscale rendering in 2D | Off |
| Adapt Z step to pyramid level | On |
| Animate growth | Off |
| Merge → CleanUp | On |

### Segmentation (shared)

| Setting | Default |
|---------|---------|
| Fill with | 3D Active Contour |
| Seed tube radius | **40** |

### 3D Active Contour / MGAC (branch)

| Setting | Default |
|---------|---------|
| Corridor length margin | **0.15** |
| Edge σ (physical) | **3.0** |
| Low clip | **0.0** |
| Balloon | **0.1** |
| Smoothing γ (steps) | **0** |
| Total iterations | **40** |
| Early stop (unchanged steps) | **2** (0 = all iterations) |

### Plain Region Growing (branch)

| Setting | Default |
|---------|---------|
| Smoothing σ | **2.0** |
| Flux penalty | **15.0** |
| Intensity tolerance | **3.0** |
| Cost budget | **0** (= auto) |
| Length margin | **0.0** |
| Upper threshold | Off |

---

## Practical parameter tips

Defaults are listed in [Default parameters](#default-parameters). Below: when to change them.

### 3D Active Contour (MGAC) — default method

- **Seed tube radius (default 40):** increase if bends look **under-segmented** or cut off at sharp edges; decrease only for very thin vessels.  
- **Edge σ (default 3.0):** physical units; **lower** on coarse pyramid levels if edges look over-smoothed.  
- **Balloon (default 0.1):** ~0.1–0.3 for strong edges; higher if growth stalls in weak contrast.  
- **Smoothing γ (default 0):** values **> 0** thin the mask every iteration — use sparingly.  
- **Total iterations (default 40)** / **Early stop (default 2):** set early stop to **0** to force all iterations; raise early stop if the mask stops changing too soon.  
- **Corridor length margin (default 0.15):** slack along the polyline corridor for MGAC.

### Plain Region Growing

- **Smoothing σ (default 2.0):** increase for noisier data.  
- **Flux penalty (default 15.0):** increase if leakage; decrease if growth stops early.  
- **Intensity tolerance (default 3.0):** increase if true lumen voxels are rejected.  
- **Cost budget (default 0 = auto):** increase only if growth stops prematurely.  
- **Length margin (default 0.0):** axis slack for plain grow along the polyline.

### Morphology *(optional post-processing)*

- **Erosion** radius **1** often removes one-voxel **Z** over-thickness in anisotropic data.  
- Large 3D radii change topology quickly.

---

## Branch point tips

- Add a point wherever the vessel **deviates** from a straight line between existing points.  
- **Split** long or tapering branches into multiple **Compute Branch → Merge** cycles.  
- Use **Blocker_Mask** on whole-organ data where vessels meet the organ surface.

---

## Post-processing *(optional)*

Not part of the core branch loop. Use when you want **finer grids**, **full-finest masks**, or **morphological touch-up** after you already have a merged segmentation.

Requires a mask on the **working pyramid grid** — **Merge Branch** at least once so the plugin tracks a layer.

| Control | When to use |
|---------|-------------|
| **Upsample segmentation to finer level…** | You finished at a **coarse pyramid level** and want a **new editable mask** on the **next finer** level (optional step toward more detail). |
| **Upsample Result to Original Size** | Legacy path when working shape ≠ finest and you want one full-finest labels layer via nearest-neighbour zoom. |
| **Dilation / Erosion** | Small 3D corrections (default operation **None**, radius **1**). |

**Saving** (separate dock section) writes the selected **Segmentation mask** to **`labels/…`**.

| Setting | Default / notes |
|---------|------------------|
| **Save resolution** | **Working pyramid level** (low RAM); **Full finest** only when you need full-grid export on disk. |
| **Save target** | **New version** (`segmentation_vN`); or overwrite **`segmentation_autosave`**, loaded group, or choose another store. |

---

## GIF capture, growth animation *(optional)*

Under Visualization, enable Capture animation (GIF) to record the viewer during each Grow. By default, Skeletal Preview (the polyline tube before propagation) stays hidden in the layer list; show it in napari if you want that overlay.

One GIF per Grow: leave Combine branch grows in one GIF unchecked. After a successful grow you are prompted for a save path; encoding runs in the background.

One GIF for several branches: check Combine branch grows in one GIF (commit on Merge). Each grow is still recorded, but frames are appended to the combined clip only when you click Merge. Reset branch preview discards the last grow’s recording without merging. Starting a new Grow clears any unmerged recording from the previous attempt. When finished, use Save combined GIF… (encoding clears the in-memory combined buffer after a successful save).

Playback length follows the number of captured frames and GIF playback FPS. Frame subsample (N) keeps every N-th displayed step (use with Animate growth and the Plain / MGAC step controls). Max frames caps frames per grow segment. Capture region is viewer canvas or full napari window; Frame scale shrinks frames before encoding. GIF canvas width / height control the pixel size of every frame in the saved GIF (letterboxing): leave both at Auto to use the largest width and height seen in that save (enough for a single grow); for combined GIFs, set both to the same fixed size (e.g. 960×720) so segments from different window layouts still stack. If Animate growth is off but capture is on, the plugin still uses the same step intervals as when animation is on so the recording is usable.

---

## Troubleshooting

| Message / issue | What to try |
|-----------------|-------------|
| **seed_mask is empty** | Increase **Seed tube radius**; check points on vessel; check blockers. |
| **Under-segmentation at bends** | Larger **Seed tube radius**; more branch points; disable **ROI crop** if box is too tight. |
| **start_point and end_point coincide** | ≥2 distinct points. |
| **NaN/Inf in image** | Reload; another pyramid level; `holvesseg-preprocess-zarr`. |
| **does not match any image pyramid level** (save) | Match **Pyramid level** or save at **Full finest resolution**. |
| **not an .ome.zarr store** | Select store **root**. |
| **Out-of-memory on grow** | ROI crop on; coarser level; preprocess on disk. |
| **GIF stops mid-grow** | Lower frame scale or increase subsample (see [GIF capture](#gif-capture-growth-animation-optional)). |

---

## References

1. Dijkstra EW. A note on two problems in connexion with graphs. *Numerische Mathematik* 1959;1:269–271.  
2. Vasilevskiy A, Siddiqi K. Flux maximizing geometric flows. *IEEE TPAMI* 2002;24(12):1565–1578.  
3. Marquez-Neila P, Baumela L, Alvarez L. A morphological approach to curvature-based evolution of curves and surfaces. *IEEE TPAMI* 2014;36(1):2–17.  
4. Welford BP. Note on a method for calculating corrected sums of squares and products. *Technometrics* 1962;4(3):419–420.

---

## License

BSD-3-Clause
