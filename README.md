# Vessel Morphometrics Viewer

A [napari](https://napari.org) widget for browsing the output of a
[vessel_analysis_3d](https://github.com/MMV-Lab/vessel_analysis_3d) run: skeleton, segments,
branch/end points, and per-segment statistics, all in napari.

## Install

Requires **Python >= 3.11** and a working **napari** with a Qt backend (e.g.
`pip install "napari[all]"`).

```bash
pip install -e .
```

## Running it

Open napari, then **Plugins -> Vessel Morphometrics Viewer**, then **Load results folder...**
and select a `vessel_analysis_3d` run's timestamped output folder (selecting its parent
`output_dir` also works -- it finds the most recent run inside it).

## Features

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

## Known limitations

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
