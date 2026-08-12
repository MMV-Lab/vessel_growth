"""Napari widget for viewing vessel_analysis_3d morphometrics results.

Loads the CSV/OME-TIFF outputs of a completed vessel_analysis_3d pipeline run
(*_raw_segmentation.ome.tiff, *_skeleton_final.ome.tiff, *_branch_points.ome.tiff,
*_end_points.ome.tiff, *_Segment_Statistics.csv, *_Summary_Statistics.csv) and
renders them as napari layers with global and per-segment toggling.

Deliberately decoupled from vessel_analysis_3d itself: this widget only reads
the standard output file schema, so it works with any pipeline run producing
that schema, without importing vessel_analysis_3d as a dependency.

Segments are drawn as straight lines between their start/end skeleton nodes
(the CSV only stores segment endpoints + aggregate stats, not the full,
possibly-curved voxel path) -- the full true skeleton is still shown as an
always-on reference layer underneath, so this is a labelled overlay on top of
the real geometry, not a replacement for it.

Branch angles are NOT shown as a single per-segment value (a branch point
with N segments has up to N*(N-1)/2 meaningful pairwise angles, not one) --
instead every pairwise angle between segments meeting at a branch point is
drawn geometrically, as an arc between the two segment directions at the
branch point itself, with the degree value labelled on the arc.

Precise diameter along a branch, at a user-chosen step size, is read from
*_Segment_Diameter_Profiles.csv -- a long-format table (one row per skeleton
voxel per segment) that vessel_analysis_3d now exports alongside the
aggregate per-segment stats. Labels are placed at the segment's *real*
skeleton coordinates (not the straight-line approximation the "segments"
layer uses), so they hug the true, possibly-curved path. Results folders
generated before this export existed simply won't have this file; the
step-size control is then disabled with an explanatory tooltip instead of
failing.

vessel_analysis_3d's removeBorderEndPts option strips any true endpoint that
touches the imaged volume's boundary from the official endpoint counts
(Summary/Filament_Statistics) -- almost always the right call, since a
vessel ending exactly at the crop edge is far more likely to just continue
outside the field of view than to be a real anatomical tip, and you can't
tell the two apart from a single volume. Rather than silently discarding
those points, the pipeline now also exports them to
*_border_end_points.ome.tiff, and this widget shows them as their own
"Border-excluded end points" layer (off by default, gray) so you can
inspect what got excluded and why, without it ever affecting the reported
statistics or requiring a pipeline rerun to check.
"""

from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import napari

# branchingAngle intentionally excluded: it is one value per segment (the
# angle vs. whichever neighbor the DFS happened to pick as predecessor), not
# a real per-junction quantity when a branch point has more than 2 segments.
# See the angle-arc layer below for the actual, geometrically-correct
# per-pair angles.
COLORABLE_PROPERTIES = [
    "diameter",
    "minDiameter",
    "maxDiameter",
    "length",
    "straightness",
    "volume",
    "surfaceArea",
]

TABLE_COLUMNS = [
    ("show", None),
    ("segment", None),
    ("diameter", "diameter"),
    ("min", "minDiameter"),
    ("max", "maxDiameter"),
    ("length", "length"),
    ("straightness", "straightness"),
    ("filament", "filamentID"),
]


def _find_basename(folder: Path) -> str:
    matches = sorted(folder.glob("*_Segment_Statistics.csv"))
    if not matches:
        raise FileNotFoundError(f"no *_Segment_Statistics.csv found in {folder}")
    return matches[0].name[: -len("_Segment_Statistics.csv")]


def _read_physical_scale(path: Path) -> Optional[np.ndarray]:
    """Read the (z, y, x) physical voxel size vessel_analysis_3d embeds in
    each OME-TIFF it writes (processing_pipeline.py's PhysicalSizeZ/Y/X),
    so napari can render this anisotropic data (typically ~8um Z vs ~3.5um
    XY) at its true proportions instead of silently defaulting every layer
    to an isotropic [1, 1, 1] scale, which visibly distorts the volume.
    Returns None if the file has no OME metadata or is missing physical
    size (e.g. results from a run without embedded input metadata) --
    callers should then leave napari's isotropic default in place rather
    than guess a scale.
    """
    try:
        with tifffile.TiffFile(str(path)) as tif:
            ome_xml = tif.ome_metadata
    except Exception:
        return None
    if not ome_xml:
        return None
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return None
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    pixels = root.find(".//ome:Pixels", ns)
    if pixels is None:
        return None
    try:
        z = float(pixels.attrib["PhysicalSizeZ"])
        y = float(pixels.attrib["PhysicalSizeY"])
        x = float(pixels.attrib["PhysicalSizeX"])
    except (KeyError, ValueError):
        return None
    return np.array([z, y, x])


def _load_results(folder: Path) -> dict:
    basename = _find_basename(folder)

    def _tiff_path(suffix):
        path = folder / f"{basename}_{suffix}.ome.tiff"
        return path if path.exists() else None

    def _read_tiff(suffix):
        path = _tiff_path(suffix)
        return tifffile.imread(str(path)) if path is not None else None

    def _read_csv(suffix):
        path = folder / f"{basename}_{suffix}.csv"
        return pd.read_csv(path, sep=";") if path.exists() else None

    seg_stats = _read_csv("Segment_Statistics")
    if seg_stats is not None and "segmentID" in seg_stats.columns:
        # segmentID is stored as the string form of a ((z,y,x), (z,y,x)) tuple
        starts, ends = [], []
        for raw in seg_stats["segmentID"]:
            pair = ast.literal_eval(raw) if isinstance(raw, str) else raw
            starts.append(pair[0])
            ends.append(pair[1])
        seg_stats = seg_stats.copy()
        seg_stats["_start"] = starts
        seg_stats["_end"] = ends

    scale = None
    for suffix in ("raw_segmentation", "skeleton_final", "branch_points"):
        path = _tiff_path(suffix)
        if path is not None:
            scale = _read_physical_scale(path)
            if scale is not None:
                break
    if scale is None:
        scale = np.array([1.0, 1.0, 1.0])

    return {
        "basename": basename,
        "scale": scale,
        "raw_segmentation": _read_tiff("raw_segmentation"),
        "skeleton_final": _read_tiff("skeleton_final"),
        "branch_points": _read_tiff("branch_points"),
        "end_points": _read_tiff("end_points"),
        "border_end_points": _read_tiff("border_end_points"),
        "segment_stats": seg_stats,
        "summary_stats": _read_csv("Summary_Statistics"),
        "diameter_profiles": _read_csv("Segment_Diameter_Profiles"),
    }


def _sample_profile_at_step(
    profiles: pd.DataFrame, step_um: float
) -> tuple[np.ndarray, list[str]]:
    """Pick points spaced >= step_um apart (by path position) along each
    segment's diameter profile, for labelling at a chosen density instead of
    every single skeleton voxel."""
    points, labels = [], []
    for _, group in profiles.groupby(["filamentID", "segmentID"], sort=False):
        group = group.sort_values("position_um")
        next_target = 0.0
        for _, row in group.iterrows():
            if row["position_um"] < next_target:
                continue
            points.append([row["z"], row["y"], row["x"]])
            labels.append(f"{row['diameter_um']:.1f} um")
            next_target = row["position_um"] + step_um
    return np.array(points) if points else np.empty((0, 3)), labels


def _slerp(u1: np.ndarray, u2: np.ndarray, t: float, omega: float) -> np.ndarray:
    if omega < 1e-6:
        return u1
    if omega > np.pi - 1e-6:
        # u1 and u2 are antipodal: sin(omega) -> 0 here too, and there is no
        # unique great circle between two antipodal points, so the formula
        # below is undefined. Callers should avoid this range entirely (see
        # _compute_branch_angles), this is a last-resort guard.
        return u1
    return (np.sin((1 - t) * omega) * u1 + np.sin(t * omega) * u2) / np.sin(omega)


def _compute_branch_angles(seg_stats: pd.DataFrame, n_arc_points: int = 12) -> list[dict]:
    """For every branch point, compute the angle between every pair of
    segments meeting there, plus a 3D arc (via slerp between the two segment
    directions) to draw it at the branch point itself -- the "angle between
    two rays at a vertex" a geometry-class diagram would draw, done in 3D.

    Segment direction is approximated as the straight line from the branch
    point to the segment's far endpoint (same approximation the "segments"
    Shapes layer already uses for the segment geometry itself).

    Each result also carries "segment_rows": the (row_index, row_index) pair
    into `seg_stats` (in its iteration order, matching the segment table) of
    the two segments this angle is between -- so a caller can decide this
    angle should only be shown while both of its segments are shown.
    """
    node_far_ends = defaultdict(list)  # node -> list of (far_end, row_index)
    for row_index, (_, row) in enumerate(seg_stats.iterrows()):
        node_far_ends[row["_start"]].append((row["_end"], row_index))
        node_far_ends[row["_end"]].append((row["_start"], row_index))

    results = []
    for node, far_ends in node_far_ends.items():
        if len(far_ends) < 2:
            continue
        node_arr = np.array(node, dtype=float)
        dirs, lens, row_indices = [], [], []
        for far, row_index in far_ends:
            v = np.array(far, dtype=float) - node_arr
            norm = np.linalg.norm(v)
            if norm < 1e-9:
                continue
            dirs.append(v / norm)
            lens.append(norm)
            row_indices.append(row_index)

        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                u1, u2 = dirs[i], dirs[j]
                cos_a = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
                angle_deg = float(np.degrees(np.arccos(cos_a)))
                omega = float(np.arccos(cos_a))
                if omega < 1e-4:
                    # two segments leaving this branch point in essentially
                    # the same direction have no meaningful arc to draw --
                    # _slerp degenerates to a single repeated point here,
                    # which is a zero-length path. Rendered in 3D, vispy's
                    # tube visual normalizes each segment's tangent vector,
                    # so a zero-length one divides by zero (the "invalid
                    # value encountered in divide" warning on switching to
                    # 3D). Skip drawing an arc for it; the near-0deg angle
                    # itself isn't useful to visualize as a shape anyway.
                    continue
                if omega > np.pi - 1e-4:
                    # the opposite extreme: two segments leaving in nearly
                    # opposite directions are antipodal points on the unit
                    # sphere, where sin(omega) -> 0 too. _slerp's formula is
                    # mathematically singular there (an antipodal pair has no
                    # unique great circle, hence no unique interpolation
                    # path), which can produce NaN/extreme arc points -- the
                    # same tube-rendering divide-by-zero this whole function
                    # is written to avoid. Skip it for the same reason.
                    continue
                # keep the arc well inside the shorter of the two segments
                arc_radius = max(1.0, 0.25 * min(lens[i], lens[j]))
                ts = np.linspace(0.0, 1.0, n_arc_points)
                arc_points = np.array(
                    [node_arr + arc_radius * _slerp(u1, u2, t, omega) for t in ts]
                )
                label_pos = node_arr + 1.15 * arc_radius * _slerp(u1, u2, 0.5, omega)
                results.append(
                    {
                        "branch_point": node,
                        "angle_deg": angle_deg,
                        "arc_points": arc_points,
                        "label_pos": label_pos,
                        "segment_rows": (row_indices[i], row_indices[j]),
                    }
                )
    return results


class MorphometricsWidget(QWidget):
    """Widget for viewing vessel_analysis_3d morphometrics results with toggling."""

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer
        self._data: Optional[dict] = None
        self._segments_layer = None
        self._skeleton_layer = None
        self._segmentation_layer = None
        self._branch_pts_layer = None
        self._end_pts_layer = None
        self._border_end_pts_layer = None
        self._diameter_labels_layer = None
        self._angle_arcs_layer = None
        self._angle_labels_layer = None
        self._angle_segment_rows: list[tuple[int, int]] = []
        self._precise_diameter_layer = None
        self._shown_segment_ids: Optional[set] = None

        # Everything lives inside a QScrollArea: this panel has grown a lot
        # of controls (filters, layer toggles, the segment table below all of
        # them), and a napari dock panel does not auto-scroll on its own --
        # without this, a short dock panel can push the table out of view
        # with no way to reach it.
        content = QWidget()
        outer = QVBoxLayout()
        content.setLayout(outer)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        top_layout = QVBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(scroll)
        self.setLayout(top_layout)

        load_row = QHBoxLayout()
        self._load_btn = QPushButton("Load results folder...")
        self._load_btn.clicked.connect(self._on_load_clicked)
        load_row.addWidget(self._load_btn)
        outer.addLayout(load_row)

        self._summary_box = QGroupBox("Summary")
        self._summary_layout = QFormLayout()
        self._summary_box.setLayout(self._summary_layout)
        outer.addWidget(self._summary_box)

        layers_box = QGroupBox("Layers")
        layers_layout = QVBoxLayout()
        layers_box.setLayout(layers_layout)
        self._layer_checkboxes: dict[str, QCheckBox] = {}
        for key, label, default_on in [
            ("segmentation", "Raw segmentation", True),
            ("skeleton", "Full skeleton (reference)", True),
            ("segments", "Segments (toggleable)", True),
            ("branch_points", "Branch points", True),
            ("end_points", "End points", True),
            ("border_end_points", "Border-excluded end points", False),
            ("diameter_labels", "Diameter labels", False),
            ("angles", "Branch angles (geometric)", False),
            ("precise_diameter", "Precise diameter (stepped)", False),
        ]:
            cb = QCheckBox(label)
            cb.setChecked(default_on)
            cb.stateChanged.connect(
                lambda state, k=key: self._set_layer_visible_for_key(k, bool(state))
            )
            self._layer_checkboxes[key] = cb
            layers_layout.addWidget(cb)

        self._diameter_show_range = QCheckBox("  include min/max in diameter labels")
        self._diameter_show_range.setChecked(False)
        self._diameter_show_range.stateChanged.connect(self._refresh_diameter_labels)
        layers_layout.addWidget(self._diameter_show_range)
        outer.addWidget(layers_box)

        filter_box = QGroupBox("Segment filter (updates 'Segments' layer live)")
        filter_layout = QFormLayout()
        filter_box.setLayout(filter_layout)

        self._color_by = QComboBox()
        self._color_by.addItems(COLORABLE_PROPERTIES)
        self._color_by.currentTextChanged.connect(self._apply_filters)
        filter_layout.addRow("Color by", self._color_by)

        self._min_length = QDoubleSpinBox()
        self._min_length.setRange(0, 1e6)
        self._min_length.valueChanged.connect(self._apply_filters)
        filter_layout.addRow("Min length (um)", self._min_length)

        self._min_diameter = QDoubleSpinBox()
        self._min_diameter.setRange(0, 1e6)
        self._min_diameter.valueChanged.connect(self._apply_filters)
        filter_layout.addRow("Min diameter (um)", self._min_diameter)

        outer.addWidget(filter_box)

        self._profile_box = QGroupBox("Precise diameter along branch")
        profile_layout = QFormLayout()
        self._profile_box.setLayout(profile_layout)
        self._step_size = QDoubleSpinBox()
        self._step_size.setRange(0.1, 1e4)
        self._step_size.setValue(10.0)
        self._step_size.setSuffix(" um")
        self._step_size.valueChanged.connect(self._refresh_precise_diameter_labels)
        self._step_size.setEnabled(False)
        self._step_size.setToolTip(
            "No *_Segment_Diameter_Profiles.csv in this results folder -- "
            "load results from a vessel_analysis_3d run new enough to export "
            "per-segment diameter profiles."
        )
        profile_layout.addRow("Step size", self._step_size)
        outer.addWidget(self._profile_box)

        select_row = QHBoxLayout()
        self._select_all_btn = QPushButton("Select all")
        self._select_all_btn.clicked.connect(lambda: self._set_all_rows_checked(True))
        self._deselect_all_btn = QPushButton("Deselect all")
        self._deselect_all_btn.clicked.connect(lambda: self._set_all_rows_checked(False))
        select_row.addWidget(self._select_all_btn)
        select_row.addWidget(self._deselect_all_btn)
        outer.addLayout(select_row)

        self._table = QTableWidget()
        self._table.setColumnCount(len(TABLE_COLUMNS))
        self._table.setHorizontalHeaderLabels([label for label, _ in TABLE_COLUMNS])
        self._table.itemSelectionChanged.connect(self._on_table_row_selected)
        self._table.setMinimumHeight(200)
        outer.addWidget(self._table)

        outer.addStretch()

    # ------------------------------------------------------------------
    def _on_load_clicked(self):
        folder = QFileDialog.getExistingDirectory(self, "Select vessel_analysis_3d results folder")
        if folder:
            self.load_results(Path(folder))

    def load_results(self, folder: Path):
        try:
            self._data = _load_results(folder)
            self._build_layers()
            self._populate_summary()
            self._populate_table()
            self._apply_filters()
            self._apply_all_layer_checkboxes()
        except Exception as exc:
            import traceback

            QMessageBox.critical(
                self,
                "Failed to load results",
                f"Could not fully load {folder}:\n\n{exc}\n\n"
                "See the terminal for the full traceback.",
            )
            traceback.print_exc()
            raise

    # ------------------------------------------------------------------
    def _build_layers(self):
        d = self._data
        scale = d["scale"]
        if d["raw_segmentation"] is not None:
            self._segmentation_layer = self.viewer.add_labels(
                (d["raw_segmentation"] > 0).astype(np.uint8),
                name=f"{d['basename']} segmentation",
                scale=scale,
            )
        if d["skeleton_final"] is not None:
            self._skeleton_layer = self.viewer.add_image(
                d["skeleton_final"] > 0,
                name=f"{d['basename']} skeleton (full)",
                blending="additive",
                colormap="gray",
                scale=scale,
            )
        if d["branch_points"] is not None:
            coords = np.argwhere(d["branch_points"] > 0)
            self._branch_pts_layer = self.viewer.add_points(
                coords,
                name=f"{d['basename']} branch points",
                size=4,
                face_color="red",
                blending="translucent_no_depth",
                scale=scale,
            )
        if d["end_points"] is not None:
            coords = np.argwhere(d["end_points"] > 0)
            self._end_pts_layer = self.viewer.add_points(
                coords,
                name=f"{d['basename']} end points",
                size=4,
                face_color="yellow",
                blending="translucent_no_depth",
                scale=scale,
            )
        if d.get("border_end_points") is not None:
            # true endpoints excluded by removeBorderEndPts (touch the
            # volume boundary -- likely a field-of-view crop artifact, not a
            # real vessel tip). Off by default: these are NOT part of the
            # official endpoint counts in Summary/Filament_Statistics, this
            # is purely "let me look at what got excluded and why".
            coords = np.argwhere(d["border_end_points"] > 0)
            self._border_end_pts_layer = self.viewer.add_points(
                coords,
                name=f"{d['basename']} border-excluded end points",
                size=4,
                face_color="gray",
                blending="translucent_no_depth",
                visible=False,
                scale=scale,
            )

        seg_stats = d["segment_stats"]
        if seg_stats is not None and len(seg_stats) > 0:
            # a zero-length line (start == end) has no well-defined
            # direction; vispy's 3D tube rendering divides by that
            # direction's length, so a degenerate segment can produce the
            # "invalid value encountered in divide" warning (and a
            # rendering glitch) the moment the viewer switches to 3D
            lines, features_rows = [], []
            for _, row in seg_stats.iterrows():
                if tuple(row["_start"]) == tuple(row["_end"]):
                    continue
                lines.append([row["_start"], row["_end"]])
                features_rows.append(row)
            features = pd.DataFrame(features_rows)[
                [c for c in COLORABLE_PROPERTIES if c in seg_stats.columns]
            ].copy()
            self._segments_layer = self.viewer.add_shapes(
                lines,
                shape_type="line",
                name=f"{d['basename']} segments",
                features=features,
                edge_width=3,
                scale=scale,
            )
            self._build_diameter_labels(seg_stats)
            self._build_angle_layers(seg_stats)

        profiles = d.get("diameter_profiles")
        has_profiles = profiles is not None and len(profiles) > 0
        self._step_size.setEnabled(has_profiles)
        self._profile_box.setTitle(
            "Precise diameter along branch"
            if has_profiles
            else "Precise diameter along branch (no profile data in this folder)"
        )
        if has_profiles:
            self._build_precise_diameter_labels()

    def _build_diameter_labels(self, seg_stats: pd.DataFrame):
        midpoints = np.array(
            [
                (np.array(r["_start"], dtype=float) + np.array(r["_end"], dtype=float)) / 2
                for _, r in seg_stats.iterrows()
            ]
        )
        labels_df = pd.DataFrame({"label": self._format_diameter_labels(seg_stats)})
        self._diameter_labels_layer = self.viewer.add_points(
            midpoints,
            name=f"{self._data['basename']} diameter labels",
            size=0,
            features=labels_df,
            text={"string": "{label}", "color": "cyan", "anchor": "center"},
            visible=False,
            blending="translucent_no_depth",
            scale=self._data["scale"],
        )

    def _format_diameter_labels(self, seg_stats: pd.DataFrame) -> list[str]:
        show_range = self._diameter_show_range.isChecked()
        labels = []
        for _, r in seg_stats.iterrows():
            diameter = r.get("diameter", float("nan"))
            if show_range and "minDiameter" in seg_stats.columns and "maxDiameter" in seg_stats.columns:
                labels.append(
                    f"{diameter:.1f} um ({r['minDiameter']:.1f}-{r['maxDiameter']:.1f})"
                )
            else:
                labels.append(f"{diameter:.1f} um")
        return labels

    def _refresh_diameter_labels(self):
        if self._diameter_labels_layer is None or self._data.get("segment_stats") is None:
            return
        seg_stats = self._data["segment_stats"]
        self._diameter_labels_layer.features = pd.DataFrame(
            {"label": self._format_diameter_labels(seg_stats)}
        )
        self._diameter_labels_layer.refresh_text()

    def _build_angle_layers(self, seg_stats: pd.DataFrame):
        angles = _compute_branch_angles(seg_stats)
        self._angle_segment_rows = [a["segment_rows"] for a in angles]
        if not angles:
            return
        arcs = [a["arc_points"] for a in angles]
        self._angle_arcs_layer = self.viewer.add_shapes(
            arcs,
            shape_type="path",
            name=f"{self._data['basename']} branch angles (arcs)",
            edge_color="orange",
            edge_width=1,
            visible=False,
            blending="translucent_no_depth",
            scale=self._data["scale"],
        )
        label_pos = np.array([a["label_pos"] for a in angles])
        labels_df = pd.DataFrame({"label": [f"{a['angle_deg']:.1f}deg" for a in angles]})
        self._angle_labels_layer = self.viewer.add_points(
            label_pos,
            name=f"{self._data['basename']} branch angles (labels)",
            size=0,
            features=labels_df,
            text={"string": "{label}", "color": "orange", "anchor": "center"},
            visible=False,
            blending="translucent_no_depth",
            scale=self._data["scale"],
        )

    def _build_precise_diameter_labels(self):
        profiles = self._data.get("diameter_profiles")
        if profiles is None or len(profiles) == 0:
            return
        if self._shown_segment_ids is not None and "segmentID" in profiles.columns:
            profiles = profiles[profiles["segmentID"].isin(self._shown_segment_ids)]
        points, labels = _sample_profile_at_step(profiles, self._step_size.value())
        labels_df = pd.DataFrame({"label": labels})
        # self._precise_diameter_layer can point at a layer that's no longer
        # actually in the viewer -- e.g. the user removed it by hand via
        # napari's own layer panel, or a previous load_results() call's
        # layer wasn't cleaned up -- so removing it unconditionally can
        # raise "<Points layer ...> is not in list" instead of just
        # replacing it.
        if self._precise_diameter_layer is not None and self._precise_diameter_layer in self.viewer.layers:
            self.viewer.layers.remove(self._precise_diameter_layer)
        self._precise_diameter_layer = self.viewer.add_points(
            points,
            name=f"{self._data['basename']} precise diameter (stepped)",
            size=0,
            features=labels_df,
            text={"string": "{label}", "color": "lime", "anchor": "center"},
            visible=self._layer_checkboxes["precise_diameter"].isChecked(),
            blending="translucent_no_depth",
            scale=self._data["scale"],
        )

    def _refresh_precise_diameter_labels(self):
        if self._data is None or self._data.get("diameter_profiles") is None:
            return
        self._build_precise_diameter_labels()

    def _populate_summary(self):
        while self._summary_layout.rowCount():
            self._summary_layout.removeRow(0)
        summary = self._data.get("summary_stats")
        if summary is None or len(summary) == 0:
            return
        row = summary.iloc[0]
        for col in summary.columns:
            if col == "image":
                continue
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.4g}"
            self._summary_layout.addRow(QLabel(col), QLabel(str(value)))

    def _populate_table(self):
        seg_stats = self._data.get("segment_stats")
        self._table.setRowCount(0 if seg_stats is None else len(seg_stats))
        if seg_stats is None:
            return
        for i, (_, row) in enumerate(seg_stats.iterrows()):
            cb = QCheckBox()
            cb.setChecked(True)
            cb.stateChanged.connect(self._apply_filters)
            self._table.setCellWidget(i, 0, cb)
            short_id = f"{row['_start']} -> {row['_end']}"
            self._table.setItem(i, 1, QTableWidgetItem(short_id))
            for col_index, (_, key) in enumerate(TABLE_COLUMNS[2:], start=2):
                value = row.get(key, "")
                if isinstance(value, float):
                    value = f"{value:.2f}"
                self._table.setItem(i, col_index, QTableWidgetItem(str(value)))
        self._table.resizeColumnsToContents()

    def _set_all_rows_checked(self, checked: bool):
        for i in range(self._table.rowCount()):
            cb = self._table.cellWidget(i, 0)
            if cb is not None:
                cb.blockSignals(True)
                cb.setChecked(checked)
                cb.blockSignals(False)
        self._apply_filters()

    # ------------------------------------------------------------------
    def _layers_for_key(self, key: str) -> list:
        if key == "angles":
            return [self._angle_arcs_layer, self._angle_labels_layer]
        return [
            {
                "segmentation": self._segmentation_layer,
                "skeleton": self._skeleton_layer,
                "segments": self._segments_layer,
                "branch_points": self._branch_pts_layer,
                "end_points": self._end_pts_layer,
                "border_end_points": self._border_end_pts_layer,
                "diameter_labels": self._diameter_labels_layer,
                "precise_diameter": self._precise_diameter_layer,
            }.get(key)
        ]

    def _set_layer_visible_for_key(self, key: str, visible: bool):
        """Apply one checkbox's state to *only* its own layer(s). Deliberately
        scoped this way: a shared handler that re-applied every checkbox to
        every layer on every change would stomp on any layer visibility the
        user set directly in napari's own layer list (e.g. hiding the
        segmentation there) the next time any other checkbox changed."""
        for layer in self._layers_for_key(key):
            if layer is not None:
                layer.visible = visible

    def _apply_all_layer_checkboxes(self):
        """Set every layer's initial visibility from its checkbox. Only call
        this right after (re)building layers -- never wire it to a live
        checkbox signal, or it reintroduces the stomping bug above."""
        for key in self._layer_checkboxes:
            self._set_layer_visible_for_key(key, self._layer_checkboxes[key].isChecked())

    def _compute_shown_mask(self, seg_stats: pd.DataFrame) -> list:
        min_len = self._min_length.value()
        min_dia = self._min_diameter.value()
        shown = []
        for i in range(self._table.rowCount()):
            cb = self._table.cellWidget(i, 0)
            row_checked = cb.isChecked() if cb is not None else True
            length_ok = seg_stats.iloc[i].get("length", 0) >= min_len
            diameter_ok = seg_stats.iloc[i].get("diameter", 0) >= min_dia
            shown.append(bool(row_checked and length_ok and diameter_ok))
        return shown

    def _apply_filters(self):
        if self._segments_layer is None or self._data.get("segment_stats") is None:
            return
        seg_stats = self._data["segment_stats"]
        shown = self._compute_shown_mask(seg_stats)

        # napari Shapes layers don't support per-shape visibility directly;
        # approximate with edge width 0 (invisible) vs the normal width for
        # filtered-out vs visible segments, keeping it a "soft toggle" that
        # preserves spatial context.
        widths = [3 if s else 0 for s in shown]
        try:
            self._segments_layer.edge_width = widths
        except Exception:
            pass

        color_by = self._color_by.currentText()
        if color_by in seg_stats.columns:
            try:
                self._segments_layer.edge_color = color_by
                self._segments_layer.edge_colormap = "viridis"
            except Exception:
                pass

        # bind every label/stat layer to the same per-segment "show" state
        # (checkbox + length/diameter filters) that drives the segments layer
        if self._diameter_labels_layer is not None:
            try:
                self._diameter_labels_layer.shown = shown
            except Exception:
                pass

        if self._angle_segment_rows:
            angle_shown = [
                shown[i] and shown[j] for (i, j) in self._angle_segment_rows
            ]
            if self._angle_arcs_layer is not None:
                try:
                    self._angle_arcs_layer.edge_width = [
                        1 if s else 0 for s in angle_shown
                    ]
                except Exception:
                    pass
            if self._angle_labels_layer is not None:
                try:
                    self._angle_labels_layer.shown = angle_shown
                except Exception:
                    pass

        if "segmentID" in seg_stats.columns:
            shown_mask = np.array(shown, dtype=bool)
            self._shown_segment_ids = set(seg_stats.loc[shown_mask, "segmentID"])
        else:
            self._shown_segment_ids = None
        self._build_precise_diameter_labels()

    def _on_table_row_selected(self):
        if self._segments_layer is None:
            return
        rows = {idx.row() for idx in self._table.selectedIndexes()}
        self._segments_layer.selected_data = rows
