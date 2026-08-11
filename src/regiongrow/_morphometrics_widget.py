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
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import napari

COLORABLE_PROPERTIES = ["diameter", "length", "branchingAngle", "straightness"]


def _find_basename(folder: Path) -> str:
    matches = sorted(folder.glob("*_Segment_Statistics.csv"))
    if not matches:
        raise FileNotFoundError(f"no *_Segment_Statistics.csv found in {folder}")
    return matches[0].name[: -len("_Segment_Statistics.csv")]


def _load_results(folder: Path) -> dict:
    basename = _find_basename(folder)

    def _read_tiff(suffix):
        path = folder / f"{basename}_{suffix}.ome.tiff"
        return tifffile.imread(str(path)) if path.exists() else None

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

    return {
        "basename": basename,
        "raw_segmentation": _read_tiff("raw_segmentation"),
        "skeleton_final": _read_tiff("skeleton_final"),
        "branch_points": _read_tiff("branch_points"),
        "end_points": _read_tiff("end_points"),
        "segment_stats": seg_stats,
        "summary_stats": _read_csv("Summary_Statistics"),
    }


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

        outer = QVBoxLayout()
        self.setLayout(outer)

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
        for key, label in [
            ("segmentation", "Raw segmentation"),
            ("skeleton", "Full skeleton (reference)"),
            ("segments", "Segments (toggleable)"),
            ("branch_points", "Branch points"),
            ("end_points", "End points"),
        ]:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_layer_toggle)
            self._layer_checkboxes[key] = cb
            layers_layout.addWidget(cb)
        outer.addWidget(layers_box)

        filter_box = QGroupBox("Segment filter (updates 'Segments' layer live)")
        filter_layout = QFormLayout()
        filter_box.setLayout(filter_layout)

        self._color_by = None
        from qtpy.QtWidgets import QComboBox

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

        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["show", "segment", "diameter", "length", "angle", "filament"]
        )
        self._table.itemSelectionChanged.connect(self._on_table_row_selected)
        outer.addWidget(self._table)

        outer.addStretch()

    # ------------------------------------------------------------------
    def _on_load_clicked(self):
        folder = QFileDialog.getExistingDirectory(self, "Select vessel_analysis_3d results folder")
        if folder:
            self.load_results(Path(folder))

    def load_results(self, folder: Path):
        self._data = _load_results(folder)
        self._build_layers()
        self._populate_summary()
        self._populate_table()
        self._apply_filters()

    # ------------------------------------------------------------------
    def _build_layers(self):
        d = self._data
        if d["raw_segmentation"] is not None:
            self._segmentation_layer = self.viewer.add_labels(
                (d["raw_segmentation"] > 0).astype(np.uint8),
                name=f"{d['basename']} segmentation",
            )
        if d["skeleton_final"] is not None:
            self._skeleton_layer = self.viewer.add_image(
                d["skeleton_final"] > 0,
                name=f"{d['basename']} skeleton (full)",
                blending="additive",
                colormap="gray",
            )
        if d["branch_points"] is not None:
            coords = np.argwhere(d["branch_points"] > 0)
            self._branch_pts_layer = self.viewer.add_points(
                coords, name=f"{d['basename']} branch points", size=4, face_color="red"
            )
        if d["end_points"] is not None:
            coords = np.argwhere(d["end_points"] > 0)
            self._end_pts_layer = self.viewer.add_points(
                coords, name=f"{d['basename']} end points", size=4, face_color="yellow"
            )

        seg_stats = d["segment_stats"]
        if seg_stats is not None and len(seg_stats) > 0:
            lines = [[row["_start"], row["_end"]] for _, row in seg_stats.iterrows()]
            features = seg_stats[
                [c for c in COLORABLE_PROPERTIES if c in seg_stats.columns]
            ].copy()
            self._segments_layer = self.viewer.add_shapes(
                lines,
                shape_type="line",
                name=f"{d['basename']} segments",
                features=features,
                edge_width=3,
            )

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
            self._table.setItem(i, 2, QTableWidgetItem(f"{row.get('diameter', float('nan')):.2f}"))
            self._table.setItem(i, 3, QTableWidgetItem(f"{row.get('length', float('nan')):.2f}"))
            angle = row.get("branchingAngle", "Null")
            self._table.setItem(i, 4, QTableWidgetItem(str(angle)))
            self._table.setItem(i, 5, QTableWidgetItem(str(row.get("filamentID", ""))))
        self._table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    def _on_layer_toggle(self):
        mapping = {
            "segmentation": self._segmentation_layer,
            "skeleton": self._skeleton_layer,
            "segments": self._segments_layer,
            "branch_points": self._branch_pts_layer,
            "end_points": self._end_pts_layer,
        }
        for key, layer in mapping.items():
            if layer is not None:
                layer.visible = self._layer_checkboxes[key].isChecked()

    def _apply_filters(self):
        if self._segments_layer is None or self._data.get("segment_stats") is None:
            return
        seg_stats = self._data["segment_stats"]
        min_len = self._min_length.value()
        min_dia = self._min_diameter.value()

        shown = []
        for i in range(self._table.rowCount()):
            row_checked = self._table.cellWidget(i, 0).isChecked()
            length_ok = seg_stats.iloc[i].get("length", 0) >= min_len
            diameter_ok = seg_stats.iloc[i].get("diameter", 0) >= min_dia
            shown.append(row_checked and length_ok and diameter_ok)

        # napari Shapes layers don't support per-shape visibility directly;
        # approximate with edge width 0 (invisible) vs the normal width for
        # filtered-out vs visible segments, and fade opacity to keep it as a
        # "soft toggle" that preserves spatial context.
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

    def _on_table_row_selected(self):
        if self._segments_layer is None:
            return
        rows = {idx.row() for idx in self._table.selectedIndexes()}
        self._segments_layer.selected_data = rows
