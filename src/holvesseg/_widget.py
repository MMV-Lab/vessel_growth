"""Napari widget for interactive 3D vessel segmentation via region growing."""

from datetime import datetime
from pathlib import Path
import re
from typing import Any, List, Optional, Tuple, Tuple

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
    QFormLayout,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QFrame,
    QMessageBox,
    QFileDialog,
    QApplication,
    QSlider,
    QInputDialog,
)
from napari.qt.threading import thread_worker
import napari
from napari.utils.colormaps import DirectLabelColormap
from scipy.ndimage import (
    zoom as ndimage_zoom,
    binary_dilation,
    binary_erosion,
    generate_binary_structure,
)

from ._spatial import (
    scale_translate_zyx_from_spatial_kwargs,
    spatial_alignment_for_pyramid_level,
    spatial_alignment_kwargs,
    world_bounds_zyx_for_pyramid_level,
    world_to_voxel_indices_zyx,
)
from ._image_display import (
    copy_multiscale_source_display_to_proxy,
    copy_proxy_display_to_multiscale_source,
)
from ._volume_utils import (
    axis_margin_voxels_for_work_level,
    check_materialization_budget,
    clear_image_level_cache,
    invalidate_image_level_cache,
    default_pyramid_level_index,
    image_finest_shape,
    image_level_data,
    image_level_is_lazy,
    image_level_shape,
    is_multiscale_image_layer,
    labels_pyramid_level_for_image_level,
    layer_data_shape,
    materialize_image_level,
    materialize_image_level_cached,
    materialize_labels_level,
    multiscale_level_count,
    multiscale_level_label,
    multiscale_level_tooltip,
    pyramid_axis_steps,
    pyramid_navigation_axis_ranges_zyx,
    tube_radius_voxels_for_work_level,
    voxel_spacing_zyx_for_level,
)
from ._grow_roi import (
    crop_bool_mask_to_roi,
    grow_roi_slices_zyx,
    paste_roi_mask_into_full,
    polyline_to_roi_local,
    roi_shape_from_slices,
)
from ._image_validation import (
    count_channel_split_image_layers,
    has_blocking_validation_issues,
    looks_like_channel_layer_name,
    summarize_validation_issues,
    validate_image_layer,
)

# Rough RAM cap before Grow (float32 image + masks).
_MAX_MATERIALIZE_GB = 12.0
_MAX_MATERIALIZE_BYTES = _MAX_MATERIALIZE_GB * 1e9
_GROW_WORK_DTYPE = np.float32
_AUTOSAVE_DEBOUNCE_MS = 2500
# Hard RAM ceiling for in-memory GIF capture frames (full-res RGB screenshots
# add up fast); stops appending regardless of the frame-count setting.
_MAX_GIF_CAPTURE_BYTES = 1.5e9

# Separate cap for 3D display materialization (users can switch levels; keep safer default).
_MAX_3D_DISPLAY_GB = 3.0
_MAX_3D_DISPLAY_BYTES = _MAX_3D_DISPLAY_GB * 1e9

_SAVE_TARGET_NEW = 0
_SAVE_TARGET_AUTOSAVE = 1
_SAVE_TARGET_OVERWRITE_LOADED = 2
_SAVE_TARGET_CHOOSE_FOLDER = 3

# Three-layer workflow for vessel branches.
MERGED_SEG_LAYER_NAME = "Merged_Segmentation"
DRAFT_BRANCH_LAYER_NAME = "Draft_Branch"
BLOCKER_MASK_LAYER_NAME = "Blocker_Mask"
THRESHOLD_MASK_LAYER_NAME = "Threshold_Mask"

# Napari color names for segmentation overlays and branch point markers.
LAYER_COLOR_CHOICES: tuple[str, ...] = (
    "magenta",
    "cyan",
    "lime",
    "yellow",
    "orange",
    "deepskyblue",
    "violet",
    "green",
    "red",
    "white",
)

BRANCH_LAYER_COLOR_CHOICES: tuple[str, ...] = tuple(
    c for c in LAYER_COLOR_CHOICES if c != "magenta"
)

DRAFT_BRANCH_COLOR_CHOICES: tuple[str, ...] = BRANCH_LAYER_COLOR_CHOICES

DEFAULT_SEGMENTATION_COLOR = "magenta"
DEFAULT_BRANCH_POINTS_COLOR = "cyan"

BRANCH_POINTS_COLOR_CYCLE: tuple[str, ...] = BRANCH_LAYER_COLOR_CHOICES


def _branch_points_layer_color_index(name: str) -> int:
    """Stable palette slot: ``BranchPoints`` → 0, ``BranchPoints_1`` → 1, …"""
    name = str(name).strip()
    if name == "BranchPoints":
        return 0
    m = re.match(r"^BranchPoints_(\d+)$", name)
    if m:
        return int(m.group(1))
    return 0


def _branch_points_color_for_layer_name(name: str) -> str:
    """All branch point layers share one color (typically only one is active)."""
    _ = name
    return DEFAULT_BRANCH_POINTS_COLOR


def _draft_branch_layer_color_index(name: str) -> int:
    """Stable palette slot: ``Draft_Branch`` → 0, ``Draft_Branch (1)`` → 1, …"""
    name = str(name).strip()
    if name == DRAFT_BRANCH_LAYER_NAME:
        return 0
    m = re.match(r"^Draft_Branch \((\d+)\)$", name)
    if m:
        return int(m.group(1))
    return 0


def _draft_branch_color_for_layer_name(name: str) -> str:
    """Palette slot for live or archived ``Draft_Branch*`` preview labels."""
    idx = _draft_branch_layer_color_index(name)
    return BRANCH_POINTS_COLOR_CYCLE[idx % len(BRANCH_POINTS_COLOR_CYCLE)]


def _draft_branch_archive_color(name: str) -> str:
    """Palette slot for archived ``Draft_Branch (N)`` layers."""
    return _draft_branch_color_for_layer_name(name)


def _sanitize_branch_display_color(color: str) -> str:
    """Branch / draft preview colors never use reserved segmentation magenta."""
    c = str(color).strip()
    if c in DRAFT_BRANCH_COLOR_CHOICES:
        return c
    return DEFAULT_BRANCH_POINTS_COLOR


def _binary_segmentation_colormap(foreground_color: str) -> DirectLabelColormap:
    """Colormap for binary vessel masks (background 0, foreground label 1)."""
    return DirectLabelColormap(
        color_dict={
            0: "transparent",
            1: str(foreground_color),
            None: "transparent",
        },
    )


def _configure_binary_labels_layer(lyr: Any) -> None:
    """Editable defaults for user-painted segmentation / draft preview masks."""
    if not isinstance(lyr, napari.layers.Labels):
        return
    try:
        lyr.editable = True
        lyr.preserve_labels = False
        if int(getattr(lyr, "ndim", 0)) >= 3:
            lyr.n_edit_dimensions = 3
    except Exception:
        pass


def _sync_binary_labels_colormap(lyr: Any, color: str) -> None:
    """Apply mask color without resetting the user's selected label or erase mode."""
    if not isinstance(lyr, napari.layers.Labels):
        return
    try:
        lyr.colormap = _binary_segmentation_colormap(color)
        lyr.colormap.use_selection = bool(getattr(lyr, "show_selected_label", False))
        lyr.colormap.selection = int(getattr(lyr, "selected_label", 1))
    except Exception:
        pass


def _sync_mask_colormap_for_ndisplay(lyr: Any, color: str, *, nd: int) -> None:
    """2D: binary colormap + optional show-selected (eraser). 3D: all foreground labels, no selection."""
    if not isinstance(lyr, napari.layers.Labels):
        return
    _configure_binary_labels_layer(lyr)
    try:
        if nd >= 3:
            data = np.asarray(lyr.data)
            color_dict: dict = {0: "transparent", None: "transparent"}
            for raw in np.unique(data):
                lab = int(raw)
                if lab == 0:
                    continue
                color_dict[lab] = str(color)
            if len(color_dict) <= 2:
                color_dict[1] = str(color)
            lyr.colormap = DirectLabelColormap(
                color_dict=color_dict,
                use_selection=False,
            )
            lyr.show_selected_label = False
            lyr.rendering = "translucent"
        else:
            _sync_binary_labels_colormap(lyr, color)
    except Exception:
        pass


def _init_binary_labels_layer(lyr: Any, color: str) -> None:
    """One-time setup for a new editable binary mask layer."""
    _configure_binary_labels_layer(lyr)
    try:
        lyr.selected_label = 1
    except Exception:
        pass
    _sync_binary_labels_colormap(lyr, color)


def _ensure_labels_show_selected(lyr: Any, *, label: int = 1) -> None:
    """Keep napari's "show selected" enabled for display-only preview layers."""
    if not isinstance(lyr, napari.layers.Labels):
        return
    try:
        lyr.selected_label = int(label)
        lyr.show_selected_label = True
        lyr.colormap.use_selection = True
        lyr.colormap.selection = int(label)
    except Exception:
        pass


def _skeletal_preview_colormap() -> DirectLabelColormap:
    """Only label 1 is drawn; background 0 stays invisible in 3-D volume mode."""
    return DirectLabelColormap(
        color_dict={
            0: "transparent",
            1: "#00cc33",
            None: "transparent",
        },
        use_selection=True,
        selection=1,
    )


def _polyline_indices_for_level(
    poly_fine: np.ndarray,
    image_layer: Any,
    level: int,
    shape_work: tuple,
) -> np.ndarray:
    """Map integer ZYX indices on the finest grid to the chosen pyramid level.

    Uses napari's ``downsample_factors`` for multiscale images so indices match
    the same pyramid convention as the viewer (not only ``shape_work/shape_fine``
    ratios, which can disagree for non-uniform level sizes).
    """
    if poly_fine.size == 0:
        return poly_fine.astype(np.int64)
    sz = tuple(int(x) for x in shape_work)
    from holvesseg._volume_utils import _level_downsample_factors

    df = np.asarray(
        _level_downsample_factors(image_layer, int(level), sz),
        dtype=np.float64,
    ).ravel()
    if df.size >= 3:
        df3 = df[-3:].copy()
    else:
        df3 = np.ones(3, dtype=np.float64)
        df3[-df.size :] = df
    df3[df3 <= 0] = 1.0
    q = np.rint(poly_fine.astype(np.float64) / df3.reshape(1, 3)).astype(
        np.int64
    )
    q[:, 0] = np.clip(q[:, 0], 0, max(sz[0] - 1, 0))
    q[:, 1] = np.clip(q[:, 1], 0, max(sz[1] - 1, 0))
    q[:, 2] = np.clip(q[:, 2], 0, max(sz[2] - 1, 0))
    return q


def _branch_effective_margin(skel: np.ndarray, margin: float, spacing) -> float:
    """Extra length margin for branch grow so the chord axis mask fits curved paths."""
    sk = np.asarray(skel, dtype=bool)
    zz, yy, xx = np.nonzero(sk)
    if zz.size == 0:
        return float(margin)
    s = np.asarray(spacing, dtype=np.float64).ravel()[:3]
    ex = max(
        float(zz.max() - zz.min()) * s[0],
        float(yy.max() - yy.min()) * s[1],
        float(xx.max() - xx.min()) * s[2],
    )
    mean_s = float(np.mean(s))
    return float(margin) + 0.45 * ex + 3.5 * mean_s


def _voxel_spacing_zyx(image_layer: Any) -> tuple:
    """``(s_z, s_y, s_x)`` from the napari image layer (last three scale entries)."""
    s = np.asarray(image_layer.scale, dtype=np.float64).ravel()
    if s.size < 3:
        return (1.0, 1.0, 1.0)
    s = s[-3:].copy()
    s[s <= 0] = 1.0
    return tuple(float(x) for x in s)


def _branch_points_to_working_grid_zyx(
    image_layer: Any,
    points_layer: Any,
    level: int,
    shape_work: tuple,
) -> np.ndarray:
    """Map branch points to integer ZYX indices on the working pyramid grid.

    Uses world coordinates and the same per-level ``scale`` / ``translate`` as
    segmentation overlays (:func:`spatial_alignment_for_pyramid_level`), so face
    slices (first/last Z) align with what the user sees at the selected pyramid
    level — not finest indices rescaled by downsample factors.
    """
    pts = np.asarray(points_layer.data, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int64)
    skw = spatial_alignment_for_pyramid_level(image_layer, int(level))
    translate, scale = scale_translate_zyx_from_spatial_kwargs(skw)
    worlds: List[np.ndarray] = []
    for i in range(pts.shape[0]):
        row = np.asarray(pts[i], dtype=np.float64).ravel()
        if row.size < 3:
            raise ValueError("Each point must have at least 3 coordinates.")
        worlds.append(
            np.asarray(points_layer.data_to_world(row[:3]), dtype=np.float64).ravel()[
                :3
            ]
        )
    return world_to_voxel_indices_zyx(
        np.stack(worlds, axis=0),
        translate=translate,
        scale=scale,
        shape=shape_work,
    )


def _is_auto_sized_branch_points_name(name: str) -> bool:
    """Layers we resize for visibility (default naming from the plugin / README)."""
    return name == "BranchPoints" or name.startswith("BranchPoints_")


def _is_blocker_labels_name(name: str) -> bool:
    """Default blocker layer names from **New Blocker**."""
    if name == BLOCKER_MASK_LAYER_NAME:
        return True
    if name.startswith(f"{BLOCKER_MASK_LAYER_NAME}_"):
        return True
    return name == f"{BLOCKER_MASK_LAYER_NAME}_extra"


def _is_branch_draft_labels_name(name: str) -> bool:
    """Live or archived branch preview layers (never a merge target)."""
    name = str(name).strip()
    if name == DRAFT_BRANCH_LAYER_NAME:
        return True
    return name.startswith(f"{DRAFT_BRANCH_LAYER_NAME} (")


def _is_segmentation_mask_numbered_name(name: str) -> bool:
    """``Segmentation`` or ``Segmentation_<n>`` (plugin default mask names)."""
    name = str(name).strip()
    if name == "Segmentation":
        return True
    return bool(re.match(r"^Segmentation_\d+$", name))


def _segmentation_mask_name_sort_key(name: str) -> tuple[int, int]:
    """Order masks: ``Segmentation``, then ``Segmentation_1``, ``Segmentation_2``, …"""
    if name == "Segmentation":
        return (0, 0)
    m = re.match(r"^Segmentation_(\d+)$", str(name).strip())
    if m:
        return (1, int(m.group(1)))
    return (2, 0)


def _is_merge_target_labels_name(name: str) -> bool:
    """Labels layers eligible as branch merge / grow-context mask."""
    name = str(name).strip()
    if _is_branch_draft_labels_name(name):
        return False
    if name == "Skeletal Preview":
        return False
    if _is_blocker_labels_name(name):
        return False
    if name == THRESHOLD_MASK_LAYER_NAME or name.startswith("Threshold_Mask"):
        return False
    return True


def _preferred_draft_branch_layer_name(names: List[str]) -> Optional[str]:
    """Default branch preview for merge: live ``Draft_Branch``, else latest archive."""
    ordered = [str(n) for n in names if n]
    if DRAFT_BRANCH_LAYER_NAME in ordered:
        return DRAFT_BRANCH_LAYER_NAME
    archived: List[tuple[int, str]] = []
    prefix = f"{DRAFT_BRANCH_LAYER_NAME} ("
    for n in ordered:
        if not n.startswith(prefix) or not n.endswith(")"):
            continue
        try:
            k = int(n[len(prefix) : -1])
        except ValueError:
            continue
        archived.append((k, n))
    if archived:
        return max(archived, key=lambda t: t[0])[1]
    return ordered[0] if ordered else None


def _preferred_merge_target_layer_name(names: List[str]) -> Optional[str]:
    """Pick a sensible default when the trunk combo loses its selection."""
    ordered = [str(n) for n in names if n]
    numbered = sorted(
        (n for n in ordered if _is_segmentation_mask_numbered_name(n)),
        key=_segmentation_mask_name_sort_key,
    )
    if numbered:
        return numbered[0]
    for legacy in (MERGED_SEG_LAYER_NAME, "Segmentation Result", "Mask"):
        if legacy in ordered:
            return legacy
    for n in ordered:
        if n.startswith("Mask_"):
            return n
    return ordered[0] if ordered else None


def _ensure_blocker_labels_ndim3(lyr: Any) -> None:
    """Keep blocker masks editable in 3-D (paint through Z, not a single 2-D plane)."""
    try:
        if int(getattr(lyr, "ndim", 0)) >= 3:
            lyr.n_edit_dimensions = 3
    except Exception:
        pass


def _suggested_branch_point_base_size(
    image_layer: Any, pyramid_level: int = 0
) -> float:
    """Marker diameter in Points *data* coordinates for a readable default on huge volumes.

    Sized for the working pyramid grid so markers stay consistent in world units
    across levels (coarser grids use smaller data-space radii).
    """
    shape = image_finest_shape(image_layer)
    if len(shape) >= 3:
        m = float(min(int(shape[-3]), int(shape[-2]), int(shape[-1])))
    elif len(shape) >= 1:
        m = float(min(int(s) for s in shape))
    else:
        m = 128.0
    base_finest = float(max(22.0, min(m * 0.020, 420.0)))
    lvl = int(pyramid_level)
    if lvl <= 0:
        return base_finest
    fz, fy, fx = pyramid_axis_steps(image_layer, lvl)
    f_min = float(min(fz, fy, fx))
    return float(max(8.0, base_finest / max(f_min, 1.0)))


# Max width for the plugin dock content (prevents napari from widening the sidebar).
_DOCK_CONTENT_MAX_WIDTH = 340
_DOCK_COMBO_MAX_WIDTH = 236


def _configure_form_layout(form: QFormLayout) -> None:
    """Keep form fields from forcing the dock wider than the content cap."""
    form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
    form.setFormAlignment(Qt.AlignTop | Qt.AlignLeft)
    form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)


def _configure_dock_combo(
    cb: QComboBox, *, min_chars: int = 12, max_width: int = _DOCK_COMBO_MAX_WIDTH
) -> None:
    """Limit combo size hints so long layer names / shapes do not expand the dock."""
    cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    cb.setMinimumContentsLength(int(min_chars))
    cb.setMaximumWidth(int(max_width))
    cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


def _configure_dock_spin(spin: QWidget, *, max_width: int = 120) -> None:
    spin.setMaximumWidth(int(max_width))
    spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


def _elided_layer_combo_text(name: str, max_len: int = 34) -> str:
    name = str(name)
    if len(name) <= max_len:
        return name
    return name[: max(1, max_len - 1)] + "…"


def _combo_layer_name(combo: QComboBox) -> str:
    """Full layer name from combo item data (falls back to displayed text)."""
    idx = int(combo.currentIndex())
    if idx >= 0:
        data = combo.itemData(idx)
        if data is not None and str(data).strip():
            return str(data)
    return combo.currentText()


def _combo_find_layer_name(combo: QComboBox, name: str) -> int:
    if not name:
        return -1
    for i in range(combo.count()):
        data = combo.itemData(i)
        if data is not None and str(data) == name:
            return i
        if combo.itemText(i) == name:
            return i
    return -1


def _select_combo_layer(combo: QComboBox, layer_name: str) -> None:
    idx = _combo_find_layer_name(combo, layer_name)
    if idx >= 0:
        combo.setCurrentIndex(idx)


def _apply_spatial_kwargs_to_layer(lyr: Any, skw: dict) -> None:
    """Update *lyr* transform metadata from ``spatial_alignment_*`` dict."""
    for key in ("scale", "translate", "rotate", "shear", "units"):
        if key in skw:
            setattr(lyr, key, skw[key])


def _update_labels_layer_data(lyr: Any, mask: Any) -> None:
    """Assign label volume and notify napari/VisPy to redraw."""
    new = np.asarray(mask)
    if new.dtype == bool or new.dtype == np.bool_:
        new_i = new.astype(np.int32, copy=False)
    elif new.dtype != np.int32:
        new_i = new.astype(np.int32, copy=False)
    else:
        new_i = new
    cur = getattr(lyr, "data", None)
    if (
        isinstance(cur, np.ndarray)
        and cur.shape == new_i.shape
        and cur.dtype == np.int32
    ):
        if np.array_equal(cur, new_i):
            return
        np.copyto(cur, new_i)
        # Reassign through the setter — in-place copy + events alone does not
        # refresh 3D label textures until visibility is toggled.
        lyr.data = cur
        return
    lyr.data = np.asarray(new_i, dtype=np.int32, order="C")


def _tip(text):
    """Return a styled circular '?' label that shows *text* on hover."""
    lbl = QLabel("?")
    lbl.setFixedSize(16, 16)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setCursor(Qt.WhatsThisCursor)
    lbl.setToolTip(text)
    lbl.setStyleSheet(
        "QLabel { border: 1px solid #888; border-radius: 8px;"
        " color: #555; font-size: 9px; font-weight: bold;"
        " background: #f5f5f5; }"
        "QLabel:hover { background: #ddd; }"
    )
    return lbl


def _row(widget, tip_text=None):
    """Wrap *widget* with a fixed-size '?' badge at its right edge.

    *tip_text* defaults to the widget's own toolTip() when omitted,
    so callers only need to call setToolTip() once.
    The widget stretches; the badge stays compact.
    """
    box = QHBoxLayout()
    box.setContentsMargins(0, 0, 0, 0)
    box.addWidget(widget, 1)
    box.addWidget(
        _tip(tip_text if tip_text is not None else widget.toolTip()), 0
    )
    return box


def _collapsible_section(
    title: str,
    inner: QWidget,
    *,
    start_open: bool = True,
    header_tooltip: Optional[str] = None,
) -> QWidget:
    """Return a widget with a clickable header that shows or hides *inner*."""
    outer = QWidget()
    v = QVBoxLayout(outer)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(2)
    toggle = QToolButton()
    toggle.setText(title)
    if header_tooltip:
        toggle.setToolTip(header_tooltip)
    toggle.setCheckable(True)
    toggle.setChecked(start_open)
    toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    toggle.setArrowType(Qt.DownArrow if start_open else Qt.RightArrow)
    toggle.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    toggle.setStyleSheet(
        "QToolButton { text-align: left; font-weight: bold; padding: 2px; }"
    )
    frame = QFrame()
    fl = QVBoxLayout(frame)
    fl.setContentsMargins(4, 0, 4, 6)
    fl.addWidget(inner)

    def _on_toggled(checked: bool) -> None:
        frame.setVisible(checked)
        toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    toggle.toggled.connect(_on_toggled)
    frame.setVisible(start_open)
    v.addWidget(toggle)
    v.addWidget(frame)
    return outer


class HolvessegWidget(QWidget):
    """Widget for 3D vessel segmentation via polyline-based plain / MGAC grow and merge."""

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        from . import apply_macos_multiprocessing_workaround

        apply_macos_multiprocessing_workaround()
        self.viewer = napari_viewer
        self._worker = None
        self._result_layer = None
        self._preview_layer = None
        self._image_working_metadata: dict = {}
        self._pending_branch_grow_cleanup = False
        self._active_branch_job = None
        self._branch_pts_sync = None
        self._branch_step_target_layer = None
        self._grow_view_transition = False
        self._pending_grow_step_result: Optional[Any] = None
        self._ndisplay_sync_timer = QTimer(self)
        self._ndisplay_sync_timer.setSingleShot(True)
        self._ndisplay_sync_timer.setInterval(50)
        self._ndisplay_sync_timer.timeout.connect(
            self._finish_deferred_ndisplay_sync
        )
        self._merge_postprocess_timer = QTimer(self)
        self._merge_postprocess_timer.setSingleShot(True)
        self._merge_postprocess_timer.setInterval(50)
        self._merge_postprocess_timer.timeout.connect(
            self._run_deferred_merge_postprocess
        )
        self._merge_postprocess_payload: Optional[dict] = None
        self._merge_layers_mutating = False
        self._merge_ndisplay_deferred = False
        self._labels_colormap_resync_timer = QTimer(self)
        self._labels_colormap_resync_timer.setSingleShot(True)
        self._labels_colormap_resync_timer.setInterval(120)
        self._labels_colormap_resync_timer.timeout.connect(
            self._resync_mask_colormaps_after_ndisplay
        )
        self._labels_show_selected_2d: dict[str, bool] = {}
        self._ndisplay_sync_pending = False
        self._growth_capture_frames: List[np.ndarray] = []
        self._growth_capture_step_counter = 0
        self._growth_capture_finalized = False
        self._growth_capture_hit_max = False
        self._growth_capture_bytes = 0
        self._gif_encode_worker = None
        self._gif_capture_combined_frames: List[np.ndarray] = []
        self._gif_capture_pending_segment: List[np.ndarray] = []
        self._gif_capture_combine_this_run = False
        self._skeletal_preview_vispy_registered = False
        self._branch_point_size_bases: dict = {}
        self._layer_display_colors: dict[str, str] = {}
        self._color_target: str = "segmentation"
        self._syncing_color_combo = False
        self._branch_point_zoom_ref: Optional[float] = None
        self._forced_3d_display_layer_name: Optional[str] = None
        self._forced_2d_display_layer_name: Optional[str] = None
        self._forced_3d_proxy_level: Optional[int] = None
        self._forced_2d_proxy_level: Optional[int] = None
        self._last_pyramid_level: Optional[int] = None
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._run_debounced_autosave)
        self._autosave_worker = None
        self._autosave_pending: Optional[tuple] = None
        self._save_segmentation_worker = None
        self._loaded_segmentation_source: Optional[Tuple[str, str]] = None
        self._load_segmentation_worker = None
        self._dims_nav_override = False
        self._saved_dims_range: Optional[tuple] = None
        self._dims_nav_applying = False
        self._pyramid_dims_nav_timer = QTimer(self)
        self._pyramid_dims_nav_timer.setSingleShot(True)
        self._pyramid_dims_nav_timer.setInterval(16)
        self._pyramid_dims_nav_timer.timeout.connect(self._apply_pyramid_dims_navigation)
        self._camera_points_timer = QTimer(self)
        self._camera_points_timer.setSingleShot(True)
        self._camera_points_timer.setInterval(33)
        self._camera_points_timer.timeout.connect(
            self._apply_camera_branch_point_sizes
        )
        self._refresh_layers_timer = QTimer(self)
        self._refresh_layers_timer.setSingleShot(True)
        self._refresh_layers_timer.setInterval(50)
        self._refresh_layers_timer.timeout.connect(self._refresh_layers_now)
        self._forced_3d_proxy_pinned: Optional[str] = None
        self._validation_dialog_keys: set[str] = set()

        self._build_ui()
        self._sync_save_combined_gif_button()
        self._connect_signals()
        self._on_branch_ac_early_stop_slider_changed(
            self.branch_ac_early_stop_slider.value()
        )
        self._refresh_layers_now()
        self._on_ndisplay_changed()
        self._update_image_validation_ui(show_dialog=True)
        _init_img = self._get_image_layer()
        if _init_img is not None:
            self._ensure_default_segmentation_mask_for_image(_init_img, select=True)
        self._update_save_target_combo_state()

        # Keep combos in sync with layer changes
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        for lyr in list(self.viewer.layers):
            self._hook_layer_dims_nav_events(lyr)
        try:
            self.viewer.dims.events.range.connect(
                self._on_dims_range_changed_for_pyramid_nav
            )
            self.viewer.dims.events.point.connect(
                self._on_dims_point_changed_for_pyramid_nav
            )
            self.viewer.dims.events.current_step.connect(
                self._on_dims_point_changed_for_pyramid_nav
            )
        except AttributeError:
            pass
        # 2D canvas + empty Points can trigger VisPy glTexSubImage2D(height=0); hide those layers.
        try:
            self.viewer.dims.events.ndisplay.connect(self._on_ndisplay_changed)
        except AttributeError:
            pass

    def _layer_color(self, layer_name: str, default: str) -> str:
        return str(self._layer_display_colors.get(str(layer_name), default))

    def _set_layer_color(self, layer_name: str, color: str) -> None:
        self._layer_display_colors[str(layer_name)] = str(color).strip()

    def _register_branch_points_layer_color(self, layer_name: str) -> str:
        color = _branch_points_color_for_layer_name(layer_name)
        self._set_layer_color(layer_name, color)
        return color

    def _apply_branch_points_palette_color(self, layer_name: str) -> None:
        """Apply the shared branch-points color (all layers use cyan)."""
        if not _is_auto_sized_branch_points_name(layer_name):
            return
        color = _branch_points_color_for_layer_name(layer_name)
        self._set_layer_color(layer_name, color)
        self._apply_color_to_layer_name(layer_name, color)

    def _sync_all_branch_points_palette_colors(self) -> None:
        for lyr in self.viewer.layers:
            if isinstance(lyr, napari.layers.Points) and _is_auto_sized_branch_points_name(
                lyr.name
            ):
                self._apply_branch_points_palette_color(str(lyr.name))

    def _draft_branch_labels_color(self, name: str) -> str:
        """Color for a branch preview labels layer (live or archived)."""
        return _draft_branch_color_for_layer_name(name)

    def _sync_draft_branch_labels_colors(self) -> None:
        for lyr in self.viewer.layers:
            if not isinstance(lyr, napari.layers.Labels):
                continue
            nm = str(lyr.name)
            if not _is_branch_draft_labels_name(nm):
                continue
            color = _draft_branch_color_for_layer_name(nm)
            self._set_layer_color(nm, color)
            self._apply_stored_labels_color(lyr, color)

    def _reset_branch_workflow_colors_after_merge(self) -> None:
        """Restart branch palette at cyan (BranchPoints + Draft_Branch)."""
        default = DEFAULT_BRANCH_POINTS_COLOR
        for nm in ("BranchPoints", DRAFT_BRANCH_LAYER_NAME):
            self._set_layer_color(nm, default)
            if nm in self.viewer.layers:
                self._apply_color_to_layer_name(nm, default)
        for key in list(self._layer_display_colors.keys()):
            if _is_auto_sized_branch_points_name(key):
                if key != "BranchPoints":
                    self._layer_display_colors.pop(key, None)
            elif _is_branch_draft_labels_name(key) and key != DRAFT_BRANCH_LAYER_NAME:
                self._layer_display_colors.pop(key, None)

    def _register_segmentation_layer_color(
        self, layer_name: str, *, color: Optional[str] = None
    ) -> str:
        c = color or DEFAULT_SEGMENTATION_COLOR
        self._set_layer_color(layer_name, c)
        return c

    def _active_color_target_layer_name(self) -> Optional[str]:
        if self._color_target == "branch":
            name = self.branch_combo.currentText().strip()
        else:
            name = _combo_layer_name(self.branch_trunk_combo)
            if not name:
                name = self.branch_trunk_combo.currentText().strip()
        if name and name in self.viewer.layers:
            return name
        return None

    def _sync_color_combo_from_target(self) -> None:
        name = self._active_color_target_layer_name()
        if not name:
            return
        default = (
            _branch_points_color_for_layer_name(name)
            if self._color_target == "branch" and name
            else DEFAULT_SEGMENTATION_COLOR
        )
        color = _sanitize_branch_display_color(
            self._layer_color(name, default)
        )
        if self._color_target == "branch" and name:
            color = _branch_points_color_for_layer_name(name)
        choices = (
            LAYER_COLOR_CHOICES
            if self._color_target == "segmentation"
            else DRAFT_BRANCH_COLOR_CHOICES
        )
        if color not in choices:
            color = default
        cb = self.seg_color_combo
        self._syncing_color_combo = True
        cb.blockSignals(True)
        cb.clear()
        cb.addItems(list(choices))
        idx = cb.findText(color)
        if idx >= 0:
            cb.setCurrentIndex(idx)
        cb.blockSignals(False)
        self._syncing_color_combo = False

    def _on_branch_combo_color_target(self, *_args: Any) -> None:
        self._color_target = "branch"
        self._sync_color_combo_from_target()
        self._sync_all_branch_points_palette_colors()
        self._sync_draft_branch_labels_colors()

    def _on_branch_trunk_combo_color_target(self, *_args: Any) -> None:
        self._color_target = "segmentation"
        self._sync_color_combo_from_target()

    def _apply_color_to_layer_name(self, layer_name: str, color: str) -> None:
        if layer_name not in self.viewer.layers:
            return
        lyr = self.viewer.layers[layer_name]
        if isinstance(lyr, napari.layers.Points) and _is_auto_sized_branch_points_name(
            layer_name
        ):
            try:
                lyr.face_color = color
                lyr.border_color = "white"
            except Exception:
                pass
            return
        if isinstance(lyr, napari.layers.Labels) and self._is_segmentation_labels_layer(
            lyr
        ):
            self._apply_stored_labels_color(lyr, color)

    def _apply_stored_labels_color(
        self, lyr: Any, color: Optional[str] = None
    ) -> None:
        if not isinstance(lyr, napari.layers.Labels):
            return
        if str(getattr(lyr, "name", "")) == "Skeletal Preview":
            return
        c = color or self._layer_color(
            str(lyr.name), DEFAULT_SEGMENTATION_COLOR
        )
        _configure_binary_labels_layer(lyr)
        _sync_binary_labels_colormap(lyr, c)

    def _draft_branch_preview_color(self) -> str:
        return _draft_branch_color_for_layer_name(DRAFT_BRANCH_LAYER_NAME)

    def _segmentation_label_color(self) -> str:
        name = _combo_layer_name(self.branch_trunk_combo)
        if not name:
            name = self.branch_trunk_combo.currentText().strip()
        if name:
            return self._layer_color(name, DEFAULT_SEGMENTATION_COLOR)
        return DEFAULT_SEGMENTATION_COLOR

    def _segmentation_label_color_hex(self, layer_name: Optional[str] = None) -> str:
        """RGB hex string (no ``#``) for OME-Zarr label metadata."""
        from napari.utils.color import ColorArray

        if layer_name:
            color = self._layer_color(layer_name, DEFAULT_SEGMENTATION_COLOR)
        else:
            color = self._segmentation_label_color()
        rgba = np.asarray(ColorArray(color)).reshape(-1)
        if rgba.size < 3:
            return "FF00FF"
        r, g, b = (int(np.clip(round(float(c) * 255), 0, 255)) for c in rgba[:3])
        return f"{r:02X}{g:02X}{b:02X}"

    def _is_segmentation_labels_layer(self, lyr: Any) -> bool:
        if not isinstance(lyr, napari.layers.Labels):
            return False
        nm = str(lyr.name)
        if nm in (MERGED_SEG_LAYER_NAME, DRAFT_BRANCH_LAYER_NAME):
            return True
        if nm.startswith("Draft_Branch"):
            return True
        if nm.startswith("Segmentation Result"):
            return True
        if nm.startswith(f"{MERGED_SEG_LAYER_NAME} ("):
            return True
        if _is_segmentation_mask_numbered_name(nm):
            return True
        if nm.startswith("Mask") or nm.startswith("Mask_"):
            return True
        return False

    def _on_layer_color_changed(self, color: str) -> None:
        if self._syncing_color_combo:
            return
        name = self._active_color_target_layer_name()
        if not name:
            return
        if _is_auto_sized_branch_points_name(name):
            color = _sanitize_branch_display_color(color)
        self._set_layer_color(name, color)
        self._apply_color_to_layer_name(name, color)
        if (
            _is_auto_sized_branch_points_name(name)
            and name == self.branch_combo.currentText().strip()
            and DRAFT_BRANCH_LAYER_NAME in self.viewer.layers
        ):
            self._apply_stored_labels_color(
                self.viewer.layers[DRAFT_BRANCH_LAYER_NAME], color
            )

    def _merge_postprocess_active(self) -> bool:
        """True while merge cleanup is queued or running."""
        if getattr(self, "_merge_layers_mutating", False):
            return True
        if getattr(self, "_merge_postprocess_payload", None) is not None:
            return True
        tmr = getattr(self, "_merge_postprocess_timer", None)
        return bool(tmr is not None and tmr.isActive())

    def _safe_remove_viewer_layer(self, name: str) -> None:
        """Hide a layer before removal (avoid VisPy/napari list races)."""
        if not name or name not in self.viewer.layers:
            return
        lyr = self.viewer.layers[name]
        try:
            lyr.visible = False
        except Exception:
            pass
        try:
            self.viewer.layers.remove(name)
        except Exception:
            pass

    def _forced_3d_layer_name(self, image_layer_name: str) -> str:
        return f"{image_layer_name} (3D view)"

    def _forced_2d_layer_name(self, image_layer_name: str) -> str:
        return f"{image_layer_name} (2D fixed level)"

    def _restore_proxy_display_to_source(
        self,
        proxy_layer_name: Optional[str],
        *,
        pyramid_level: Optional[int] = None,
    ) -> None:
        """Push display edits from a hidden proxy layer back onto the source image."""
        if not proxy_layer_name or proxy_layer_name not in self.viewer.layers:
            return
        proxy = self.viewer.layers[proxy_layer_name]
        source = self._get_selected_image_layer()
        if source is None or not isinstance(proxy, napari.layers.Image):
            return
        if not isinstance(source, napari.layers.Image):
            return
        level = (
            int(pyramid_level)
            if pyramid_level is not None
            else int(self._selected_pyramid_level())
        )
        try:
            copy_proxy_display_to_multiscale_source(
                proxy,
                source,
                pyramid_level=level,
            )
        except Exception:
            pass

    def _apply_source_display_to_proxy(self, source: Any, proxy: Any, level: int) -> None:
        if not isinstance(source, napari.layers.Image) or not isinstance(
            proxy, napari.layers.Image
        ):
            return
        try:
            copy_multiscale_source_display_to_proxy(
                source, proxy, pyramid_level=int(level)
            )
        except Exception:
            pass

    def _remove_forced_2d_display_layer(self, *, restore_display: bool = True) -> None:
        name = getattr(self, "_forced_2d_display_layer_name", None)
        if restore_display:
            self._restore_proxy_display_to_source(name)
        if not name:
            return
        if name in self.viewer.layers:
            try:
                self.viewer.layers.remove(name)
            except Exception:
                pass
        self._forced_2d_display_layer_name = None
        self._forced_2d_proxy_level = None

    def _use_fixed_2d_pyramid_level(self) -> bool:
        """True when 2D view should lock to the plugin pyramid level combo."""
        cb = getattr(self, "ms_2d_multiscale_render_check", None)
        if cb is None or not cb.isVisible() or cb.isChecked():
            return False
        return True

    def _get_selected_image_layer(self) -> Optional[Any]:
        iname = self._selected_image_layer_name()
        if not iname or iname not in self.viewer.layers:
            return None
        img = self.viewer.layers[iname]
        if not isinstance(img, napari.layers.Image):
            return None
        return img

    def _update_pyramid_display_layers(self) -> None:
        """Sync 2D/3D display proxies vs napari multiscale auto-rendering."""
        try:
            nd = int(getattr(self.viewer.dims, "ndisplay", 2))
        except Exception:
            nd = 2
        if nd >= 3:
            self._remove_forced_2d_display_layer()
            self._update_forced_3d_display_layer()
            return
        self._remove_forced_3d_display_layer()
        self._update_forced_2d_display_layer()
        self._apply_pyramid_dims_navigation()

    def _update_forced_2d_display_layer(self) -> None:
        """In 2D, optionally show the selected pyramid level instead of auto-multiscale."""
        try:
            nd = int(getattr(self.viewer.dims, "ndisplay", 2))
        except Exception:
            nd = 2
        if nd >= 3:
            self._remove_forced_2d_display_layer()
            return

        img = self._get_selected_image_layer()
        if img is None or not is_multiscale_image_layer(img):
            self._remove_forced_2d_display_layer()
            return

        if not self._use_fixed_2d_pyramid_level():
            self._remove_forced_2d_display_layer()
            try:
                img.visible = True
            except Exception:
                pass
            return

        level = int(self._selected_pyramid_level())
        try:
            shp = image_level_shape(img, level)
        except Exception:
            shp = ()
        if len(shp) != 3:
            self._remove_forced_2d_display_layer()
            return

        block = image_level_data(img, level)
        if block is None:
            self._remove_forced_2d_display_layer()
            return

        skw = spatial_alignment_for_pyramid_level(img, level)
        nm = self._forced_2d_layer_name(img.name)
        if (
            self._forced_2d_display_layer_name
            and self._forced_2d_display_layer_name != nm
        ):
            self._remove_forced_2d_display_layer(restore_display=True)
        self._forced_2d_display_layer_name = nm

        sync_display = getattr(self, "_forced_2d_proxy_level", None) != level
        if nm in self.viewer.layers:
            try:
                lyr = self.viewer.layers[nm]
                lyr.data = block
                _apply_spatial_kwargs_to_layer(lyr, skw)
                if sync_display:
                    self._apply_source_display_to_proxy(img, lyr, level)
                lyr.visible = True
            except Exception:
                pass
        else:
            lyr = self.viewer.add_image(
                block,
                name=nm,
                multiscale=False,
                **skw,
            )
            self._apply_source_display_to_proxy(img, lyr, level)
        self._forced_2d_proxy_level = level
        self._pin_forced_display_layer_to_stack_bottom(nm)
        QTimer.singleShot(0, lambda: self._pin_forced_display_layer_to_stack_bottom(nm))
        try:
            img.visible = False
        except Exception:
            pass
        self._schedule_pyramid_dims_navigation()

    def _remove_forced_3d_display_layer(self, *, restore_display: bool = True) -> None:
        name = getattr(self, "_forced_3d_display_layer_name", None)
        if restore_display:
            self._restore_proxy_display_to_source(name)
        if not name:
            return
        if name in self.viewer.layers:
            try:
                self.viewer.layers.remove(name)
            except Exception:
                pass
        self._forced_3d_display_layer_name = None
        self._forced_3d_proxy_level = None
        self._forced_3d_proxy_pinned = None

    def _update_forced_3d_display_layer(self) -> None:
        """In 3D view, show the selected pyramid level instead of napari's auto-coarsest."""
        try:
            nd = int(getattr(self.viewer.dims, "ndisplay", 2))
        except Exception:
            nd = 2
        if nd < 3:
            self._remove_forced_3d_display_layer()
            return

        iname = self._selected_image_layer_name()
        if not iname or iname not in self.viewer.layers:
            self._remove_forced_3d_display_layer()
            return
        img = self.viewer.layers[iname]
        if not isinstance(img, napari.layers.Image) or not is_multiscale_image_layer(img):
            self._remove_forced_3d_display_layer()
            return

        level = int(self._selected_pyramid_level())
        nm = self._forced_3d_layer_name(img.name)
        if (
            nd >= 3
            and nm in self.viewer.layers
            and self._forced_3d_display_layer_name == nm
            and getattr(self, "_forced_3d_proxy_level", None) == level
        ):
            return
        # Materialize just this level for display. For lazy data, this still reads the full level.
        try:
            shp = image_level_shape(img, level)
        except Exception:
            shp = ()
        if len(shp) != 3:
            self._remove_forced_3d_display_layer()
            return
        msg = check_materialization_budget(
            shp,
            np.dtype(np.float32),
            max_bytes=_MAX_3D_DISPLAY_BYTES,
            copies=1.5,
            context="3D display (forced pyramid level)",
        )
        if msg:
            # Too big to safely materialize; fall back to napari default behavior.
            self._remove_forced_3d_display_layer()
            self.status_label.setText(msg + " (3D display: pick a coarser level.)")
            return

        data = materialize_image_level_cached(
            img,
            level,
            dtype=np.float32,
            use_cache=self.grow_cache_level_check.isChecked(),
        )
        skw = spatial_alignment_for_pyramid_level(img, level)
        # Keep unique and stable across renames.
        if self._forced_3d_display_layer_name and self._forced_3d_display_layer_name != nm:
            self._remove_forced_3d_display_layer(restore_display=True)
        self._forced_3d_display_layer_name = nm

        sync_display = getattr(self, "_forced_3d_proxy_level", None) != level
        created_proxy = False
        if nm in self.viewer.layers:
            try:
                lyr = self.viewer.layers[nm]
                if sync_display:
                    lyr.data = np.asarray(data)
                _apply_spatial_kwargs_to_layer(lyr, skw)
                if sync_display:
                    self._apply_source_display_to_proxy(img, lyr, level)
                lyr.visible = True
            except Exception:
                pass
        else:
            lyr = self.viewer.add_image(
                np.asarray(data),
                name=nm,
                **skw,
            )
            self._apply_source_display_to_proxy(img, lyr, level)
            created_proxy = True
        self._forced_3d_proxy_level = level
        if created_proxy or getattr(self, "_forced_3d_proxy_pinned", None) != nm:
            self._pin_forced_display_layer_to_stack_bottom(nm)
            self._forced_3d_proxy_pinned = nm
        # Hide the original multiscale layer in 3D so the user doesn't see the coarsest proxy.
        try:
            img.visible = False
        except Exception:
            pass
        self._schedule_pyramid_dims_navigation()

    def _pin_forced_display_layer_to_stack_bottom(self, name: str) -> None:
        """Move a synthetic display image to internal index 0 (bottom of the layer dock)."""
        if not name or name not in self.viewer.layers:
            return
        try:
            layers = self.viewer.layers
            idx = int(layers.index(name))
            if idx != 0:
                layers.move(idx, 0)
        except Exception:
            pass

    def _pin_forced_3d_display_layer_to_stack_bottom(self) -> None:
        """Move the synthetic ``(3D view)`` image to internal index 0 (bottom of the layer dock)."""
        nm = getattr(self, "_forced_3d_display_layer_name", None)
        self._pin_forced_display_layer_to_stack_bottom(nm or "")

    @staticmethod
    def _looks_like_channel_layer_name(name: str) -> bool:
        return looks_like_channel_layer_name(name)

    def _suggest_image_layer_name(self, layer: Any) -> Optional[str]:
        """Suggest a nicer name than 'C:0' using metadata/source when available."""
        md = dict(getattr(layer, "metadata", {}) or {})
        # ome-zarr-py Reader uses node.metadata["metadata"]={"image":..., "path": <basename>}
        meta2 = md.get("metadata")
        if isinstance(meta2, dict):
            p = meta2.get("path")
            if isinstance(p, str) and p.strip():
                return p.strip()
        # Our TIFF readers store source_path in metadata.
        sp = md.get("source_path")
        if isinstance(sp, str) and sp.strip():
            return Path(sp).name
        # Try napari layer.source if present.
        src = getattr(layer, "source", None)
        for attr in ("path", "uri", "url"):
            val = getattr(src, attr, None) if src is not None else None
            if isinstance(val, str) and val.strip():
                return Path(val).name
        return None

    def _unique_layer_name(self, base: str) -> str:
        base = str(base).strip() or "Image"
        if base not in self.viewer.layers:
            return base
        k = 2
        while f"{base} ({k})" in self.viewer.layers:
            k += 1
        return f"{base} ({k})"

    def _infer_omezarr_store_path(self, layer: Any) -> Optional[Path]:
        """Best-effort infer the originally loaded .ome.zarr folder for an image layer."""
        from ._omezarr_reader import _path_if_under_omezarr, resolve_omezarr_store_root

        md = dict(getattr(layer, "metadata", {}) or {})
        candidates: List[str] = []
        for key in ("source_path", "path", "uri", "url"):
            v = md.get(key)
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip())
        src = getattr(layer, "source", None)
        for attr in ("path", "uri", "url"):
            v = getattr(src, attr, None) if src is not None else None
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip())
        meta2 = md.get("metadata")
        if isinstance(meta2, dict):
            for k in ("path", "uri", "url"):
                v = meta2.get(k)
                if isinstance(v, str) and v.strip():
                    candidates.append(v.strip())
        for c in candidates:
            root = _path_if_under_omezarr(c)
            if root is not None:
                try:
                    return resolve_omezarr_store_root(root)
                except OSError:
                    return resolve_omezarr_store_root(root)
            p = Path(c).expanduser()
            low = str(p).lower()
            if low.endswith(".ome.zarr") or low.endswith(".zarr"):
                try:
                    return resolve_omezarr_store_root(p)
                except OSError:
                    return resolve_omezarr_store_root(p)
        return None

    def _hook_layer_dims_nav_events(self, layer: Any) -> None:
        """Re-apply pyramid Z/Y/X slider steps when the *image* grid changes.

        Do not hook Points/Labels — every new branch point would re-run
        ``dims.set_range`` and jump the Z slice.
        """
        if not isinstance(layer, napari.layers.Image):
            return
        for name in ("data", "scale", "translate", "rotate", "shear", "affine"):
            try:
                getattr(layer.events, name).connect(
                    self._schedule_pyramid_dims_navigation
                )
            except Exception:
                pass

    def _dims_navigation_snapshot(self) -> Tuple[Tuple[float, ...], int]:
        dims = self.viewer.dims
        return (tuple(float(x) for x in dims.point), int(dims.last_used))

    def _dims_navigation_restore(self, snap: Tuple[Tuple[float, ...], int]) -> None:
        point, last_used = snap
        dims = self.viewer.dims
        try:
            if len(point) == int(dims.ndim):
                dims.point = point
            dims.last_used = int(last_used)
        except Exception:
            pass

    def _schedule_pyramid_dims_navigation(self, *_args: Any) -> None:
        if getattr(self, "_pyramid_dims_nav_timer", None) is not None:
            self._pyramid_dims_nav_timer.start()

    def _on_layer_inserted(self, event=None) -> None:
        """Refresh combos and optionally rename channel-like image layers (C:0…)."""
        try:
            layer = getattr(event, "value", None)
        except Exception:
            layer = None
        if layer is not None:
            self._hook_layer_dims_nav_events(layer)
            if isinstance(
                layer, (napari.layers.Image, napari.layers.Labels)
            ):
                self._schedule_pyramid_dims_navigation()
        if layer is not None and isinstance(layer, napari.layers.Image):
            channel_split_count = count_channel_split_image_layers(
                list(self.viewer.layers)
            )
            was_channel_layer = self._looks_like_channel_layer_name(layer.name)
            if was_channel_layer:
                sug = self._suggest_image_layer_name(layer)
                if sug:
                    new_name = self._unique_layer_name(sug)
                    try:
                        layer.name = new_name
                    except Exception:
                        pass
            if was_channel_layer and channel_split_count > 1:
                QTimer.singleShot(
                    0,
                    lambda: self._update_image_validation_ui(show_dialog=True),
                )
            elif not was_channel_layer:
                QTimer.singleShot(
                    0,
                    lambda: self._update_image_validation_ui(show_dialog=False),
                )
        self._refresh_layers()

    def _on_layer_removed(self, event=None) -> None:
        """Drop cached pyramid levels for a removed layer, then refresh combos."""
        layer = None
        try:
            layer = getattr(event, "value", None)
        except Exception:
            layer = None
        if layer is not None:
            name = getattr(layer, "name", None)
            if name is not None:
                invalidate_image_level_cache(str(name))
        self._schedule_pyramid_dims_navigation()
        self._refresh_layers()

    @staticmethod
    def _worker_exception_message(exc) -> str:
        if isinstance(exc, BaseException):
            return str(exc)
        if isinstance(exc, tuple) and len(exc) >= 2 and exc[1] is not None:
            return str(exc[1])
        return str(exc)

    def _sync_save_combined_gif_button(self) -> None:
        n = len(getattr(self, "_gif_capture_combined_frames", []))
        self.btn_save_combined_gif.setEnabled(n > 0)

    def _on_capture_growth_toggled(self, checked: bool) -> None:
        self.capture_options.setVisible(checked)
        self.capture_combine_grows_check.setEnabled(checked)
        if not checked:
            self.capture_combine_grows_check.setChecked(False)
            self._gif_capture_pending_segment.clear()
        self._sync_save_combined_gif_button()

    def _on_capture_combine_toggled(self, checked: bool) -> None:
        if not checked:
            self._gif_capture_pending_segment.clear()

    def _save_combined_growth_gif(self) -> None:
        frames = list(self._gif_capture_combined_frames)
        if not frames:
            self.status_label.setText("No combined GIF frames — Merge after grows with combine mode on.")
            return
        default_name = (
            f"vessel_growth_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        )
        path, _sel = QFileDialog.getSaveFileName(
            self,
            "Save combined growth GIF",
            default_name,
            "GIF (*.gif);;All files (*)",
        )
        if not path:
            self.status_label.setText("Combined GIF save cancelled.")
            return
        if not path.lower().endswith(".gif"):
            path = path + ".gif"
        fps = float(self.capture_output_fps_spin.value())
        self._start_gif_encode_worker(
            frames,
            path,
            fps,
            hit_max=False,
            had_error=False,
            clear_combined_after=True,
        )

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        root_layout = QVBoxLayout()
        self.setLayout(root_layout)

        # Use a scroll area so the dock widget stays vertically resizable
        # even when the parameter panel grows taller than the window.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root_layout.addWidget(scroll)

        content = QWidget()
        content.setMaximumWidth(_DOCK_CONTENT_MAX_WIDTH)
        self._dock_content = content
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(content)

        # --- Shared: Layer selection ---
        layer_inner = QWidget()
        layer_inner.setMaximumWidth(_DOCK_CONTENT_MAX_WIDTH)
        layer_form = QFormLayout(layer_inner)
        _configure_form_layout(layer_form)

        self.image_combo = QComboBox()
        layer_form.addRow("Image:", _row(self.image_combo))

        self._image_validation_label = QLabel("")
        self._image_validation_label.setWordWrap(True)
        self._image_validation_label.setVisible(False)
        self._image_validation_label.setMaximumWidth(_DOCK_CONTENT_MAX_WIDTH - 8)
        self._image_validation_label.setStyleSheet(
            "color: #b45309; font-weight: bold; padding: 2px 0;"
        )
        layer_form.addRow(self._image_validation_label)

        self.ms_level_combo = QComboBox()
        self.ms_level_combo.setToolTip(
            "For multiscale / pyramid images (e.g. OME-Zarr), pick which resolution "
            "Grow and masks use. Level 0 is finest; higher levels are smaller and faster."
        )
        self._ms_level_row_label = QLabel("Pyramid level:")
        self._ms_level_row_label.setVisible(False)
        self.ms_level_combo.setVisible(False)
        layer_form.addRow(self._ms_level_row_label, _row(self.ms_level_combo))

        self.ms_2d_multiscale_render_check = QCheckBox(
            "Enable multiscale rendering in 2D"
        )
        self.ms_2d_multiscale_render_check.setChecked(False)
        self.ms_2d_multiscale_render_check.setVisible(False)
        self.ms_2d_multiscale_render_check.setToolTip(
            "When enabled, napari switches pyramid levels automatically as you zoom "
            "(default OME-Zarr behaviour).\n"
            "When disabled, the canvas stays on the selected **Pyramid level** so you "
            "can zoom and pan for layout without loading finer resolution."
        )
        layer_form.addRow(_row(self.ms_2d_multiscale_render_check))

        self.ms_adapt_slice_step_check = QCheckBox(
            "Adapt Z step to pyramid level"
        )
        self.ms_adapt_slice_step_check.setChecked(True)
        self.ms_adapt_slice_step_check.setVisible(False)
        self.ms_adapt_slice_step_check.setToolTip(
            "When viewing a coarser pyramid level (2D with multiscale rendering off, "
            "or 3D), increase the napari Z slider step so each keypress shows a new "
            "slice instead of several identical subsampled planes.\n"
            "Step size is derived from the image downsample factors (Z, then Y/X)."
        )
        layer_form.addRow(_row(self.ms_adapt_slice_step_check))

        self.grow_use_roi_check = QCheckBox("Crop Compute Branch to polyline ROI")
        self.grow_use_roi_check.setChecked(True)
        self.grow_use_roi_check.setToolTip(
            "When enabled, only a bounding box around the branch polyline is loaded and "
            "processed (padding ≈ 3× tube radius in Z and XY). Disable to run on the "
            "full pyramid level (slower; may hit the RAM limit on large volumes)."
        )
        layer_form.addRow(_row(self.grow_use_roi_check))

        self.grow_cache_level_check = QCheckBox("Cache pyramid level in session (RAM)")
        self.grow_cache_level_check.setChecked(True)
        self.grow_cache_level_check.setToolTip(
            "Keep each pyramid level in memory after the first full read on that level. "
            "Speeds up repeated Grows on the same level. Oldest levels are evicted when "
            "total cache exceeds ~6 GB. Uncheck to always read from disk (lower RAM). "
            "Turning off clears the cache immediately."
        )
        layer_form.addRow(_row(self.grow_cache_level_check))

        self.btn_load_saved_segmentation = QPushButton(
            "Load saved segmentation from OME-Zarr…"
        )
        self.btn_load_saved_segmentation.setToolTip(
            "Load a saved NGFF labels group into the Segmentation layer on the "
            "current Image and pyramid level (replaces the empty default mask).\n"
            "Select the .ome.zarr root or labels/ to choose a group; select "
            "labels/<name>/ to load that group directly."
        )
        layer_form.addRow(_row(self.btn_load_saved_segmentation))

        layout.addWidget(_collapsible_section("Layers", layer_inner))

        # --- Shared: Visualization (valid for both modes) ---
        vis_inner = QWidget()
        vis_inner.setMaximumWidth(_DOCK_CONTENT_MAX_WIDTH)
        vis_form = QFormLayout(vis_inner)
        _configure_form_layout(vis_form)

        self.animate_check = QCheckBox("Animate growth")
        self.animate_check.setChecked(False)
        self.animate_check.setToolTip(
            "Show the growing contour at each display step.\n"
            "Disable for maximum speed.\n"
            "Update frequency is set below (Plain / MGAC)."
        )
        self.capture_growth_check = QCheckBox("Capture animation (GIF)")
        self.capture_growth_check.setChecked(False)
        self.capture_growth_check.setToolTip(
            "Record the viewer during each Grow.\n"
            "Either save one GIF per Grow, or enable “Combine…” to build one GIF from "
            "several grows (frames are added only when you Merge; Reset branch preview "
            "discards the last grow’s frames).\n"
            "Uses the same steps as Animate growth; turn Animate on for a smooth clip."
        )
        anim_cap_row = QWidget()
        anim_cap_layout = QHBoxLayout(anim_cap_row)
        anim_cap_layout.setContentsMargins(0, 0, 0, 0)
        anim_cap_layout.addWidget(self.animate_check)
        anim_cap_layout.addWidget(self.capture_growth_check)
        anim_cap_layout.addStretch(1)
        vis_form.addRow(anim_cap_row)

        self._vis_plain_step_label = QLabel("Every N voxels:")
        self.branch_plain_step_spin = QSpinBox()
        self.branch_plain_step_spin.setRange(1, 10000)
        self.branch_plain_step_spin.setValue(500)
        self.branch_plain_step_spin.setToolTip(
            "Animate branch plain growth every N accepted voxels when Animate growth is on."
        )
        self._vis_plain_step_field = QWidget()
        self._vis_plain_step_field.setLayout(_row(self.branch_plain_step_spin))
        vis_form.addRow(self._vis_plain_step_label, self._vis_plain_step_field)

        self._vis_ac_yield_label = QLabel("Every N iterations:")
        self.branch_ac_yield_spin = QSpinBox()
        self.branch_ac_yield_spin.setRange(1, 500)
        self.branch_ac_yield_spin.setValue(5)
        self.branch_ac_yield_spin.setToolTip(
            "Refresh the viewer every N MGAC iterations when Animate growth is on."
        )
        self._vis_ac_yield_field = QWidget()
        self._vis_ac_yield_field.setLayout(_row(self.branch_ac_yield_spin))
        vis_form.addRow(self._vis_ac_yield_label, self._vis_ac_yield_field)

        self.seg_color_combo = QComboBox()
        self.seg_color_combo.addItems(list(LAYER_COLOR_CHOICES))
        self.seg_color_combo.setCurrentText(DEFAULT_SEGMENTATION_COLOR)
        self.seg_color_combo.setToolTip(
            "Color for the layer selected in Branch points layer or Segmentation mask. "
            f"Magenta is reserved for segmentation; all branch point layers use cyan. "
            f"Draft_Branch preview colors rotate by layer name (cyan, lime, yellow, …). "
            f"New segmentation layers default to {DEFAULT_SEGMENTATION_COLOR}."
        )
        vis_form.addRow("Layer color:", _row(self.seg_color_combo))

        self.capture_options = QWidget()
        cap_form = QFormLayout(self.capture_options)
        _configure_form_layout(cap_form)
        cap_form.setContentsMargins(12, 0, 0, 0)

        self.capture_output_fps_spin = QDoubleSpinBox()
        self.capture_output_fps_spin.setRange(0.5, 60.0)
        self.capture_output_fps_spin.setValue(10.0)
        self.capture_output_fps_spin.setDecimals(1)
        self.capture_output_fps_spin.setSingleStep(0.5)
        self.capture_output_fps_spin.setToolTip(
            "GIF playback speed (frames per second in the exported file)."
        )
        cap_form.addRow("GIF playback FPS:", _row(self.capture_output_fps_spin))

        self.capture_subsample_spin = QSpinBox()
        self.capture_subsample_spin.setRange(1, 100)
        self.capture_subsample_spin.setValue(1)
        self.capture_subsample_spin.setToolTip(
            "Keep every N-th displayed grow step in the GIF (1 = all steps shown in the viewer)."
        )
        cap_form.addRow("Frame subsample (N):", _row(self.capture_subsample_spin))

        self.capture_max_frames_spin = QSpinBox()
        self.capture_max_frames_spin.setRange(0, 50_000)
        self.capture_max_frames_spin.setValue(800)
        self.capture_max_frames_spin.setSpecialValueText("No limit")
        self.capture_max_frames_spin.setToolTip(
            "Stop adding frames after this many (0 = no limit). The grow still runs to the end."
        )
        cap_form.addRow("Max frames:", _row(self.capture_max_frames_spin))

        self.capture_region_combo = QComboBox()
        self.capture_region_combo.addItems(["Viewer canvas", "Full napari window"])
        self.capture_region_combo.setToolTip(
            "Canvas: 3D viewer only. Full window: includes docks (e.g. this plugin panel)."
        )
        cap_form.addRow("Capture region:", _row(self.capture_region_combo))

        self.capture_scale_spin = QSpinBox()
        self.capture_scale_spin.setRange(10, 100)
        self.capture_scale_spin.setValue(100)
        self.capture_scale_spin.setSuffix(" %")
        self.capture_scale_spin.setToolTip(
            "Scale each frame to this percentage of width/height before saving (smaller GIF files)."
        )
        cap_form.addRow("Frame scale:", _row(self.capture_scale_spin))

        self.capture_gif_canvas_w_spin = QSpinBox()
        self.capture_gif_canvas_w_spin.setRange(0, 8192)
        self.capture_gif_canvas_w_spin.setValue(0)
        self.capture_gif_canvas_w_spin.setSpecialValueText("Auto")
        self.capture_gif_canvas_w_spin.setToolTip(
            "Output width in pixels for every GIF frame (letterboxed). "
            "Auto = use the widest frame in this save; set both width and height "
            "for a fixed canvas (recommended for combined GIFs)."
        )
        cap_form.addRow("GIF canvas width:", _row(self.capture_gif_canvas_w_spin))

        self.capture_gif_canvas_h_spin = QSpinBox()
        self.capture_gif_canvas_h_spin.setRange(0, 8192)
        self.capture_gif_canvas_h_spin.setValue(0)
        self.capture_gif_canvas_h_spin.setSpecialValueText("Auto")
        self.capture_gif_canvas_h_spin.setToolTip(
            "Output height in pixels for every GIF frame (letterboxed). "
            "Auto = use the tallest frame in this save; set both width and height "
            "for a fixed canvas (recommended for combined GIFs)."
        )
        cap_form.addRow("GIF canvas height:", _row(self.capture_gif_canvas_h_spin))

        self.capture_combine_grows_check = QCheckBox(
            "Combine branch grows in one GIF (commit on Merge)"
        )
        self.capture_combine_grows_check.setChecked(False)
        self.capture_combine_grows_check.setEnabled(False)
        self.capture_combine_grows_check.setToolTip(
            "Each Grow is recorded, but frames are appended to the combined clip only "
            "when you click Merge. Reset branch preview discards the last grow’s "
            "recording without merging. Starting a new Grow drops any unmerged "
            "recording from the previous attempt. Use Save combined GIF when done."
        )
        cap_form.addRow(_row(self.capture_combine_grows_check))

        self.btn_save_combined_gif = QPushButton("Save combined GIF…")
        self.btn_save_combined_gif.setEnabled(False)
        self.btn_save_combined_gif.setToolTip(
            "Save all segments committed with Merge while combine mode was on."
        )
        cap_form.addRow(_row(self.btn_save_combined_gif))

        self.capture_options.setVisible(False)
        vis_form.addRow(self.capture_options)

        layout.addWidget(
            _collapsible_section("Visualization", vis_inner, start_open=False)
        )

        # --- Segmentation (polyline + grow + merge; same flow for trunk and side branches) ---
        seg_inner = QWidget()
        seg_inner.setMaximumWidth(_DOCK_CONTENT_MAX_WIDTH)
        seg_form = QFormLayout(seg_inner)
        _configure_form_layout(seg_form)

        self.branch_trunk_combo = QComboBox()
        self.branch_trunk_combo.setToolTip(
            "Labels layer that receives Merge and supplies the existing mask during Grow "
            "(union with the growing region). Same shape as Image. "
            "Draft_Branch is never listed here. Selecting an image creates Segmentation "
            "when no mask exists yet. Use New Mask for additional masks on this grid."
        )
        mask_row_widget = QWidget()
        mask_row = QHBoxLayout(mask_row_widget)
        mask_row.setContentsMargins(0, 0, 0, 0)
        mask_row.addWidget(self.branch_trunk_combo, 1)
        self.btn_new_segmentation_mask = QPushButton("New Mask")
        self.btn_new_segmentation_mask.setToolTip(
            "Add a new empty Segmentation / Segmentation_N layer on the current image "
            "grid and select it here."
        )
        mask_row.addWidget(self.btn_new_segmentation_mask)
        seg_form.addRow("Segmentation mask:", mask_row_widget)

        pts_btn_row = QHBoxLayout()
        self.btn_reset_branch_points_layer = QPushButton("Reset branch points")
        self.btn_reset_branch_points_layer.setToolTip(
            "Clears all points in the layer currently selected under Branch points layer."
        )
        self.btn_new_branch_points_layer = QPushButton("New BranchPoints Layer")
        self.btn_new_branch_points_layer.setToolTip(
            "Adds a new points layer (BranchPoints, then BranchPoints_1, …) and "
            "selects it; existing branch point layers are kept."
        )
        pts_btn_row.addWidget(self.btn_new_branch_points_layer)
        pts_btn_row.addWidget(self.btn_reset_branch_points_layer)
        seg_form.addRow(pts_btn_row)

        self.branch_combo = QComboBox()
        self.branch_combo.setToolTip(
            "Points layer: all points in data order form one polyline (at least two points)."
        )
        seg_form.addRow("Branch points layer:", _row(self.branch_combo))

        self.draft_branch_combo = QComboBox()
        self.draft_branch_combo.setToolTip(
            "Branch preview merged by **Merge Branch**: the live "
            f'"{DRAFT_BRANCH_LAYER_NAME}" or an archived '
            f'"{DRAFT_BRANCH_LAYER_NAME} (N)" from an earlier Compute Branch.\n'
            "Compute Branch always writes to the live layer; pick an archive here "
            "to merge a preview you kept without merging earlier."
        )
        seg_form.addRow("Branch preview (merge from):", _row(self.draft_branch_combo))

        blocker_row = QWidget()
        blocker_h = QHBoxLayout(blocker_row)
        blocker_h.setContentsMargins(0, 0, 0, 0)
        self.blocker_combo = QComboBox()
        self.blocker_combo.setToolTip(
            "Optional labels layer on the same grid as Image / pyramid level: painted "
            "foreground blocks Plain grow and MGAC (speed 0, mask cleared each step).\n"
            "Pick (none) to disable."
        )
        blocker_h.addWidget(self.blocker_combo, 1)
        self.btn_new_blocker_layer = QPushButton("New Blocker")
        self.btn_new_blocker_layer.setToolTip(
            f'Add an empty labels layer named "{BLOCKER_MASK_LAYER_NAME}" (or with suffix) aligned to the '
            "current Image and pyramid level — paint walls where segmentation must not leak."
        )
        blocker_h.addWidget(self.btn_new_blocker_layer)
        seg_form.addRow("Blocker_Mask (optional):", blocker_row)

        self.branch_ac_radius_spin = QDoubleSpinBox()
        self.branch_ac_radius_spin.setRange(0.5, 500.0)
        self.branch_ac_radius_spin.setValue(40.0)
        self.branch_ac_radius_spin.setSingleStep(0.5)
        self.branch_ac_radius_spin.setToolTip(
            "Finest-level isotropic voxel radii: physical radius is value × min(finest spacing),\n"
            "independent of the pyramid level used for Grow. Builds the polyline seed tube "
            "for Plain and MGAC and is the MGAC nominal radius for 3D Active Contour."
        )

        self.branch_method_combo = QComboBox()
        self.branch_method_combo.addItems(
            [
                "3D Active Contour",
                "Plain Region Growing",
            ]
        )
        self.branch_method_combo.setCurrentIndex(0)
        self.branch_method_combo.setToolTip(
            "Fill method for Compute Branch.\n"
            "Plain = priority-queue region growing. 3D Active Contour = morphological MGAC on the edge image."
        )
        seg_form.addRow("Fill with:", _row(self.branch_method_combo))

        branch_params_inner = QWidget()
        branch_params_layout = QVBoxLayout(branch_params_inner)
        branch_params_layout.setContentsMargins(0, 0, 0, 0)
        branch_tube_form = QFormLayout()
        branch_tube_form.addRow("Seed tube radius (vox):", _row(self.branch_ac_radius_spin))
        branch_params_layout.addLayout(branch_tube_form)

        branch_plain_inner = QWidget()
        branch_plain_form = QFormLayout(branch_plain_inner)
        self.branch_plain_section = _collapsible_section(
            "Plain parameters (branch)", branch_plain_inner, start_open=True
        )
        self.branch_plain_section.setToolTip(
            "Used when Fill with is Plain Region Growing."
        )

        self.branch_plain_sigma_spin = QDoubleSpinBox()
        self.branch_plain_sigma_spin.setRange(0.1, 20.0)
        self.branch_plain_sigma_spin.setValue(2.0)
        self.branch_plain_sigma_spin.setSingleStep(0.5)
        self.branch_plain_sigma_spin.setToolTip(
            "Gaussian σ for edge costs (branch plain only)."
        )
        branch_plain_form.addRow("Smoothing σ:", _row(self.branch_plain_sigma_spin))

        self.branch_plain_flux_spin = QDoubleSpinBox()
        self.branch_plain_flux_spin.setRange(0.0, 50.0)
        self.branch_plain_flux_spin.setValue(15.0)
        self.branch_plain_flux_spin.setSingleStep(1.0)
        self.branch_plain_flux_spin.setDecimals(1)
        self.branch_plain_flux_spin.setToolTip("Flux penalty weight (branch plain only).")
        branch_plain_form.addRow("Flux penalty:", _row(self.branch_plain_flux_spin))

        self.branch_plain_intensity_tol_spin = QDoubleSpinBox()
        self.branch_plain_intensity_tol_spin.setRange(0.5, 10.0)
        self.branch_plain_intensity_tol_spin.setValue(3.0)
        self.branch_plain_intensity_tol_spin.setSingleStep(0.5)
        self.branch_plain_intensity_tol_spin.setToolTip(
            "Intensity gate: σ below region mean (branch plain only)."
        )
        branch_plain_form.addRow(
            "Intensity tolerance:", _row(self.branch_plain_intensity_tol_spin)
        )

        self.branch_plain_cost_budget_spin = QDoubleSpinBox()
        self.branch_plain_cost_budget_spin.setRange(0.0, 100000.0)
        self.branch_plain_cost_budget_spin.setValue(0.0)
        self.branch_plain_cost_budget_spin.setDecimals(1)
        self.branch_plain_cost_budget_spin.setToolTip(
            "Accumulated growth-cost budget; 0 = auto (branch plain only)."
        )
        branch_plain_form.addRow(
            "Cost budget (0=auto):", _row(self.branch_plain_cost_budget_spin)
        )

        self.branch_plain_margin_spin = QDoubleSpinBox()
        self.branch_plain_margin_spin.setRange(0, 200)
        self.branch_plain_margin_spin.setValue(0.0)
        self.branch_plain_margin_spin.setToolTip(
            "Length slack along the branch axis: margin × mean(finest spacing), scaled to\n"
            "the working level (stable across pyramid levels)."
        )
        branch_plain_form.addRow("Length margin:", _row(self.branch_plain_margin_spin))

        self.branch_plain_upper_thr_check = QCheckBox("Enable upper threshold")
        self.branch_plain_upper_thr_check.setChecked(False)
        self.branch_plain_upper_thr_check.setToolTip(
            "Hard intensity cap for branch plain (same idea as trunk Plain tab)."
        )
        branch_plain_form.addRow(_row(self.branch_plain_upper_thr_check))

        self.branch_plain_upper_thr_combo = QComboBox()
        self.branch_plain_upper_thr_combo.addItems([
            "Otsu",
            "Triangle",
            "Li",
            "90th percentile",
            "95th percentile",
        ])
        self.branch_plain_upper_thr_combo.setEnabled(False)
        self.branch_plain_upper_thr_combo.setToolTip(
            "Method for upper threshold when enabled (branch plain only)."
        )
        branch_plain_form.addRow("Threshold method:", _row(self.branch_plain_upper_thr_combo))

        branch_ac_inner = QWidget()
        branch_ac_form = QFormLayout(branch_ac_inner)
        self.branch_ac_section = _collapsible_section(
            "MGAC parameters (branch)", branch_ac_inner, start_open=True
        )
        self.branch_ac_section.setToolTip(
            "Shown when Fill with is 3D Active Contour (MGAC on the polyline tube)."
        )

        self.branch_ac_margin_spin = QDoubleSpinBox()
        self.branch_ac_margin_spin.setRange(0, 200)
        self.branch_ac_margin_spin.setValue(0.15)
        self.branch_ac_margin_spin.setToolTip(
            "Extra slack for the MGAC polyline corridor (same convention as Plain length margin:\n"
            "margin × mean(finest spacing), scaled to the working pyramid level)."
        )
        branch_ac_form.addRow("Corridor length margin:", _row(self.branch_ac_margin_spin))

        self.branch_ac_sigma_spin = QDoubleSpinBox()
        self.branch_ac_sigma_spin.setRange(0.1, 20.0)
        self.branch_ac_sigma_spin.setValue(3.0)
        self.branch_ac_sigma_spin.setSingleStep(0.5)
        self.branch_ac_sigma_spin.setToolTip(
            "Gaussian σ for the inverse-gradient edge image (branch MGAC only)."
        )
        branch_ac_form.addRow("Smoothing σ:", _row(self.branch_ac_sigma_spin))

        self.branch_ac_low_clip_spin = QDoubleSpinBox()
        self.branch_ac_low_clip_spin.setRange(0.0, 100000.0)
        self.branch_ac_low_clip_spin.setValue(0.0)
        self.branch_ac_low_clip_spin.setSingleStep(1.0)
        self.branch_ac_low_clip_spin.setDecimals(1)
        self.branch_ac_low_clip_spin.setToolTip(
            "Optional lumen/background flattening: set all intensities ≤ this value to 0 before MGAC.\n"
            "Useful when lumen/background is ~1–20 but noisy; reduces micro-gradients that create holes.\n"
            "Set to 0 to disable."
        )
        branch_ac_form.addRow("Low-intensity clip (≤ → 0):", _row(self.branch_ac_low_clip_spin))

        self.branch_ac_balloon_spin = QDoubleSpinBox()
        self.branch_ac_balloon_spin.setRange(-5.0, 5.0)
        self.branch_ac_balloon_spin.setValue(0.1)
        self.branch_ac_balloon_spin.setSingleStep(0.1)
        self.branch_ac_balloon_spin.setDecimals(2)
        self.branch_ac_balloon_spin.setToolTip("Balloon coefficient (branch MGAC only).")
        branch_ac_form.addRow("Balloon:", _row(self.branch_ac_balloon_spin))

        self.branch_ac_smoothing_spin = QSpinBox()
        self.branch_ac_smoothing_spin.setRange(0, 10)
        self.branch_ac_smoothing_spin.setValue(0)
        self.branch_ac_smoothing_spin.setToolTip(
            "Morphological curvature (_curvop) per MGAC iteration after balloon+edge.\n"
            "Higher values act like a stronger smoothing / shrink prior (γ-like).\n"
            "0 avoids thinning narrow tubes. A light binary closing after each chunk\n"
            "still fills 1-voxel stripe gaps."
        )
        branch_ac_form.addRow("Smoothing γ (steps):", _row(self.branch_ac_smoothing_spin))

        self.branch_ac_total_iter_spin = QSpinBox()
        self.branch_ac_total_iter_spin.setRange(1, 5000)
        self.branch_ac_total_iter_spin.setValue(40)
        self.branch_ac_total_iter_spin.setToolTip("Total MGAC iterations (branch only).")
        branch_ac_form.addRow("Total iterations:", _row(self.branch_ac_total_iter_spin))

        self.branch_ac_early_stop_row = QWidget()
        early_stop_layout = QHBoxLayout(self.branch_ac_early_stop_row)
        early_stop_layout.setContentsMargins(0, 0, 0, 0)
        self.branch_ac_early_stop_slider = QSlider(Qt.Horizontal)
        self.branch_ac_early_stop_slider.setRange(0, 10)
        self.branch_ac_early_stop_slider.setValue(2)
        self.branch_ac_early_stop_slider.setTickPosition(QSlider.TicksBelow)
        self.branch_ac_early_stop_slider.setTickInterval(1)
        self.branch_ac_early_stop_slider.setToolTip(
            "Stop MGAC early after this many consecutive display updates with an "
            "unchanged mask. 0 = run all Total iterations. Higher = wait longer before "
            "stopping (more stable mask required)."
        )
        self.branch_ac_early_stop_value_label = QLabel("2")
        self.branch_ac_early_stop_value_label.setMinimumWidth(18)
        early_stop_layout.addWidget(self.branch_ac_early_stop_slider, 1)
        early_stop_layout.addWidget(self.branch_ac_early_stop_value_label)
        branch_ac_form.addRow("Early stop:", self.branch_ac_early_stop_row)

        branch_params_layout.addWidget(self.branch_plain_section)
        branch_params_layout.addWidget(self.branch_ac_section)
        seg_form.addRow(
            _collapsible_section(
                "Segmentation parameters (tube + Plain / MGAC)",
                branch_params_inner,
            )
        )

        self.btn_grow_branches = QPushButton("Compute Branch")
        self.btn_grow_branches.setToolTip(
            "Uses all points in the branch layer (in order) as one polyline (at least two points).\n"
            f'Writes the current computation result to "{DRAFT_BRANCH_LAYER_NAME}".\n'
            "Optional polyline ROI and session cache are under Layers."
        )
        grow_row = QHBoxLayout()
        grow_row.addWidget(self.btn_grow_branches)
        self.btn_reset_branch_seg = QPushButton("Reset Branch")
        self.btn_reset_branch_seg.setToolTip(
            f'Clears "{DRAFT_BRANCH_LAYER_NAME}" and shows the branch points layer again '
            "(does not remove points)."
        )
        grow_row.addWidget(self.btn_reset_branch_seg)
        self.btn_merge_branch_seg = QPushButton("Merge Branch")
        self.btn_merge_branch_seg.setToolTip(
            "Logical OR from the layer selected under Branch preview (merge from) "
            "into the Segmentation mask."
        )
        grow_row.addWidget(self.btn_merge_branch_seg)
        seg_form.addRow(grow_row)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        seg_ctrl = QVBoxLayout()
        seg_ctrl.setContentsMargins(0, 0, 0, 0)
        seg_ctrl.setSpacing(4)
        seg_ctrl.addWidget(self.btn_stop)
        self.merge_cleanup_check = QCheckBox("CleanUp")
        self.merge_cleanup_check.setChecked(True)
        self.merge_cleanup_check.setToolTip(
            "After **Merge Branch**, remove archived Draft_Branch (N) layers and "
            "extra BranchPoints_N layers. Keeps a single empty Draft_Branch and "
            "empty BranchPoints ready for the next branch."
        )
        seg_ctrl.addWidget(self.merge_cleanup_check)
        seg_form.addRow(seg_ctrl)

        layout.addWidget(
            _collapsible_section(
                "Segmentation",
                seg_inner,
                header_tooltip=(
                    "Add ordered points along one vessel segment, Compute Branch, then Merge Branch. "
                    "Use Blocker_Mask to paint walls where segmentation must not leak."
                ),
            )
        )

        # --- Post-processing (mask on working grid must exist; use Merge first) ---
        post_inner = QWidget()
        post_inner.setMaximumWidth(_DOCK_CONTENT_MAX_WIDTH)
        post_form = QFormLayout(post_inner)
        _configure_form_layout(post_form)

        self.btn_postprocess = QPushButton("Upsample Result to Original Size")
        self.btn_postprocess.setToolTip(
            "Enabled when the segmentation mask shape differs from the original "
            "image grid (e.g. after **Grow** on a coarser pyramid level).\n"
            "Creates a full-resolution mask by zooming to the original shape."
        )
        self.btn_postprocess.setEnabled(False)
        post_form.addRow(_row(self.btn_postprocess))

        self.btn_upsample_to_finer = QPushButton("Upsample segmentation to finer level…")
        self.btn_upsample_to_finer.setToolTip(
            "For large OME-Zarr volumes you often work on a coarse pyramid level.\n"
            "This creates a NEW editable labels layer on a finer (or finest) pyramid grid using\n"
            "nearest-neighbour upsampling, so you can inspect and refine details there."
        )
        post_form.addRow(_row(self.btn_upsample_to_finer))

        self.morph_op_combo = QComboBox()
        self.morph_op_combo.addItems(["None", "Dilation", "Erosion"])
        self.morph_op_combo.setCurrentIndex(0)
        self.morph_op_combo.setToolTip(
            "Apply morphological refinement to the segmentation:\n"
            "  - Dilation: expands the mask (fills tiny gaps/connects close parts)\n"
            "  - Erosion: shrinks the mask (removes thin over-segmentation)\n\n"
            "For anisotropic volumes, a common cleanup is one erosion\n"
            "with radius = 1 to reduce slight extra thickness along Z."
        )
        post_form.addRow("Operation:", _row(self.morph_op_combo))

        self.morph_radius_spin = QSpinBox()
        self.morph_radius_spin.setRange(1, 50)
        self.morph_radius_spin.setValue(1)
        self.morph_radius_spin.setToolTip(
            "Ball radius (voxels) for morphological refinement.\n"
            "Start with radius = 1 for minimal correction.\n"
            "In anisotropic images, radius-1 erosion is often enough\n"
            "to clean slight Z-direction over-segmentation."
        )
        post_form.addRow("Ball radius (vox):", _row(self.morph_radius_spin))

        self.btn_apply_morph = QPushButton("Apply Morphological Operation")
        self.btn_apply_morph.setToolTip(
            "Apply the selected operation to the tracked segmentation layer\n"
            "after visual inspection. Creates a new refined result layer."
        )
        post_form.addRow(_row(self.btn_apply_morph))

        layout.addWidget(
            _collapsible_section(
                "Post-processing",
                post_inner,
                start_open=False,
                header_tooltip=(
                    "Uses the last merged labels layer tracked by the plugin (often Segmentation Result). "
                    "Run Merge first if upsample or morphology is disabled."
                ),
            )
        )

        # --- Saving (OME-Zarr labels export) ---
        save_inner = QWidget()
        save_inner.setMaximumWidth(_DOCK_CONTENT_MAX_WIDTH)
        save_form = QFormLayout(save_inner)
        _configure_form_layout(save_form)

        self.save_resolution_combo = QComboBox()
        self.save_resolution_combo.addItems(
            ["Working pyramid level", "Full finest resolution"]
        )
        self.save_resolution_combo.setCurrentIndex(0)
        self.save_resolution_combo.setToolTip(
            "How to store labels in the OME-Zarr:\n"
            "  - Working pyramid level: keep the mask on the current grid (Layers → "
            "Pyramid level). Writes that level plus any coarser image pyramid levels "
            "(e.g. level 4 of 5 → arrays 0 and maybe 1, not five empty finer levels). "
            "Lowest RAM; recommended on limited hardware.\n"
            "  - Full finest resolution: upsample to the image finest grid and write a "
            "full label pyramid (chunked on disk). Peak RAM stays near the working mask "
            "size; the store still grows to full finest size on disk (can be tens of GB)."
        )
        save_form.addRow("Save resolution:", _row(self.save_resolution_combo))

        self.save_target_combo = QComboBox()
        self.save_target_combo.addItems(
            [
                "New version (segmentation_vN)",
                "Overwrite autosave (segmentation_autosave)",
                "Overwrite loaded segmentation",
                "Choose folder…",
            ]
        )
        self.save_target_combo.setCurrentIndex(_SAVE_TARGET_NEW)
        self.save_target_combo.setToolTip(
            "Where to write under labels/ in the image's .ome.zarr store:\n"
            "  - New version: next segmentation_vN in the loaded image store\n"
            "  - Overwrite autosave: replaces labels/segmentation_autosave in that store\n"
            "  - Overwrite loaded: replaces the labels group from Load saved segmentation…\n"
            "  - Choose folder…: pick another .ome.zarr root or labels/<name>/ to write into"
        )
        save_form.addRow("Save target:", _row(self.save_target_combo))

        self.btn_save_segmentation = QPushButton("Save segmentation")
        self.btn_save_segmentation.setToolTip(
            "Write the selected segmentation mask into an OME-Zarr labels group.\n"
            "New version, autosave, and overwrite-loaded use the store from the "
            "selected image (no folder dialog). Choose folder… opens a path picker.\n"
            "Merge still autosaves to labels/segmentation_autosave in the background."
        )
        save_form.addRow(_row(self.btn_save_segmentation))

        layout.addWidget(
            _collapsible_section(
                "Saving",
                save_inner,
                start_open=False,
                header_tooltip=(
                    "Export segmentation masks as NGFF labels under labels/ in the source "
                    ".ome.zarr store. Use Layers → Load saved segmentation… to import."
                ),
            )
        )

        # --- Shared: Status ---
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setMaximumWidth(_DOCK_CONTENT_MAX_WIDTH - 8)
        self.status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self._update_branch_method_dependent_widgets()
        self._apply_dock_layout_constraints()
        layout.addStretch()
        self.setMaximumWidth(_DOCK_CONTENT_MAX_WIDTH + 24)

    def _selected_image_layer_name(self) -> str:
        return _combo_layer_name(self.image_combo)

    def _apply_dock_layout_constraints(self) -> None:
        """Re-apply width caps (safe after combo repopulation / pyramid level changes)."""
        for cb in (
            self.image_combo,
            self.ms_level_combo,
            self.branch_combo,
            self.blocker_combo,
            self.morph_op_combo,
            self.branch_method_combo,
            self.branch_plain_upper_thr_combo,
            self.branch_trunk_combo,
            self.draft_branch_combo,
            self.capture_region_combo,
            self.save_resolution_combo,
            self.save_target_combo,
        ):
            _configure_dock_combo(cb)
        for spin in (
            self.branch_ac_radius_spin,
            self.branch_ac_margin_spin,
            self.branch_ac_sigma_spin,
            self.branch_ac_low_clip_spin,
            self.branch_ac_balloon_spin,
            self.branch_ac_smoothing_spin,
            self.branch_ac_total_iter_spin,
            self.branch_ac_yield_spin,
            self.branch_plain_sigma_spin,
            self.branch_plain_flux_spin,
            self.branch_plain_intensity_tol_spin,
            self.branch_plain_cost_budget_spin,
            self.branch_plain_margin_spin,
            self.branch_plain_step_spin,
        ):
            _configure_dock_spin(spin)
        self.status_label.setMaximumWidth(_DOCK_CONTENT_MAX_WIDTH - 8)
        content = getattr(self, "_dock_content", None)
        if content is not None:
            content.setMaximumWidth(_DOCK_CONTENT_MAX_WIDTH)

    def _apply_combo_width_policies(self) -> None:
        """Backward-compatible alias."""
        self._apply_dock_layout_constraints()

    # ------------------------------------------------------------- signals --
    def _connect_signals(self):
        self.btn_reset_branch_points_layer.clicked.connect(
            self._reset_branch_points_layer
        )
        self.btn_new_branch_points_layer.clicked.connect(
            self._create_new_branch_points_layer
        )
        self.btn_new_segmentation_mask.clicked.connect(
            self._create_new_segmentation_mask_layer
        )
        self.btn_new_blocker_layer.clicked.connect(
            self._create_new_blocker_labels_layer
        )
        self.btn_grow_branches.clicked.connect(self._grow_branches_into_result)
        self.btn_reset_branch_seg.clicked.connect(self._reset_branch_segmentation)
        self.btn_merge_branch_seg.clicked.connect(self._merge_branch_segmentation)
        self.branch_method_combo.currentTextChanged.connect(
            self._update_branch_method_dependent_widgets
        )
        self.branch_plain_upper_thr_check.toggled.connect(
            self.branch_plain_upper_thr_combo.setEnabled
        )
        self.grow_cache_level_check.toggled.connect(self._on_grow_cache_level_toggled)
        self.branch_ac_early_stop_slider.valueChanged.connect(
            self._on_branch_ac_early_stop_slider_changed
        )
        self.btn_postprocess.clicked.connect(self._upsample_result_to_original)
        self.btn_upsample_to_finer.clicked.connect(self._upsample_segmentation_to_finer_level)
        self.btn_apply_morph.clicked.connect(self._apply_morphological_operation)
        self.btn_save_segmentation.clicked.connect(self._save_segmentation_to_omezarr)
        self.save_target_combo.currentIndexChanged.connect(
            self._update_save_target_combo_state
        )
        self.btn_load_saved_segmentation.clicked.connect(
            self._load_saved_segmentation_from_omezarr
        )
        self.btn_stop.clicked.connect(self._stop)
        self.image_combo.currentTextChanged.connect(self._on_image_selection_changed)
        self.ms_level_combo.currentIndexChanged.connect(
            self._on_multiscale_working_level_changed
        )
        self.ms_2d_multiscale_render_check.toggled.connect(
            self._on_ms_2d_multiscale_render_toggled
        )
        self.ms_2d_multiscale_render_check.toggled.connect(
            self._apply_pyramid_dims_navigation
        )
        self.ms_adapt_slice_step_check.toggled.connect(
            self._apply_pyramid_dims_navigation
        )
        self.seg_color_combo.currentTextChanged.connect(self._on_layer_color_changed)
        self.branch_combo.currentIndexChanged.connect(self._on_branch_combo_color_target)
        self.branch_trunk_combo.currentIndexChanged.connect(
            self._on_branch_trunk_combo_color_target
        )
        self.capture_growth_check.toggled.connect(self._on_capture_growth_toggled)
        self.capture_combine_grows_check.toggled.connect(
            self._on_capture_combine_toggled
        )
        self.btn_save_combined_gif.clicked.connect(self._save_combined_growth_gif)
        self.viewer.camera.events.zoom.connect(self._on_camera_for_branch_points)
        self.viewer.camera.events.center.connect(self._on_camera_for_branch_points)

    # -------------------------------------------------------- layer helpers --
    def _on_ms_2d_multiscale_render_toggled(self, *_args: Any) -> None:
        self._branch_point_zoom_ref = None
        self._update_pyramid_display_layers()
        self._sync_branch_point_bases_from_image()

    def _branch_points_use_camera_zoom_scaling(self) -> bool:
        """Only napari auto-multiscale 2D needs zoom-compensated marker sizes."""
        try:
            nd = int(self.viewer.dims.ndisplay)
        except Exception:
            nd = 2
        if nd >= 3:
            return False
        return not self._use_fixed_2d_pyramid_level()

    def _selected_pyramid_level(self) -> int:
        if not self.ms_level_combo.isVisible() or self.ms_level_combo.count() == 0:
            return 0
        return int(self.ms_level_combo.currentIndex())

    def _on_multiscale_working_level_changed(self, *_args: Any) -> None:
        """Keep trunk/blocker combos, display proxies, branch points, and draft preview aligned with the working pyramid level."""
        new_level = int(self._selected_pyramid_level())
        old_level = getattr(self, "_last_pyramid_level", None)
        if old_level is not None and old_level != new_level:
            proxy_name = (
                getattr(self, "_forced_2d_display_layer_name", None)
                or getattr(self, "_forced_3d_display_layer_name", None)
            )
            if proxy_name:
                self._restore_proxy_display_to_source(
                    proxy_name, pyramid_level=int(old_level)
                )
        self._last_pyramid_level = new_level
        self._branch_point_zoom_ref = None
        self._refresh_branch_trunk_combo()
        self._update_pyramid_display_layers()
        self._apply_pyramid_dims_navigation()
        self._apply_dock_layout_constraints()
        self._update_image_validation_ui(show_dialog=False)
        if not self.ms_level_combo.isVisible() or self.ms_level_combo.count() == 0:
            return
        if getattr(self, "_active_branch_job", None):
            return
        self._sync_layers_after_pyramid_working_level_change()

    def _resync_branch_points_to_working_grid(self) -> None:
        """Re-align BranchPoints* to the working pyramid grid (world-fixed)."""
        iname = self._selected_image_layer_name()
        if not iname or iname not in self.viewer.layers:
            return
        img = self.viewer.layers[iname]
        if not isinstance(img, napari.layers.Image):
            return
        level = int(self._selected_pyramid_level())
        try:
            shape_work = tuple(int(x) for x in image_level_shape(img, level))
        except (TypeError, ValueError, IndexError):
            return
        if len(shape_work) != 3:
            return
        skw = spatial_alignment_for_pyramid_level(img, level)
        translate, scale = scale_translate_zyx_from_spatial_kwargs(skw)
        for lyr in list(self.viewer.layers):
            if not isinstance(lyr, napari.layers.Points):
                continue
            if not _is_auto_sized_branch_points_name(lyr.name):
                continue
            if len(lyr.data) == 0:
                _apply_spatial_kwargs_to_layer(lyr, skw)
                continue
            pts = np.asarray(lyr.data, dtype=np.float64)
            world_rows: List[np.ndarray] = []
            for i in range(pts.shape[0]):
                row = np.asarray(pts[i], dtype=np.float64).ravel()
                if row.size < 3:
                    continue
                world_rows.append(
                    np.asarray(lyr.data_to_world(row[:3]), dtype=np.float64).ravel()[:3]
                )
            if not world_rows:
                _apply_spatial_kwargs_to_layer(lyr, skw)
                continue
            world = np.stack(world_rows, axis=0)
            _apply_spatial_kwargs_to_layer(lyr, skw)
            grid = world_to_voxel_indices_zyx(
                world, translate=translate, scale=scale, shape=shape_work
            )
            lyr.data = grid.astype(np.float64)
            self._sync_branch_point_features_layer(lyr)
        self._sync_branch_point_bases_from_image()
        self._on_camera_for_branch_points()

    def _archive_nonempty_live_draft_branch(self) -> None:
        """Rename a non-empty ``Draft_Branch`` so a new draft can be created (any grid shape)."""
        name = DRAFT_BRANCH_LAYER_NAME
        if name not in self.viewer.layers:
            return
        lyr = self.viewer.layers[name]
        if not isinstance(lyr, napari.layers.Labels):
            return
        if not np.any(np.asarray(lyr.data) > 0):
            return
        try:
            existing = dict(getattr(lyr, "color", {}) or {})
            if 1 not in existing:
                self._apply_stored_labels_color(lyr)
        except Exception:
            pass
        k = 1
        base = DRAFT_BRANCH_LAYER_NAME
        while f"{base} ({k})" in self.viewer.layers:
            k += 1
        archived_name = f"{base} ({k})"
        archived_color = _draft_branch_color_for_layer_name(archived_name)
        lyr.name = archived_name
        self._set_layer_color(archived_name, archived_color)
        self._apply_stored_labels_color(lyr, archived_color)
        self._refresh_draft_branch_combo()

    def _sync_layers_after_pyramid_working_level_change(self) -> None:
        """After the user changes the working pyramid index, keep overlays consistent."""
        iname = self._selected_image_layer_name()
        if not iname or iname not in self.viewer.layers:
            return
        img = self.viewer.layers[iname]
        if not isinstance(img, napari.layers.Image):
            return
        if not is_multiscale_image_layer(img):
            return
        level = int(self._selected_pyramid_level())
        try:
            shp = image_level_shape(img, level)
        except (TypeError, ValueError, IndexError):
            return
        if len(shp) != 3:
            return

        self._resync_branch_points_to_working_grid()

        if DRAFT_BRANCH_LAYER_NAME in self.viewer.layers:
            dlyr = self.viewer.layers[DRAFT_BRANCH_LAYER_NAME]
            if layer_data_shape(dlyr) != tuple(shp):
                if np.any(np.asarray(dlyr.data) > 0):
                    self._archive_nonempty_live_draft_branch()
                self._ensure_draft_branch_layer(img, shp, level)
                self.viewer.layers[DRAFT_BRANCH_LAYER_NAME].data = np.zeros(
                    shp, dtype=np.int32
                )
            else:
                self._ensure_draft_branch_layer(img, shp, level)
        else:
            self._ensure_draft_branch_layer(img, shp, level)
            self.viewer.layers[DRAFT_BRANCH_LAYER_NAME].data = np.zeros(
                shp, dtype=np.int32
            )

        if "Skeletal Preview" in self.viewer.layers:
            self._ensure_skeletal_preview_layer(img, shp, level)
            self._clear_skeletal_preview_data(shp)

        seg_msg = ""
        primary = self._primary_segmentation_mask_layer()
        if primary is not None:
            if self._resample_segmentation_mask_to_working_pyramid_level(
                primary, img, level=level, target_shape=tuple(shp)
            ):
                self._prune_orphan_segmentation_masks(primary.name, tuple(shp))
                self._refresh_branch_trunk_combo()
                _select_combo_layer(self.branch_trunk_combo, primary.name)
                seg_msg = " Segmentation resampled to this level."
            else:
                seg_msg = " Could not resample segmentation — pick or create a mask."

        self.status_label.setText(
            "Pyramid level changed — branch points kept; draft preview reset."
            + seg_msg
        )
        self._apply_dock_layout_constraints()

    def _refresh_multiscale_level_combo(self, *_args: Any) -> None:
        iname = self._selected_image_layer_name()
        lyr = None
        if iname and iname in self.viewer.layers:
            cand = self.viewer.layers[iname]
            if isinstance(cand, napari.layers.Image):
                lyr = cand
        if lyr is None:
            self.ms_level_combo.blockSignals(True)
            self.ms_level_combo.clear()
            self.ms_level_combo.blockSignals(False)
            self.ms_level_combo.setVisible(False)
            self._ms_level_row_label.setVisible(False)
            self.ms_2d_multiscale_render_check.setVisible(False)
            self.ms_adapt_slice_step_check.setVisible(False)
            self._reset_pyramid_dims_navigation()
            self._refresh_branch_trunk_combo()
            self._update_pyramid_display_layers()
            return
        n = multiscale_level_count(lyr)
        if n <= 1:
            self.ms_level_combo.blockSignals(True)
            self.ms_level_combo.clear()
            self.ms_level_combo.blockSignals(False)
            self.ms_level_combo.setVisible(False)
            self._ms_level_row_label.setVisible(False)
            self.ms_2d_multiscale_render_check.setVisible(False)
            self.ms_adapt_slice_step_check.setVisible(False)
            self._reset_pyramid_dims_navigation()
            self._refresh_branch_trunk_combo()
            self._update_pyramid_display_layers()
            return
        prev = self.ms_level_combo.currentIndex()
        self.ms_level_combo.blockSignals(True)
        self.ms_level_combo.clear()
        for i in range(n):
            self.ms_level_combo.addItem(multiscale_level_label(lyr, i))
            self.ms_level_combo.setItemData(
                i, multiscale_level_tooltip(lyr, i), Qt.ToolTipRole
            )
        idx = prev if 0 <= prev < n else default_pyramid_level_index(n)
        self.ms_level_combo.setCurrentIndex(idx)
        self.ms_level_combo.blockSignals(False)
        self.ms_level_combo.setVisible(True)
        self._ms_level_row_label.setVisible(True)
        self.ms_2d_multiscale_render_check.setVisible(True)
        self.ms_adapt_slice_step_check.setVisible(True)
        self._refresh_branch_trunk_combo()
        self._update_pyramid_display_layers()
        self._apply_pyramid_dims_navigation()
        self._apply_dock_layout_constraints()

    def _should_adapt_pyramid_slice_step(self) -> bool:
        cb = getattr(self, "ms_adapt_slice_step_check", None)
        if cb is None or not cb.isChecked() or not cb.isVisible():
            return False
        if int(self._selected_pyramid_level()) <= 0:
            return False
        img = self._get_selected_image_layer()
        if img is None or not is_multiscale_image_layer(img):
            return False
        try:
            nd = int(self.viewer.dims.ndisplay)
        except Exception:
            nd = 2
        if nd >= 3:
            return True
        return self._use_fixed_2d_pyramid_level()

    def _display_layer_for_pyramid_navigation(self) -> Optional[Any]:
        """Image actually shown at the locked pyramid level (2D/3D proxy), if any."""
        for attr in ("_forced_2d_display_layer_name", "_forced_3d_display_layer_name"):
            nm = getattr(self, attr, None)
            if nm and nm in self.viewer.layers:
                lyr = self.viewer.layers[nm]
                if isinstance(lyr, napari.layers.Image):
                    return lyr
        return self._get_selected_image_layer()

    def _pyramid_dims_navigation_plan(
        self,
    ) -> Optional[List[Tuple[int, float, float, float]]]:
        """Per-axis ``(axis, lo, hi, world_step)`` for coarse pyramid navigation."""
        if not self._should_adapt_pyramid_slice_step():
            return None
        img = self._get_selected_image_layer()
        if img is None:
            return None
        level = int(self._selected_pyramid_level())
        axis_ranges = pyramid_navigation_axis_ranges_zyx(img, level)
        z_axis = max(0, int(getattr(img, "ndim", 3)) - 3)
        out: List[Tuple[int, float, float, float]] = []
        for i, rng in enumerate(axis_ranges):
            if rng is None:
                continue
            lo, hi, world_step = rng
            out.append((z_axis + i, lo, hi, world_step))
        return out or None

    def _pyramid_dims_world_steps(self) -> Optional[List[Tuple[int, float]]]:
        """Legacy view of :meth:`_pyramid_dims_navigation_plan` (axis, step only)."""
        plan = self._pyramid_dims_navigation_plan()
        if plan is None:
            return None
        return [(axis, step) for axis, _lo, _hi, step in plan]

    def _on_dims_range_changed_for_pyramid_nav(self, event=None) -> None:
        """Napari resets dims.range from layer extents; re-apply our coarser steps."""
        if getattr(self, "_dims_nav_applying", False):
            return
        expected = self._pyramid_dims_navigation_plan()
        if expected is None:
            return
        dims = self.viewer.dims
        for axis, lo, hi, world_step in expected:
            if axis >= dims.ndim:
                break
            try:
                cur_lo, cur_hi, cur = dims.range[axis]
            except (IndexError, TypeError, ValueError):
                continue
            if (
                abs(float(cur_lo) - float(lo)) > 1e-6
                or abs(float(cur_hi) - float(hi)) > 1e-6
                or abs(float(cur or 0.0) - float(world_step)) > 1e-6
            ):
                self._schedule_pyramid_dims_navigation()
                return

    def _on_dims_point_changed_for_pyramid_nav(self, event=None) -> None:
        """Keep dims.point on valid coarse-grid slices (no black past last plane)."""
        if getattr(self, "_dims_nav_applying", False):
            return
        if not getattr(self, "_dims_nav_override", False):
            return
        self._clamp_dims_point_to_ranges()

    def _reset_pyramid_dims_navigation(self) -> None:
        if not self._dims_nav_override:
            return
        saved = self._saved_dims_range
        if saved is not None:
            dims = self.viewer.dims
            snap = self._dims_navigation_snapshot()
            try:
                self._dims_nav_applying = True
                for ax, rng in enumerate(saved):
                    if ax < dims.ndim:
                        dims.set_range(ax, rng)
            except Exception:
                pass
            finally:
                self._dims_nav_applying = False
            self._dims_navigation_restore(snap)
        self._dims_nav_override = False
        self._saved_dims_range = None

    def _clamp_dims_point_to_ranges(self) -> None:
        dims = self.viewer.dims
        pt = [float(x) for x in dims.point]
        changed = False
        plan = self._pyramid_dims_navigation_plan()
        plan_by_axis = (
            {axis: (lo, hi, st) for axis, lo, hi, st in plan} if plan else {}
        )
        for ax in range(int(dims.ndim)):
            try:
                lo, hi, st = dims.range[ax]
            except (IndexError, TypeError, ValueError):
                continue
            if ax in plan_by_axis:
                lo, hi, st = plan_by_axis[ax]
            lo_f, hi_f, st_f = float(lo), float(hi), float(st or 0.0)
            if ax in plan_by_axis and st_f > 0:
                k_max = int(round((hi_f - lo_f) / st_f))
                k = int(round((pt[ax] - lo_f) / st_f))
                k = min(max(k, 0), k_max)
                snapped = lo_f + k * st_f
                if abs(pt[ax] - snapped) > 1e-6:
                    pt[ax] = snapped
                    changed = True
                continue
            if pt[ax] < lo_f:
                pt[ax] = lo_f
                changed = True
            elif pt[ax] > hi_f:
                pt[ax] = hi_f
                changed = True
        if changed:
            try:
                self._dims_nav_applying = True
                dims.point = pt
            except Exception:
                pass
            finally:
                self._dims_nav_applying = False

    def _apply_pyramid_dims_navigation(self, *_args: Any) -> None:
        """Widen napari dims step on Z (and Y/X) when browsing a coarse pyramid grid."""
        expected = self._pyramid_dims_navigation_plan()
        if expected is None:
            self._reset_pyramid_dims_navigation()
            return
        dims = self.viewer.dims
        if not self._dims_nav_override:
            self._saved_dims_range = tuple(tuple(r) for r in dims.range)
        snap = self._dims_navigation_snapshot()
        try:
            self._dims_nav_applying = True
            for axis, lo, hi, world_step in expected:
                if axis >= dims.ndim:
                    break
                dims.set_range(axis, (lo, hi, world_step))
            self._dims_nav_override = True
        except Exception:
            pass
        finally:
            self._dims_nav_applying = False
        self._dims_navigation_restore(snap)
        self._clamp_dims_point_to_ranges()

    def _on_image_selection_changed(self, *_args: Any) -> None:
        self._refresh_multiscale_level_combo()
        self._update_postprocess_button()
        self._sync_branch_point_bases_from_image()
        self._update_pyramid_display_layers()
        self._update_save_target_combo_state()
        self._update_image_validation_ui(show_dialog=True)
        img = self._get_image_layer()
        if img is not None:
            self._ensure_default_segmentation_mask_for_image(img, select=True)

    def _collect_image_validation_issues(self) -> List[Any]:
        img = self._get_image_layer()
        store_path = self._infer_omezarr_store_path(img) if img is not None else None
        store_channels: Optional[int] = None
        if store_path is not None:
            from ._omezarr_reader import omzarr_store_channel_count

            store_channels = omzarr_store_channel_count(store_path)
        return validate_image_layer(
            img,
            pyramid_level=int(self._selected_pyramid_level()),
            channel_split_layer_count=count_channel_split_image_layers(
                list(self.viewer.layers)
            ),
            omzarr_store_channels=store_channels,
        )

    def _update_image_validation_ui(self, *, show_dialog: bool = False) -> None:
        """Show inline warnings and optionally a one-time dialog for blocking issues."""
        issues = self._collect_image_validation_issues()
        blocking = has_blocking_validation_issues(issues)
        msg = summarize_validation_issues(issues)
        lbl = getattr(self, "_image_validation_label", None)
        if lbl is not None:
            if msg:
                lbl.setText(msg)
                lbl.setVisible(True)
            else:
                lbl.clear()
                lbl.setVisible(False)
        grow_btn = getattr(self, "btn_grow_branches", None)
        if grow_btn is not None and getattr(self, "_active_branch_job", None) is None:
            grow_btn.setEnabled(not blocking)
        if show_dialog and blocking and issues:
            key = f"{self._selected_image_layer_name()}:{issues[0].code}"
            if key not in self._validation_dialog_keys:
                self._validation_dialog_keys.add(key)
                QMessageBox.warning(
                    self,
                    "Image not ready for segmentation",
                    msg
                    + "\n\nCompute Branch stays disabled until you load a "
                    "single-channel 3D volume (Z×Y×X). See the README preprocessing section.",
                )

    def _refresh_layers(self, event=None) -> None:
        """Debounced layer-list sync (combos only — not pyramid display proxies)."""
        self._refresh_layers_timer.start(50)

    def _refresh_layers_now(self, event=None) -> None:
        for combo, layer_type in [
            (self.image_combo, napari.layers.Image),
            (self.branch_combo, napari.layers.Points),
        ]:
            if combo is self.image_combo:
                current = self._selected_image_layer_name()
            else:
                current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            for layer in self.viewer.layers:
                if isinstance(layer, layer_type):
                    if combo is self.image_combo:
                        combo.addItem(
                            _elided_layer_combo_text(layer.name), layer.name
                        )
                    else:
                        combo.addItem(layer.name)
            idx = (
                _combo_find_layer_name(combo, current)
                if combo is self.image_combo
                else combo.findText(current)
            )
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        self._refresh_multiscale_level_combo()
        self._update_postprocess_button()
        alive = {lyr.name for lyr in self.viewer.layers}
        for k in list(self._branch_point_size_bases.keys()):
            if k not in alive:
                del self._branch_point_size_bases[k]
        for k in list(self._layer_display_colors.keys()):
            if k not in alive:
                del self._layer_display_colors[k]
        self._refresh_branch_trunk_combo()
        self._refresh_draft_branch_combo()
        self._sync_all_branch_points_palette_colors()
        self._sync_draft_branch_labels_colors()
        self._sync_branch_point_bases_from_image()
        self._prune_empty_archived_draft_layers()
        iname = self._selected_image_layer_name()
        if not iname or iname not in self.viewer.layers:
            self._remove_forced_2d_display_layer(restore_display=False)
            self._remove_forced_3d_display_layer(restore_display=False)
        self._apply_dock_layout_constraints()
        self._update_image_validation_ui(show_dialog=False)

    def _prune_empty_archived_draft_layers(self) -> None:
        """Drop empty ``Draft_Branch (N)`` layers left over from archived previews."""
        to_remove: List[str] = []
        for lyr in self.viewer.layers:
            if not isinstance(lyr, napari.layers.Labels):
                continue
            nm = str(lyr.name)
            if nm == DRAFT_BRANCH_LAYER_NAME:
                continue
            if not nm.startswith(f"{DRAFT_BRANCH_LAYER_NAME} ("):
                continue
            try:
                if not np.any(np.asarray(lyr.data) > 0):
                    to_remove.append(nm)
            except Exception:
                continue
        for nm in to_remove:
            self._safe_remove_viewer_layer(nm)

    def _sync_branch_point_bases_from_image(self) -> None:
        """Assign default marker sizes for BranchPoints* from the current image extent."""
        iname = self._selected_image_layer_name()
        level = int(self._selected_pyramid_level())
        if not iname or iname not in self.viewer.layers:
            self._apply_camera_branch_point_sizes()
            return
        img = self.viewer.layers[iname]
        if not isinstance(img, napari.layers.Image):
            self._apply_camera_branch_point_sizes()
            return
        base = _suggested_branch_point_base_size(img, level)
        sel = self.branch_combo.currentText()
        for lyr in self.viewer.layers:
            if not isinstance(lyr, napari.layers.Points):
                continue
            if not _is_auto_sized_branch_points_name(lyr.name):
                continue
            self._branch_point_size_bases[lyr.name] = base
        self._apply_camera_branch_point_sizes(active_name=sel or None)

    def _on_camera_for_branch_points(self, event=None) -> None:
        """Debounce zoom/pan-driven marker rescaling (avoids work every camera event)."""
        self._camera_points_timer.start(33)

    def _apply_camera_branch_point_sizes(
        self, *, active_name: Optional[str] = None
    ) -> None:
        """Keep BranchPoints markers stable on screen (auto-multiscale 2D only)."""
        if active_name is None:
            active_name = self.branch_combo.currentText()
        if not self._branch_points_use_camera_zoom_scaling():
            for lyr in self.viewer.layers:
                if not isinstance(lyr, napari.layers.Points):
                    continue
                if not _is_auto_sized_branch_points_name(lyr.name):
                    continue
                if active_name and lyr.name != active_name:
                    continue
                if len(lyr.data) == 0:
                    continue
                base = self._branch_point_size_bases.get(lyr.name)
                if base is None:
                    continue
                new_size = float(np.clip(base, 8.0, 900.0))
                if len(lyr.size):
                    cur = float(np.max(lyr.size))
                    if abs(cur - new_size) <= max(0.5, 0.02 * new_size):
                        continue
                lyr.size = new_size
            return
        try:
            z = float(self.viewer.camera.zoom)
        except (TypeError, ValueError):
            return
        if z <= 1e-9 or np.isnan(z):
            return
        if self._branch_point_zoom_ref is None:
            self._branch_point_zoom_ref = z
        z0 = float(self._branch_point_zoom_ref)
        scale = z0 / max(z, 1e-9)
        for lyr in self.viewer.layers:
            if not isinstance(lyr, napari.layers.Points):
                continue
            if not _is_auto_sized_branch_points_name(lyr.name):
                continue
            if active_name and lyr.name != active_name:
                continue
            if len(lyr.data) == 0:
                continue
            base = self._branch_point_size_bases.get(lyr.name)
            if base is None:
                continue
            new_size = float(np.clip(base * scale, 8.0, 900.0))
            if len(lyr.size):
                cur = float(np.max(lyr.size))
                if abs(cur - new_size) <= max(0.5, 0.02 * new_size):
                    continue
            lyr.size = new_size

    def _schedule_deferred_ndisplay_sync(self) -> None:
        """Defer pyramid proxy + label texture work out of the VisPy draw pass."""
        self._ndisplay_sync_pending = True
        self._ndisplay_sync_timer.start()

    def _schedule_mask_colormap_resync(self) -> None:
        """Run mask colormap sync after the 3D image proxy has finished rebuilding."""
        self._labels_colormap_resync_timer.start()

    def _resync_mask_colormaps_after_ndisplay(self) -> None:
        """Lightweight mask display sync after 2D↔3D (no ``.data`` reload — avoids napari loading UI glitches)."""
        try:
            nd = int(self.viewer.dims.ndisplay)
        except (TypeError, ValueError, AttributeError):
            nd = 2

        store = self._labels_show_selected_2d
        for lyr in list(self.viewer.layers):
            if not isinstance(lyr, napari.layers.Labels):
                continue
            if not self._is_segmentation_labels_layer(lyr):
                continue
            nm = str(lyr.name)
            if _is_branch_draft_labels_name(nm):
                color = _draft_branch_color_for_layer_name(nm)
            else:
                color = self._layer_color(nm, DEFAULT_SEGMENTATION_COLOR)
            if nd >= 3:
                if nm not in store:
                    store[nm] = bool(getattr(lyr, "show_selected_label", False))
            else:
                show = store.pop(nm, False)
                try:
                    lyr.show_selected_label = bool(show)
                except Exception:
                    pass
            _sync_mask_colormap_for_ndisplay(lyr, color, nd=nd)
            try:
                lyr.visible = True
            except Exception:
                pass

    def _on_ndisplay_changed(self, event=None) -> None:
        """Keep blocker labels 3D-safe; defer pyramid proxies and label textures."""
        try:
            int(self.viewer.dims.ndisplay)
        except (TypeError, ValueError, AttributeError):
            return
        for lyr in list(self.viewer.layers):
            if isinstance(lyr, napari.layers.Labels) and _is_blocker_labels_name(
                lyr.name
            ):
                _ensure_blocker_labels_ndim3(lyr)

        if getattr(self, "_active_branch_job", None) == "grow":
            self._grow_view_transition = True
        if self._merge_postprocess_active():
            self._merge_ndisplay_deferred = True
        self._schedule_deferred_ndisplay_sync()

    def _finish_deferred_ndisplay_sync(self) -> None:
        """Apply deferred pyramid proxies and label refresh after 2D↔3D switch."""
        self._ndisplay_sync_pending = False
        if getattr(self, "_active_branch_job", None) == "grow":
            try:
                if self._merge_postprocess_active():
                    self._merge_ndisplay_deferred = True
                    return
                self._update_pyramid_display_layers()
                self._schedule_mask_colormap_resync()
            finally:
                self._grow_view_transition = False
                self._apply_pending_grow_step()
            return
        if self._merge_postprocess_active():
            self._merge_ndisplay_deferred = True
            return
        if getattr(self, "_merge_ndisplay_deferred", False):
            self._merge_ndisplay_deferred = False
        self._update_pyramid_display_layers()
        self._schedule_mask_colormap_resync()

    def _apply_pending_grow_step(self) -> None:
        pending = self._pending_grow_step_result
        self._pending_grow_step_result = None
        if pending is not None:
            self._on_step(pending)

    def _clear_grow_view_transition_state(self) -> None:
        self._grow_view_transition = False
        self._pending_grow_step_result = None
        if getattr(self, "_ndisplay_sync_timer", None) is not None:
            self._ndisplay_sync_timer.stop()

    def _update_postprocess_button(self):
        name = self._selected_image_layer_name()
        meta = self._image_working_metadata.get(name)
        enabled = False
        if meta is not None:
            o = tuple(meta.get("finest_shape", ()))
            w = tuple(meta.get("working_shape", ()))
            if o and w and o != w:
                enabled = True
        self.btn_postprocess.setEnabled(enabled)

    def _upsample_result_to_original(self):
        res_layer = self._result_layer
        if res_layer is None or res_layer not in self.viewer.layers:
            res_layer = self._current_segmentation_target_layer()
        if res_layer is None or res_layer not in self.viewer.layers:
            self.status_label.setText(
                "No labels mask on this grid — run Merge first "
                '(e.g. into "Segmentation Result").'
            )
            return

        image_name = self._selected_image_layer_name()
        meta = self._image_working_metadata.get(image_name)
        if meta is not None:
            target_shape = tuple(meta["finest_shape"])
            orig_name = str(meta.get("base_image_name", image_name))
        else:
            self.status_label.setText(
                "No upsample metadata — grow on a coarser pyramid level first."
            )
            self._update_postprocess_button()
            return

        if orig_name not in self.viewer.layers:
            self.status_label.setText(f'Original layer "{orig_name}" is not in the viewer.')
            self._update_postprocess_button()
            return

        mask = np.asarray(res_layer.data) > 0
        if tuple(mask.shape) == target_shape:
            self.status_label.setText(
                "Mask shape already matches the original image — upsampling not needed."
            )
            self._update_postprocess_button()
            return

        zoom = [o / s for o, s in zip(target_shape, mask.shape)]
        upsampled = ndimage_zoom(mask.astype(np.float64), zoom, order=0) > 0.5

        result_name = "Segmentation Result (Original Size)"
        self._register_segmentation_layer_color(result_name)
        if result_name in self.viewer.layers:
            lyr = self.viewer.layers[result_name]
            lyr.data = upsampled.astype(np.int32)
            self._apply_stored_labels_color(lyr)
        else:
            orig_layer = self.viewer.layers[orig_name]
            lyr = self.viewer.add_labels(
                upsampled.astype(np.int32),
                name=result_name,
                opacity=0.5,
                **spatial_alignment_kwargs(orig_layer),
            )
            _init_binary_labels_layer(
                lyr, self._layer_color(result_name, DEFAULT_SEGMENTATION_COLOR)
            )
        self._result_layer = res_layer
        self.status_label.setText("Postprocessing complete: upsampled result created.")

    def _upsample_segmentation_to_finer_level(self) -> None:
        """Create a new editable labels layer on a finer pyramid level (nearest-neighbour)."""
        image_layer = self._get_image_layer()
        if image_layer is None:
            return
        if not is_multiscale_image_layer(image_layer):
            self.status_label.setText("Upsample-to-finer is only available for multiscale (OME-Zarr) images.")
            return
        cur_level = int(self._selected_pyramid_level())
        if cur_level <= 0:
            self.status_label.setText("Already at the finest pyramid level.")
            return
        src_labels = self._branch_trunk_labels_layer()
        if src_labels is None:
            src_labels = self._current_segmentation_target_layer()
        if src_labels is None:
            self.status_label.setText("No segmentation labels on this grid — run Merge first.")
            return
        src = (np.asarray(src_labels.data) > 0).astype(np.uint8)
        target_level = cur_level - 1
        try:
            target_shape = tuple(int(x) for x in image_level_shape(image_layer, target_level))
        except Exception:
            self.status_label.setText("Could not read target pyramid level shape.")
            return
        from ._save_segmentation_zarr import upsample_labels_nearest

        up = upsample_labels_nearest(src, target_shape).astype(np.int32)
        nm_base = f"{src_labels.name} (level {target_level})"
        nm = nm_base
        k = 2
        while nm in self.viewer.layers:
            nm = f"{nm_base} {k}"
            k += 1
        seg_color = self._register_segmentation_layer_color(nm)
        lyr = self.viewer.add_labels(
            up,
            name=nm,
            opacity=0.5,
            **spatial_alignment_for_pyramid_level(image_layer, target_level),
        )
        _init_binary_labels_layer(lyr, seg_color)
        # Switch working level to the new finer grid and select it as merge target.
        try:
            self.ms_level_combo.setCurrentIndex(int(target_level))
        except Exception:
            pass
        self._refresh_branch_trunk_combo()
        _select_combo_layer(self.branch_trunk_combo, nm)
        self._on_branch_trunk_combo_color_target()
        self.status_label.setText(f'Created refined editable layer "{nm}" on pyramid level {target_level}.')

    def _save_resolution_mode(self) -> str:
        """Return ``working`` or ``finest`` from the post-processing combo."""
        if self.save_resolution_combo.currentIndex() == 1:
            return "finest"
        return "working"

    def _save_target_mode(self) -> str:
        """Return ``new``, ``autosave``, ``overwrite_loaded``, or ``choose_folder``."""
        idx = int(self.save_target_combo.currentIndex())
        if idx == _SAVE_TARGET_AUTOSAVE:
            return "autosave"
        if idx == _SAVE_TARGET_OVERWRITE_LOADED:
            return "overwrite_loaded"
        if idx == _SAVE_TARGET_CHOOSE_FOLDER:
            return "choose_folder"
        return "new"

    def _loaded_segmentation_for_current_image(self) -> Optional[Tuple[str, str]]:
        """``(store_path, label_group)`` when load matches the selected image store."""
        src = self._loaded_segmentation_source
        if src is None:
            return None
        image_layer = self._get_image_layer()
        if image_layer is None:
            return None
        inferred = self._infer_omezarr_store_path(image_layer)
        if inferred is None:
            return None
        store_s, label_name = src
        try:
            if Path(store_s).resolve() != Path(inferred).resolve():
                return None
        except OSError:
            if str(store_s) != str(inferred):
                return None
        return (str(Path(store_s).resolve()), str(label_name))

    def _update_save_target_combo_state(self, *_args: Any) -> None:
        """Disable overwrite-loaded when no matching segmentation was loaded."""
        cb = self.save_target_combo
        loaded = self._loaded_segmentation_for_current_image()
        model = cb.model()
        overwrite_item = model.item(_SAVE_TARGET_OVERWRITE_LOADED)
        if overwrite_item is not None:
            overwrite_item.setEnabled(loaded is not None)
            if loaded is None:
                overwrite_item.setToolTip(
                    "Load a saved segmentation for this image first "
                    "(Layers → Load saved segmentation…)."
                )
            else:
                _store, label = loaded
                overwrite_item.setToolTip(
                    f"Overwrite labels/{label} in the loaded image store."
                )
        if loaded is None and cb.currentIndex() == _SAVE_TARGET_OVERWRITE_LOADED:
            cb.blockSignals(True)
            cb.setCurrentIndex(_SAVE_TARGET_NEW)
            cb.blockSignals(False)
        choose_folder = cb.currentIndex() == _SAVE_TARGET_CHOOSE_FOLDER
        self.btn_save_segmentation.setText(
            "Save segmentation to OME-Zarr…"
            if choose_folder
            else "Save segmentation"
        )

    def _resolve_save_destination(
        self, image_layer: Any, save_mode: str
    ) -> Optional[Tuple[str, Optional[str], bool]]:
        """Return ``(store_path, labels_name, use_checkpoint)`` or None if cancelled."""
        from ._omezarr_reader import resolve_label_load_target

        if save_mode == "choose_folder":
            inferred = self._infer_omezarr_store_path(image_layer)
            start_dir = str(inferred) if inferred is not None else str(Path.cwd())
            path = QFileDialog.getExistingDirectory(
                self,
                "Select .ome.zarr store or labels/<name>/ to save into",
                start_dir,
            )
            if not path:
                self.status_label.setText("Save cancelled.")
                return None
            store_path_obj, label_from_path = resolve_label_load_target(Path(path))
            store_path = str(store_path_obj)
            if label_from_path and "__tmp_" in str(label_from_path):
                QMessageBox.warning(
                    self,
                    "Invalid save target",
                    "That path is a temporary staging folder from a failed save.\n"
                    "Select the .ome.zarr root or labels/<name>/ instead.",
                )
                self.status_label.setText(
                    "Save cancelled — pick a label group, not __tmp_."
                )
                return None
            if label_from_path:
                return store_path, str(label_from_path), False
            return store_path, None, False

        inferred = self._infer_omezarr_store_path(image_layer)
        if inferred is None:
            QMessageBox.warning(
                self,
                "OME-Zarr store unknown",
                "Could not infer the .ome.zarr store for the selected image.\n\n"
                "Open the image from its .ome.zarr folder, or use "
                "'Choose folder…' under Save target.",
            )
            self.status_label.setText(
                "Save cancelled — image store path unknown. Use Choose folder…"
            )
            return None
        store_path = str(inferred)

        if save_mode == "autosave":
            return store_path, None, True
        if save_mode == "overwrite_loaded":
            loaded = self._loaded_segmentation_for_current_image()
            if loaded is None:
                self.status_label.setText(
                    "No loaded segmentation to overwrite — load one first or pick "
                    "another save target."
                )
                return None
            _store, label_name = loaded
            return _store, label_name, False
        return store_path, None, False

    def _pick_label_group_dialog(
        self,
        store_path: str,
        *,
        title: str,
        prompt: str,
        recommended: Optional[str] = None,
    ) -> Optional[str]:
        from ._omezarr_reader import (
            _pick_saved_segmentation_label,
            format_label_group_choice,
            list_segmentation_label_groups,
            _ngff_label_names_from_store,
        )

        entries = list_segmentation_label_groups(store_path, check_foreground=False)
        if not entries:
            return None
        labels = [format_label_group_choice(e) for e in entries]
        default_idx = 0
        if recommended:
            for i, e in enumerate(entries):
                if e.get("name") == recommended:
                    default_idx = i
                    break
        else:
            names = _ngff_label_names_from_store(Path(store_path))
            rec = _pick_saved_segmentation_label(
                names, store_path, check_foreground=False
            )
            if rec:
                for i, e in enumerate(entries):
                    if e.get("name") == rec:
                        default_idx = i
                        break
        choice, ok = QInputDialog.getItem(
            self, title, prompt, labels, default_idx, False
        )
        if not ok or not choice:
            return None
        idx = labels.index(choice)
        return str(entries[idx]["name"])

    def _snapshot_segmentation_for_save(
        self, seg_layer: Any, image_layer: Any, save_resolution: str
    ) -> np.ndarray:
        """Dense mask on the working grid (finest upsample is chunked on write)."""
        _ = save_resolution
        if not getattr(seg_layer, "multiscale", False):
            return (np.asarray(seg_layer.data) > 0).astype(np.uint8, copy=False)
        lvl = labels_pyramid_level_for_image_level(
            seg_layer, image_layer, int(self._selected_pyramid_level())
        )
        return (materialize_labels_level(seg_layer, lvl) > 0).astype(
            np.uint8, copy=False
        )

    def _save_segmentation_to_omezarr(self) -> None:
        """Save the current segmentation into an OME-Zarr store as NGFF labels."""
        if self._save_segmentation_worker is not None:
            self.status_label.setText("Save already in progress…")
            return
        image_layer = self._get_image_layer()
        if image_layer is None:
            return
        seg_layer = self._branch_trunk_labels_layer()
        if seg_layer is None:
            seg_layer = self._current_segmentation_target_layer()
        if seg_layer is None:
            self.status_label.setText("No segmentation labels layer selected.")
            return
        save_resolution = self._save_resolution_mode()
        try:
            seg_shape = tuple(
                int(x)
                for x in self._snapshot_segmentation_for_save(
                    seg_layer, image_layer, save_resolution
                ).shape
            )
        except (TypeError, ValueError, IndexError):
            seg_shape = layer_data_shape(seg_layer)
        msg = check_materialization_budget(
            seg_shape,
            np.uint8,
            max_bytes=_MAX_MATERIALIZE_BYTES,
            copies=1.0,
            context="Save segmentation",
        )
        if msg:
            self.status_label.setText(msg)
            return
        save_mode = self._save_target_mode()
        resolved = self._resolve_save_destination(image_layer, save_mode)
        if resolved is None:
            return
        store_path, labels_name, use_checkpoint = resolved
        # Snapshot on the GUI thread: the background worker must not read a live
        # layer the user can keep editing (torn array → corrupt store).
        seg_data = self._snapshot_segmentation_for_save(
            seg_layer, image_layer, save_resolution
        )
        if not np.any(seg_data > 0):
            QMessageBox.warning(
                self,
                "Empty segmentation",
                "The selected mask has no foreground voxels — nothing was written.",
            )
            self.status_label.setText("Save cancelled — segmentation mask is empty.")
            return
        label_color = self._segmentation_label_color_hex(seg_layer.name)
        self.btn_save_segmentation.setEnabled(False)
        self.status_label.setText("Saving segmentation (background)…")

        @thread_worker
        def _save():
            if use_checkpoint:
                from ._save_segmentation_zarr import write_segmentation_checkpoint

                meta = write_segmentation_checkpoint(
                    store_path,
                    seg_data,
                    build_pyramid=False,
                    save_resolution=save_resolution,
                    label_color=label_color,
                )
            else:
                from ._save_segmentation_zarr import write_segmentation_labels_to_ome_zarr

                meta = write_segmentation_labels_to_ome_zarr(
                    store_path,
                    seg_data,
                    build_pyramid=True,
                    save_resolution=save_resolution,
                    label_color=label_color,
                    labels_name=labels_name,
                )
            yield meta

        worker = _save()

        def _done(meta):
            self._save_segmentation_worker = None
            self.btn_save_segmentation.setEnabled(True)
            grp = (meta or {}).get("labels_group", "labels/segmentation")
            levels = (meta or {}).get("levels_written", 1)
            res = (meta or {}).get("save_resolution", save_resolution)
            self.status_label.setText(
                f"Saved segmentation to {grp} ({levels} levels, {res} resolution)."
            )

        def _err(exc):
            self._save_segmentation_worker = None
            self.btn_save_segmentation.setEnabled(True)
            self.status_label.setText(f"Save failed: {exc}")

        worker.yielded.connect(_done)
        worker.errored.connect(_err)
        worker.start()
        self._save_segmentation_worker = worker

    def _load_saved_segmentation_from_omezarr(self) -> None:
        """Add NGFF labels from a store (for workflows that opened the image without our reader)."""
        if self._load_segmentation_worker is not None:
            self.status_label.setText("Load already in progress…")
            return
        image_layer = self._get_image_layer()
        if image_layer is None:
            self.status_label.setText("No image layer selected.")
            return

        inferred = self._infer_omezarr_store_path(image_layer)
        start_dir = str(inferred) if inferred is not None else str(Path.cwd())
        path = QFileDialog.getExistingDirectory(
            self,
            "Select OME-Zarr store, labels/, or labels/<name>/ to load",
            start_dir,
        )
        if not path:
            self.status_label.setText("Load cancelled.")
            return
        from ._omezarr_reader import (
            _label_group_level_shapes,
            _ngff_label_names_from_store,
            _pick_saved_segmentation_label,
            labels_group_has_foreground,
            list_segmentation_label_groups,
            materialize_saved_labels_at_shape,
            resolve_label_load_target,
        )

        store_path, label_name = resolve_label_load_target(Path(path))
        store_s = str(store_path)
        pyramid_level = int(self._selected_pyramid_level())
        try:
            tgt_shape = tuple(
                int(x) for x in image_level_shape(image_layer, pyramid_level)
            )
        except (TypeError, ValueError, IndexError):
            tgt_shape = ()
        if len(tgt_shape) != 3:
            self.status_label.setText("Image must be 3-D.")
            return

        if label_name is not None:
            if not _label_group_level_shapes(store_s, label_name):
                QMessageBox.warning(
                    self,
                    "Label group not found",
                    f'No readable labels group "{label_name}" under\n{store_s}/labels/.',
                )
                self.status_label.setText("Load cancelled — label group missing.")
                return
        else:
            groups = list_segmentation_label_groups(store_s, check_foreground=False)
            if not groups:
                QMessageBox.information(
                    self,
                    "No saved segmentation",
                    "This store has no NGFF labels groups under labels/.",
                )
                self.status_label.setText("No labels found in that store.")
                return
            if len(groups) == 1:
                label_name = str(groups[0]["name"])
            else:
                names = _ngff_label_names_from_store(store_path)
                recommended = _pick_saved_segmentation_label(
                    names, store_s, check_foreground=False
                )
                label_name = self._pick_label_group_dialog(
                    store_s,
                    title="Load saved segmentation",
                    prompt=(
                        "Which labels/ group should be loaded?\n"
                        "(Default prefers segmentation_autosave, else the newest "
                        "segmentation_vN by name.)"
                    ),
                    recommended=recommended,
                )
                if not label_name:
                    self.status_label.setText("Load cancelled.")
                    return

        chosen_name = str(label_name)
        self.btn_load_saved_segmentation.setEnabled(False)
        self.status_label.setText(
            f'Loading labels/{chosen_name} at pyramid level {pyramid_level}…'
        )

        @thread_worker
        def _load():
            loaded = materialize_saved_labels_at_shape(
                store_s, tgt_shape, label_name=chosen_name
            )
            yield loaded

        worker = _load()

        def _done(loaded):
            self._load_segmentation_worker = None
            self.btn_load_saved_segmentation.setEnabled(True)
            if loaded is None:
                QMessageBox.information(
                    self,
                    "No saved segmentation",
                    f'Could not read labels/{chosen_name} from the store.',
                )
                self.status_label.setText("No labels found in that store.")
                return

            data, source_name = loaded
            if not np.any(data > 0):
                QMessageBox.warning(
                    self,
                    "Empty saved segmentation",
                    f'"{source_name}" contains no foreground at pyramid level '
                    f"{pyramid_level} ({tgt_shape}). Try another pyramid level under "
                    "Layers, or re-save with Full finest resolution.",
                )
                self.status_label.setText(
                    f'"{source_name}" is empty at this pyramid level.'
                )
                return

            layer_name = "Segmentation"
            pyramid_level = int(self._selected_pyramid_level())
            lyr = self._replace_segmentation_layer_with_loaded_mask(
                image_layer,
                data,
                pyramid_level=pyramid_level,
            )
            if lyr is None:
                self.status_label.setText("Could not apply loaded segmentation.")
                return
            self._refresh_layers_now()
            _select_combo_layer(self.branch_trunk_combo, layer_name)
            self._on_branch_trunk_combo_color_target()
            self._loaded_segmentation_source = (store_s, chosen_name)
            self._update_save_target_combo_state()
            n_vox = int(np.count_nonzero(data))
            self.status_label.setText(
                f'Loaded "{source_name}" into {layer_name} '
                f"({n_vox:,} voxels, pyramid level {pyramid_level})."
            )

        def _err(exc):
            self._load_segmentation_worker = None
            self.btn_load_saved_segmentation.setEnabled(True)
            self.status_label.setText(
                f"Load failed: {self._worker_exception_message(exc)}"
            )

        worker.yielded.connect(_done)
        worker.errored.connect(_err)
        worker.start()
        self._load_segmentation_worker = worker

    def _apply_morphological_operation(self):
        """Apply morphological dilation or erosion to the result layer."""
        res_layer = self._result_layer
        if res_layer is None or res_layer not in self.viewer.layers:
            res_layer = self._current_segmentation_target_layer()
        if res_layer is None or res_layer not in self.viewer.layers:
            self.status_label.setText(
                "No labels mask on this grid — run Merge first."
            )
            return

        operation = self.morph_op_combo.currentText()
        if operation == "None":
            self.status_label.setText("Select an operation (Dilation or Erosion).")
            return

        radius = self.morph_radius_spin.value()
        mask = np.asarray(res_layer.data) > 0

        # Create ball structuring element
        struct = generate_binary_structure(3, 3)  # 3D, full connectivity
        struct = struct.astype(bool)
        # Scale the structuring element by radius (approx. ball shape)
        if radius > 1:
            # Create larger ball by repeated dilation
            for _ in range(radius - 1):
                struct = binary_dilation(struct, structure=struct)

        # Apply operation
        if operation == "Dilation":
            result = binary_dilation(mask, structure=struct)
            op_name = "Dilation"
        elif operation == "Erosion":
            result = binary_erosion(mask, structure=struct)
            op_name = "Erosion"
        else:
            self.status_label.setText("Unknown operation.")
            return

        result_name = f"Segmentation Result ({op_name} r={radius})"
        if result_name in self.viewer.layers:
            lyr = self.viewer.layers[result_name]
            lyr.data = result.astype(np.int32)
            self._apply_stored_labels_color(lyr)
        else:
            lyr = self.viewer.add_labels(
                result.astype(np.int32),
                name=result_name,
                opacity=0.5,
                **spatial_alignment_kwargs(res_layer),
            )
            _init_binary_labels_layer(
                lyr, self._layer_color(result_name, DEFAULT_SEGMENTATION_COLOR)
            )

        self._result_layer = res_layer
        self.status_label.setText(
            f"Postprocessing complete: {op_name} (radius={radius}) applied."
        )

    def _sync_branch_point_features_layer(self, layer: Any) -> None:
        """Keep ``features`` row count aligned with ``data`` (no branch-label column)."""
        import pandas as pd

        n = len(layer.data)
        layer.features = (
            pd.DataFrame(index=range(n)) if n else pd.DataFrame()
        )

    def _wire_branch_points_sync(self, pts: Any) -> None:
        if hasattr(self, "_branch_pts_sync") and self._branch_pts_sync:
            old_pts, old_cb = self._branch_pts_sync
            if old_pts in self.viewer.layers:
                try:
                    old_pts.events.data.disconnect(old_cb)
                except (AttributeError, RuntimeError, TypeError):
                    pass

        def _sync(ev=None):
            self._sync_branch_point_features_layer(pts)

        pts.events.data.connect(_sync)
        self._branch_pts_sync = (pts, _sync)

    def _reset_branch_points_layer(self):
        import pandas as pd

        sel = self.branch_combo.currentText()
        if not sel or sel not in self.viewer.layers:
            self.status_label.setText(
                "Pick a layer in Branch points layer first, then reset."
            )
            return
        lyr = self.viewer.layers[sel]
        if not isinstance(lyr, napari.layers.Points):
            self.status_label.setText("Branch points layer must be a Points layer.")
            return
        lyr.data = np.empty((0, 3))
        lyr.features = pd.DataFrame()
        self._wire_branch_points_sync(lyr)
        self._refresh_layers()
        self.branch_combo.setCurrentText(sel)
        self.status_label.setText(
            f'"{sel}" cleared — add at least two points in order, then Grow.'
        )

    def _allocate_extra_branch_points_name(self) -> str:
        names = {lyr.name for lyr in self.viewer.layers}
        if "BranchPoints" not in names:
            return "BranchPoints"
        for i in range(1, 1000):
            cand = f"BranchPoints_{i}"
            if cand not in names:
                return cand
        return "BranchPoints_extra"

    def _create_new_branch_points_layer(self):
        import pandas as pd

        name = self._selected_image_layer_name()
        if not name or name not in self.viewer.layers:
            self.status_label.setText("Select an image layer first.")
            return
        img = self.viewer.layers[name]
        spatial_kw = spatial_alignment_for_pyramid_level(
            img, self._selected_pyramid_level()
        )
        self._ensure_default_segmentation_mask_for_image(img, select=True)
        bname = self._allocate_extra_branch_points_name()
        base = _suggested_branch_point_base_size(img, self._selected_pyramid_level())
        self._branch_point_size_bases[bname] = base
        bcolor = self._register_branch_points_layer_color(bname)
        empty_feat = pd.DataFrame()
        pts = self.viewer.add_points(
            np.empty((0, 3)),
            ndim=3,
            name=bname,
            size=base,
            features=empty_feat,
            face_color=bcolor,
            border_color="white",
            **spatial_kw,
        )
        pts.mode = "add"
        self._wire_branch_points_sync(pts)
        self._apply_branch_points_palette_color(bname)
        self._refresh_layers_now()
        _select_combo_layer(self.branch_combo, bname)
        self._on_branch_combo_color_target()
        self.status_label.setText(
            f'Created "{bname}" — add at least two points in order, then Grow.'
        )

    def _on_grow_cache_level_toggled(self, checked: bool) -> None:
        if not checked:
            clear_image_level_cache()

    def _on_branch_ac_early_stop_slider_changed(self, value: int) -> None:
        v = int(value)
        self.branch_ac_early_stop_value_label.setText("off" if v == 0 else str(v))

    def _update_branch_method_dependent_widgets(self) -> None:
        m = self.branch_method_combo.currentText()
        is_plain = m.startswith("Plain")
        is_ac = m.startswith("3D Active Contour")
        self.branch_plain_section.setVisible(is_plain)
        self.branch_ac_section.setVisible(is_ac)
        self.branch_ac_early_stop_row.setVisible(is_ac)
        self._vis_plain_step_label.setVisible(is_plain)
        self._vis_plain_step_field.setVisible(is_plain)
        self._vis_ac_yield_label.setVisible(is_ac)
        self._vis_ac_yield_field.setVisible(is_ac)

    def _ensure_draft_branch_layer(
        self, image_layer: Any, shape: tuple, pyramid_level: int
    ) -> Any:
        """Volatile layer for the current branch computation output."""
        skw = spatial_alignment_for_pyramid_level(image_layer, int(pyramid_level))
        if DRAFT_BRANCH_LAYER_NAME in self.viewer.layers:
            lyr = self.viewer.layers[DRAFT_BRANCH_LAYER_NAME]
            if tuple(lyr.data.shape) != tuple(shape):
                lyr.data = np.zeros(shape, dtype=np.int32)
            _apply_spatial_kwargs_to_layer(lyr, skw)
            try:
                lyr.opacity = 0.7
                preview_color = self._draft_branch_preview_color()
                _configure_binary_labels_layer(lyr)
                self._apply_stored_labels_color(lyr, preview_color)
            except Exception:
                pass
            return lyr
        preview_color = self._draft_branch_preview_color()
        lyr = self.viewer.add_labels(
            np.zeros(shape, dtype=np.int32),
            name=DRAFT_BRANCH_LAYER_NAME,
            opacity=0.7,
            **skw,
        )
        _init_binary_labels_layer(lyr, preview_color)
        return lyr

    def _reset_branch_segmentation(self) -> None:
        """Reset Branch: clears Draft_Branch; keeps branch points and unhides the points layer."""
        if DRAFT_BRANCH_LAYER_NAME in self.viewer.layers:
            lyr = self.viewer.layers[DRAFT_BRANCH_LAYER_NAME]
            if isinstance(lyr, napari.layers.Labels):
                lyr.data = np.zeros_like(np.asarray(lyr.data), dtype=np.int32)
        bname = self.branch_combo.currentText()
        if bname and bname in self.viewer.layers:
            pl = self.viewer.layers[bname]
            if isinstance(pl, napari.layers.Points):
                md = dict(getattr(pl, "metadata", {}) or {})
                if md.get("_rg_hide_empty_2d"):
                    pl.metadata = {
                        k: v for k, v in md.items() if k != "_rg_hide_empty_2d"
                    }
                pl.visible = True
        if self.capture_growth_check.isChecked() and (
            self.capture_combine_grows_check.isChecked()
        ):
            self._gif_capture_pending_segment.clear()
        self.status_label.setText(
            f'Reset: cleared "{DRAFT_BRANCH_LAYER_NAME}"; branch points unchanged and layer shown.'
        )

    def _merge_branch_segmentation(self) -> None:
        tgt = self._branch_trunk_labels_layer()
        if tgt is None:
            tgt = self._ensure_trunk_when_missing()
        if tgt is None:
            self.status_label.setText(
                "Select a segmentation mask (labels, same shape as Image)."
            )
            return
        if _is_branch_draft_labels_name(tgt.name):
            self.status_label.setText(
                f'Cannot merge into "{tgt.name}" — pick a Segmentation_* mask layer.'
            )
            return
        br = self._selected_draft_branch_layer()
        if br is None:
            self.status_label.setText(
                "No branch preview on this grid — run Compute Branch first."
            )
            return
        if tgt is br:
            self.status_label.setText(
                "Merge target cannot be the same as the branch preview — "
                "pick a Segmentation_* mask."
            )
            return
        if tuple(br.data.shape) != tuple(tgt.data.shape):
            self.status_label.setText(
                "Branch preview shape does not match the selected segmentation mask."
            )
            return
        bsel = np.asarray(br.data) > 0
        if not np.any(bsel):
            self.status_label.setText(
                f'"{br.name}" is empty — nothing to merge.'
            )
            return
        br_name = str(br.name)
        res = np.asarray(tgt.data, dtype=np.int32).copy()
        res[bsel] = np.maximum(res[bsel], np.asarray(br.data, dtype=np.int32)[bsel])
        _update_labels_layer_data(tgt, res)
        self._apply_stored_labels_color(tgt)
        self._result_layer = tgt
        self._refresh_draft_branch_combo()
        msg = f'Merged "{br_name}" into "{tgt.name}"; preview cleared.'
        if self.capture_growth_check.isChecked() and (
            self.capture_combine_grows_check.isChecked()
        ):
            pend = list(self._gif_capture_pending_segment)
            if pend:
                self._gif_capture_combined_frames.extend(pend)
                self._gif_capture_pending_segment.clear()
                self._sync_save_combined_gif_button()
                msg += (
                    f" Combined GIF: +{len(pend)} frames "
                    f"(total {len(self._gif_capture_combined_frames)})."
                )
        cleanup = bool(self.merge_cleanup_check.isChecked())
        status_after = msg + (
            " Cleaned up branch preview and points." if cleanup else ""
        )
        self.status_label.setText(msg)
        self._update_postprocess_button()
        self._schedule_autosave_after_merge()
        self._merge_postprocess_payload = {
            "br_name": br_name,
            "cleanup": cleanup,
            "status_msg": status_after,
        }
        self._merge_postprocess_timer.start()

    def _run_deferred_merge_postprocess(self) -> None:
        """Clear branch preview / run CleanUp after the canvas finishes drawing."""
        payload = self._merge_postprocess_payload
        self._merge_postprocess_payload = None
        if not payload:
            return
        if getattr(self, "_grow_view_transition", False) or (
            getattr(self, "_ndisplay_sync_timer", None) is not None
            and self._ndisplay_sync_timer.isActive()
        ):
            self._merge_postprocess_payload = payload
            self._merge_postprocess_timer.start()
            return
        br_name = str(payload.get("br_name") or "")
        cleanup = bool(payload.get("cleanup"))
        self._merge_layers_mutating = True
        try:
            if br_name and br_name in self.viewer.layers:
                will_remove = (
                    cleanup
                    and br_name != DRAFT_BRANCH_LAYER_NAME
                    and _is_branch_draft_labels_name(br_name)
                )
                if not will_remove:
                    br = self.viewer.layers[br_name]
                    if isinstance(br, napari.layers.Labels):
                        try:
                            shp = tuple(int(x) for x in layer_data_shape(br))
                            if len(shp) == 3:
                                _update_labels_layer_data(
                                    br, np.zeros(shp, dtype=np.int32)
                                )
                        except Exception:
                            pass
            if cleanup:
                self._cleanup_layers_after_merge()
        except Exception:
            pass
        finally:
            self._merge_layers_mutating = False
            status = payload.get("status_msg")
            if status:
                self.status_label.setText(str(status))
            if cleanup:
                QTimer.singleShot(50, self._finish_merge_ui_sync)
            else:
                self._activate_branch_points_layer_in_viewer()
            if getattr(self, "_merge_ndisplay_deferred", False):
                self._merge_ndisplay_deferred = False
                QTimer.singleShot(50, self._finish_merge_ndisplay_sync)

    def _finish_merge_ndisplay_sync(self) -> None:
        self._update_pyramid_display_layers()
        self._schedule_mask_colormap_resync()

    def _activate_branch_points_layer_in_viewer(self) -> None:
        """Select the current branch points layer in napari (for placing the next branch)."""
        name = self.branch_combo.currentText().strip()
        if not name or name not in self.viewer.layers:
            name = "BranchPoints"
        if name not in self.viewer.layers:
            return
        lyr = self.viewer.layers[name]
        if not isinstance(lyr, napari.layers.Points):
            return
        try:
            lyr.visible = True
        except Exception:
            pass
        try:
            self.viewer.layers.selection.active = lyr
        except Exception:
            try:
                self.viewer.layers.selection = [lyr]
            except Exception:
                pass

    def _finish_merge_ui_sync(self) -> None:
        """Refresh dock combos after merge cleanup (debounced, post-canvas)."""
        self._refresh_layers()
        _select_combo_layer(self.branch_combo, "BranchPoints")
        self._on_branch_combo_color_target()
        _select_combo_layer(self.draft_branch_combo, DRAFT_BRANCH_LAYER_NAME)
        self._activate_branch_points_layer_in_viewer()

    def _run_deferred_merge_cleanup(self) -> None:
        """Legacy alias — kept for any in-flight singleShot callbacks."""
        self._run_deferred_merge_postprocess()

    def _cleanup_layers_after_merge(self) -> None:
        """Leave a single empty Draft_Branch and BranchPoints after merge."""
        image_layer = self._get_image_layer()
        if image_layer is None:
            return
        level = int(self._selected_pyramid_level())
        try:
            shape_work = tuple(
                int(x) for x in image_level_shape(image_layer, level)
            )
        except (TypeError, ValueError, IndexError):
            shape_work = ()
        if len(shape_work) == 3:
            self._cleanup_draft_branch_layers_after_merge(
                image_layer, shape_work, level
            )
        self._cleanup_branch_points_layers_after_merge()
        self._reset_branch_workflow_colors_after_merge()

    def _cleanup_draft_branch_layers_after_merge(
        self, image_layer: Any, shape_work: tuple, level: int
    ) -> None:
        for lyr in list(self.viewer.layers):
            if not isinstance(lyr, napari.layers.Labels):
                continue
            if not _is_branch_draft_labels_name(lyr.name):
                continue
            if lyr.name == DRAFT_BRANCH_LAYER_NAME:
                continue
            self._safe_remove_viewer_layer(str(lyr.name))
        dlyr = self._ensure_draft_branch_layer(image_layer, shape_work, level)
        _update_labels_layer_data(dlyr, np.zeros(shape_work, dtype=np.int32))

    def _ensure_empty_default_branch_points_layer(self) -> None:
        """Single empty ``BranchPoints`` layer on the current image grid."""
        import pandas as pd

        keep_name = "BranchPoints"
        img = self._get_image_layer()
        if img is not None:
            spatial_kw = spatial_alignment_for_pyramid_level(
                img, self._selected_pyramid_level()
            )
            base = _suggested_branch_point_base_size(img, self._selected_pyramid_level())
        else:
            spatial_kw = {}
            base = 12.0
        self._branch_point_size_bases.setdefault(keep_name, base)
        if keep_name in self.viewer.layers:
            lyr = self.viewer.layers[keep_name]
            if isinstance(lyr, napari.layers.Points):
                lyr.data = np.empty((0, 3))
                lyr.features = pd.DataFrame()
                _apply_spatial_kwargs_to_layer(lyr, spatial_kw)
                self._wire_branch_points_sync(lyr)
                self._set_layer_color(keep_name, DEFAULT_BRANCH_POINTS_COLOR)
                self._apply_color_to_layer_name(
                    keep_name, DEFAULT_BRANCH_POINTS_COLOR
                )
                try:
                    lyr.visible = True
                except Exception:
                    pass
            return
        bcolor = self._register_branch_points_layer_color(keep_name)
        pts = self.viewer.add_points(
            np.empty((0, 3)),
            ndim=3,
            name=keep_name,
            size=base,
            features=pd.DataFrame(),
            face_color=bcolor,
            border_color="white",
            **spatial_kw,
        )
        pts.mode = "add"
        self._wire_branch_points_sync(pts)

    def _cleanup_branch_points_layers_after_merge(self) -> None:
        keep_name = "BranchPoints"
        for lyr in list(self.viewer.layers):
            if not isinstance(lyr, napari.layers.Points):
                continue
            if not _is_auto_sized_branch_points_name(lyr.name):
                continue
            if lyr.name == keep_name:
                continue
            if (
                getattr(self, "_branch_pts_sync", None)
                and self._branch_pts_sync[0] is lyr
            ):
                self._branch_pts_sync = None
            self._branch_point_size_bases.pop(lyr.name, None)
            self._safe_remove_viewer_layer(str(lyr.name))
        self._ensure_empty_default_branch_points_layer()

    def _schedule_autosave_after_merge(self) -> None:
        """Debounced background checkpoint to ``labels/segmentation_autosave``."""
        image_layer = self._get_image_layer()
        if image_layer is None:
            return
        store = self._infer_omezarr_store_path(image_layer)
        if store is None:
            return
        seg_layer = self._branch_trunk_labels_layer()
        if seg_layer is None:
            seg_layer = self._current_segmentation_target_layer()
        if seg_layer is None:
            return
        self._autosave_pending = (str(store), seg_layer.name)
        self._autosave_timer.start(_AUTOSAVE_DEBOUNCE_MS)

    def _run_debounced_autosave(self) -> None:
        pending = self._autosave_pending
        if not pending:
            return
        if self._autosave_worker is not None:
            # A previous checkpoint is still writing. Re-arm so the latest merge
            # state is not silently dropped; retry once the worker frees up.
            self._autosave_timer.start(_AUTOSAVE_DEBOUNCE_MS)
            return
        store_path, seg_layer_name = pending
        if seg_layer_name not in self.viewer.layers:
            return
        seg_layer = self.viewer.layers[seg_layer_name]
        if not isinstance(seg_layer, napari.layers.Labels):
            return
        image_layer = self._get_image_layer()
        # Snapshot on the GUI thread before handing to the background worker so a
        # subsequent paint/merge cannot tear the array being written to disk.
        if image_layer is not None:
            seg_data = self._snapshot_segmentation_for_save(
                seg_layer, image_layer, "working"
            )
        else:
            seg_data = materialize_labels_level(seg_layer, 0)
        label_color = self._segmentation_label_color_hex(seg_layer_name)

        @thread_worker
        def _autosave():
            from ._save_segmentation_zarr import write_segmentation_checkpoint

            meta = write_segmentation_checkpoint(
                store_path,
                seg_data,
                build_pyramid=False,
                save_resolution="working",
                label_color=label_color,
            )
            yield meta

        worker = _autosave()

        def _done(meta):
            self._autosave_worker = None
            grp = (meta or {}).get("labels_group", "labels/segmentation_autosave")
            self.status_label.setText(
                f'Merged; autosaved checkpoint to "{grp}" (background).'
            )

        def _err(exc):
            self._autosave_worker = None

        worker.yielded.connect(_done)
        worker.errored.connect(_err)
        worker.start()
        self._autosave_worker = worker

    def _get_image_layer(self) -> Optional[Any]:
        """Return the selected Image layer or None."""
        name = self._selected_image_layer_name()
        if not name or name not in self.viewer.layers:
            self.status_label.setText("Select an Image layer.")
            return None
        lyr = self.viewer.layers[name]
        if not isinstance(lyr, napari.layers.Image):
            self.status_label.setText("Selected layer is not an Image.")
            return None
        return lyr

    def _allocate_segmentation_mask_name(self) -> str:
        names = {lyr.name for lyr in self.viewer.layers}
        if "Segmentation" not in names:
            return "Segmentation"
        for i in range(1, 1000):
            cand = f"Segmentation_{i}"
            if cand not in names:
                return cand
        return "Segmentation_extra"

    def _primary_segmentation_mask_layer(self) -> Optional[Any]:
        """``Segmentation`` / combo selection, else lowest ``Segmentation_<n>``."""
        name = _combo_layer_name(self.branch_trunk_combo)
        if name and name in self.viewer.layers:
            lyr = self.viewer.layers[name]
            if isinstance(lyr, napari.layers.Labels) and _is_segmentation_mask_numbered_name(
                name
            ):
                return lyr
        if "Segmentation" in self.viewer.layers:
            lyr = self.viewer.layers["Segmentation"]
            if isinstance(lyr, napari.layers.Labels):
                return lyr
        numbered = [
            lyr
            for lyr in self.viewer.layers
            if isinstance(lyr, napari.layers.Labels)
            and _is_segmentation_mask_numbered_name(lyr.name)
        ]
        if not numbered:
            return None
        numbered.sort(key=lambda lyr: _segmentation_mask_name_sort_key(lyr.name))
        return numbered[0]

    def _resample_segmentation_mask_to_working_pyramid_level(
        self,
        mask_layer: Any,
        image_layer: Any,
        *,
        level: Optional[int] = None,
        target_shape: Optional[tuple] = None,
    ) -> bool:
        """Nearest-neighbour resample a segmentation mask onto the working pyramid grid."""
        if level is None:
            level = int(self._selected_pyramid_level())
        if target_shape is None:
            try:
                target_shape = tuple(
                    int(x) for x in image_level_shape(image_layer, level)
                )
            except (TypeError, ValueError, IndexError):
                return False
        if len(target_shape) != 3:
            return False
        from ._save_segmentation_zarr import upsample_labels_nearest

        src = np.asarray(mask_layer.data)
        resampled = upsample_labels_nearest(src, target_shape).astype(np.int32)
        mask_layer.data = resampled
        _apply_spatial_kwargs_to_layer(
            mask_layer, spatial_alignment_for_pyramid_level(image_layer, level)
        )
        self._apply_stored_labels_color(mask_layer)
        if self._result_layer is mask_layer or (
            self._result_layer is None
            and _is_segmentation_mask_numbered_name(mask_layer.name)
        ):
            self._result_layer = mask_layer
        return True

    def _prune_orphan_segmentation_masks(
        self, keep_name: str, target_shape: tuple
    ) -> None:
        """Drop stale pyramid copies and empty auto-created duplicates."""
        tgt = tuple(int(x) for x in target_shape)
        for lyr in list(self.viewer.layers):
            if not isinstance(lyr, napari.layers.Labels):
                continue
            if not _is_segmentation_mask_numbered_name(lyr.name):
                continue
            if lyr.name == keep_name:
                continue
            shp = layer_data_shape(lyr)
            if shp != tgt:
                try:
                    self.viewer.layers.remove(lyr.name)
                except Exception:
                    pass
                continue
            if not np.any(np.asarray(lyr.data) > 0):
                try:
                    self.viewer.layers.remove(lyr.name)
                except Exception:
                    pass

    def _add_empty_segmentation_mask_layer(
        self, image_layer: Any, *, name: Optional[str] = None
    ) -> Optional[Any]:
        try:
            shp = image_level_shape(
                image_layer, self._selected_pyramid_level()
            )
        except (TypeError, ValueError, IndexError):
            return None
        if len(shp) != 3:
            return None
        nm = name or self._allocate_segmentation_mask_name()
        seg_color = self._register_segmentation_layer_color(nm)
        lvl = self._selected_pyramid_level()
        lyr = self.viewer.add_labels(
            np.zeros(shp, dtype=np.int32),
            name=nm,
            opacity=0.5,
            **spatial_alignment_for_pyramid_level(image_layer, lvl),
        )
        _init_binary_labels_layer(lyr, seg_color)
        return lyr

    def _replace_segmentation_layer_with_loaded_mask(
        self,
        image_layer: Any,
        data: np.ndarray,
        *,
        pyramid_level: int,
    ) -> Optional[Any]:
        """Replace the default ``Segmentation`` labels layer with loaded mask data."""
        seg = np.asarray(data, dtype=np.int32)
        if seg.ndim != 3:
            return None
        tgt_shape = tuple(int(x) for x in seg.shape)
        skw = spatial_alignment_for_pyramid_level(image_layer, int(pyramid_level))
        layer_name = "Segmentation"
        self._register_segmentation_layer_color(layer_name)

        existing: Optional[Any] = None
        if layer_name in self.viewer.layers:
            lyr = self.viewer.layers[layer_name]
            if isinstance(lyr, napari.layers.Labels):
                if layer_data_shape(lyr) == tgt_shape:
                    existing = lyr

        if existing is not None:
            existing.data = seg
            _apply_spatial_kwargs_to_layer(existing, skw)
            _configure_binary_labels_layer(existing)
            self._apply_stored_labels_color(existing)
            self._result_layer = existing
            return existing

        if layer_name in self.viewer.layers:
            try:
                self.viewer.layers.remove(layer_name)
            except Exception:
                pass

        lyr = self.viewer.add_labels(
            seg,
            name=layer_name,
            opacity=0.5,
            **skw,
        )
        _init_binary_labels_layer(
            lyr, self._layer_color(layer_name, DEFAULT_SEGMENTATION_COLOR)
        )
        self._result_layer = lyr
        return lyr

    def _has_merge_target_on_grid(self, image_layer: Any, shape: tuple) -> bool:
        shp = tuple(int(x) for x in shape)
        for lyr in self.viewer.layers:
            if not isinstance(lyr, napari.layers.Labels):
                continue
            if layer_data_shape(lyr) != shp:
                continue
            if _is_merge_target_labels_name(lyr.name):
                return True
        return False

    def _ensure_default_segmentation_mask_for_image(
        self, image_layer: Any, *, select: bool = True
    ) -> Optional[Any]:
        """Create ``Segmentation`` (or next free ``Segmentation_N``) when the grid has no merge mask."""
        try:
            shp = image_level_shape(
                image_layer, self._selected_pyramid_level()
            )
        except (TypeError, ValueError, IndexError):
            return None
        if len(shp) != 3:
            return None
        if self._has_merge_target_on_grid(image_layer, shp):
            if select:
                self._refresh_branch_trunk_combo()
            return None
        primary = self._primary_segmentation_mask_layer()
        if primary is not None and layer_data_shape(primary) != tuple(shp):
            if self._resample_segmentation_mask_to_working_pyramid_level(
                primary, image_layer, target_shape=tuple(shp)
            ):
                self._prune_orphan_segmentation_masks(primary.name, tuple(shp))
                if select:
                    self._refresh_branch_trunk_combo()
                    _select_combo_layer(self.branch_trunk_combo, primary.name)
                    self._on_branch_trunk_combo_color_target()
                return primary
        nm = self._allocate_segmentation_mask_name()
        lyr = self._add_empty_segmentation_mask_layer(image_layer, name=nm)
        if lyr is None:
            return None
        if select:
            self._refresh_branch_trunk_combo()
            _select_combo_layer(self.branch_trunk_combo, nm)
            self._on_branch_trunk_combo_color_target()
        return lyr

    def _create_new_segmentation_mask_layer(self) -> None:
        image_layer = self._get_image_layer()
        if image_layer is None:
            return
        nm = self._allocate_segmentation_mask_name()
        lyr = self._add_empty_segmentation_mask_layer(image_layer, name=nm)
        if lyr is None:
            self.status_label.setText("Image must be 3-D.")
            return
        self._refresh_layers()
        _select_combo_layer(self.branch_trunk_combo, nm)
        self._on_branch_trunk_combo_color_target()
        self.status_label.setText(f'New empty mask layer "{nm}".')

    def _allocate_new_blocker_name(self) -> str:
        names = {lyr.name for lyr in self.viewer.layers}
        if BLOCKER_MASK_LAYER_NAME not in names:
            return BLOCKER_MASK_LAYER_NAME
        for i in range(2, 1000):
            cand = f"{BLOCKER_MASK_LAYER_NAME}_{i}"
            if cand not in names:
                return cand
        return f"{BLOCKER_MASK_LAYER_NAME}_extra"

    def _create_new_blocker_labels_layer(self) -> None:
        image_layer = self._get_image_layer()
        if image_layer is None:
            return
        try:
            shp = image_level_shape(
                image_layer, self._selected_pyramid_level()
            )
        except (TypeError, ValueError, IndexError):
            shp = ()
        if len(shp) != 3:
            self.status_label.setText("Image must be 3-D.")
            return
        nm = self._allocate_new_blocker_name()
        lvl = self._selected_pyramid_level()
        lyr = self.viewer.add_labels(
            np.zeros(shp, dtype=np.int32),
            name=nm,
            opacity=0.45,
            **spatial_alignment_for_pyramid_level(image_layer, lvl),
        )
        _ensure_blocker_labels_ndim3(lyr)
        self._refresh_layers()
        _select_combo_layer(self.blocker_combo, nm)
        self.status_label.setText(
            f'New empty blocker "{nm}" — paint foreground where growth must stop.'
        )

    def _ensure_segmentation_result_for_image(self, image_layer: Any) -> None:
        """Create an empty segmentation mask on the current grid if missing."""
        self._ensure_default_segmentation_mask_for_image(image_layer, select=False)

    def _current_segmentation_target_layer(self) -> Optional[Any]:
        """Live result layer used for post-processing and branch attachment."""
        if self._result_layer is not None and self._result_layer in self.viewer.layers:
            return self._result_layer
        tgt = self._branch_trunk_labels_layer()
        if tgt is not None and _is_merge_target_labels_name(tgt.name):
            return tgt
        for lyr in self.viewer.layers:
            if not isinstance(lyr, napari.layers.Labels):
                continue
            if _is_segmentation_mask_numbered_name(lyr.name):
                return lyr
        if MERGED_SEG_LAYER_NAME in self.viewer.layers:
            return self.viewer.layers[MERGED_SEG_LAYER_NAME]
        return None

    def _branch_trunk_labels_layer(self) -> Optional[Any]:
        """Labels layer selected as segmentation mask (merge target / grow context)."""
        name = _combo_layer_name(self.branch_trunk_combo)
        if not name or name not in self.viewer.layers:
            return None
        lyr = self.viewer.layers[name]
        if not isinstance(lyr, napari.layers.Labels):
            return None
        return lyr

    def _selected_draft_branch_layer(self) -> Optional[Any]:
        """Labels layer selected as branch preview source for Merge Branch."""
        name = _combo_layer_name(self.draft_branch_combo)
        if not name or name not in self.viewer.layers:
            return None
        lyr = self.viewer.layers[name]
        if not isinstance(lyr, napari.layers.Labels):
            return None
        if not _is_branch_draft_labels_name(lyr.name):
            return None
        return lyr

    def _refresh_draft_branch_combo(self) -> None:
        """List live + archived ``Draft_Branch*`` layers on the current image grid."""
        cb = self.draft_branch_combo
        cur = _combo_layer_name(cb) if cb.currentIndex() >= 0 else cb.currentText()
        cb.blockSignals(True)
        cb.clear()
        iname = self._selected_image_layer_name()
        if iname in self.viewer.layers:
            img = self.viewer.layers[iname]
            if isinstance(img, napari.layers.Image):
                lvl = self._selected_pyramid_level()
                try:
                    shp = image_level_shape(img, lvl)
                except (TypeError, ValueError, IndexError):
                    shp = ()
                if len(shp) == 3:
                    drafts: List[Any] = []
                    for lyr in self.viewer.layers:
                        if not isinstance(lyr, napari.layers.Labels):
                            continue
                        if not _is_branch_draft_labels_name(lyr.name):
                            continue
                        if layer_data_shape(lyr) != shp:
                            continue
                        drafts.append(lyr)

                    def _draft_sort_key(lyr: Any) -> tuple:
                        nm = str(lyr.name)
                        if nm == DRAFT_BRANCH_LAYER_NAME:
                            return (0, 0, nm)
                        prefix = f"{DRAFT_BRANCH_LAYER_NAME} ("
                        if nm.startswith(prefix) and nm.endswith(")"):
                            try:
                                k = int(nm[len(prefix) : -1])
                            except ValueError:
                                k = 10_000
                            return (1, k, nm)
                        return (2, 0, nm)

                    drafts.sort(key=_draft_sort_key)
                    for lyr in drafts:
                        cb.addItem(
                            _elided_layer_combo_text(lyr.name), lyr.name
                        )
        idx = _combo_find_layer_name(cb, cur)
        if idx >= 0 and _is_branch_draft_labels_name(cur):
            cb.setCurrentIndex(idx)
        else:
            names = [
                str(cb.itemData(i) or cb.itemText(i)) for i in range(cb.count())
            ]
            pref = _preferred_draft_branch_layer_name(names)
            if pref:
                j = _combo_find_layer_name(cb, pref)
                if j >= 0:
                    cb.setCurrentIndex(j)
        cb.blockSignals(False)
        self._sync_draft_branch_labels_colors()

    def _refresh_branch_trunk_combo(self) -> None:
        """Repopulate trunk-mask combo with Labels layers on the current image grid."""
        cb = self.branch_trunk_combo
        cur = _combo_layer_name(cb) if cb.currentIndex() >= 0 else cb.currentText()
        cb.blockSignals(True)
        cb.clear()
        iname = self._selected_image_layer_name()
        if iname in self.viewer.layers:
            img = self.viewer.layers[iname]
            if isinstance(img, napari.layers.Image):
                lvl = self._selected_pyramid_level()
                try:
                    shp = image_level_shape(img, lvl)
                except (TypeError, ValueError, IndexError):
                    shp = ()
                if len(shp) == 3:
                    for lyr in self.viewer.layers:
                        if not isinstance(lyr, napari.layers.Labels):
                            continue
                        if layer_data_shape(lyr) != shp:
                            continue
                        if not _is_merge_target_labels_name(lyr.name):
                            continue
                        cb.addItem(
                            _elided_layer_combo_text(lyr.name), lyr.name
                        )
        idx = _combo_find_layer_name(cb, cur)
        if idx >= 0 and _is_merge_target_labels_name(cur):
            cb.setCurrentIndex(idx)
        else:
            names = [
                str(cb.itemData(i) or cb.itemText(i))
                for i in range(cb.count())
            ]
            pref = _preferred_merge_target_layer_name(names)
            if pref:
                j = _combo_find_layer_name(cb, pref)
                if j >= 0:
                    cb.setCurrentIndex(j)
        cb.blockSignals(False)
        self._refresh_blocker_combo()

    def _refresh_blocker_combo(self) -> None:
        """Labels on the current image pyramid grid; first entry is no blocker."""
        cb = self.blocker_combo
        cur = cb.currentText()
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("(none)")
        iname = self._selected_image_layer_name()
        if iname in self.viewer.layers:
            img = self.viewer.layers[iname]
            if isinstance(img, napari.layers.Image):
                lvl = self._selected_pyramid_level()
                try:
                    shp = image_level_shape(img, lvl)
                except (TypeError, ValueError, IndexError):
                    shp = ()
                if len(shp) == 3:
                    for lyr in self.viewer.layers:
                        if isinstance(lyr, napari.layers.Labels):
                            if layer_data_shape(lyr) == shp:
                                cb.addItem(
                                    _elided_layer_combo_text(lyr.name), lyr.name
                                )
                            if _is_blocker_labels_name(lyr.name):
                                _ensure_blocker_labels_ndim3(lyr)
        cur_idx = cb.findText(cur)
        if cur_idx < 0:
            cur_idx = _combo_find_layer_name(cb, cur)
        cb.setCurrentIndex(cur_idx if cur_idx >= 0 else 0)
        cb.blockSignals(False)

    def _ensure_trunk_when_missing(self) -> Optional[Any]:
        """Return the selected segmentation mask, or None if none exist on this grid."""
        self._refresh_branch_trunk_combo()
        cb = self.branch_trunk_combo
        if cb.count() == 0:
            return None
        name = cb.currentText()
        if not name or name not in self.viewer.layers:
            cb.blockSignals(True)
            cb.setCurrentIndex(0)
            cb.blockSignals(False)
        return self._branch_trunk_labels_layer()

    def _maybe_archive_nonempty_branch_preview(
        self, image_layer: Any, shape: tuple
    ) -> None:
        """If the live draft layer already has foreground, rename it so Compute can start fresh."""
        name = DRAFT_BRANCH_LAYER_NAME
        if name not in self.viewer.layers:
            return
        lyr = self.viewer.layers[name]
        if layer_data_shape(lyr) != tuple(shape):
            return
        if not np.any(np.asarray(lyr.data) > 0):
            return
        try:
            existing = dict(getattr(lyr, "color", {}) or {})
            if 1 not in existing:
                self._apply_stored_labels_color(lyr)
        except Exception:
            pass
        k = 1
        base = DRAFT_BRANCH_LAYER_NAME
        while f"{base} ({k})" in self.viewer.layers:
            k += 1
        archived_name = f"{base} ({k})"
        archived_color = _draft_branch_color_for_layer_name(archived_name)
        lyr.name = archived_name
        self._set_layer_color(archived_name, archived_color)
        self._apply_stored_labels_color(lyr, archived_color)
        self._refresh_draft_branch_combo()

    def _finish_branch_grow_preview_if_needed(self) -> None:
        if not getattr(self, "_pending_branch_grow_cleanup", False):
            return
        self._pending_branch_grow_cleanup = False
        if "Skeletal Preview" in self.viewer.layers:
            self.viewer.layers["Skeletal Preview"].visible = False

    def _register_skeletal_preview_vispy_if_needed(self, lyr: Any) -> None:
        """VisPy only maps layers that were shown at least once; hidden-at-create breaks layer list."""
        if self._skeletal_preview_vispy_registered:
            return
        lyr.visible = True
        QApplication.processEvents()
        lyr.visible = False
        self._skeletal_preview_vispy_registered = True

    def _apply_skeletal_preview_style(self, lyr: Any) -> None:
        """Keep Skeletal Preview binary: only label 1 visible, not the background."""
        if not isinstance(lyr, napari.layers.Labels):
            return
        try:
            lyr.colormap = _skeletal_preview_colormap()
            lyr.opacity = 0.55
            lyr.rendering = "translucent"
            _ensure_labels_show_selected(lyr)
        except Exception:
            pass

    def _ensure_skeletal_preview_layer(
        self, image_layer: Any, shape: tuple, pyramid_level: int
    ) -> None:
        skw = spatial_alignment_for_pyramid_level(image_layer, int(pyramid_level))
        if "Skeletal Preview" in self.viewer.layers:
            pl = self.viewer.layers["Skeletal Preview"]
            pl.data = np.zeros(shape, dtype=np.uint32)
            _apply_spatial_kwargs_to_layer(pl, skw)
            self._apply_skeletal_preview_style(pl)
            self._register_skeletal_preview_vispy_if_needed(pl)
            pl.visible = False
            self._preview_layer = pl
            return
        lbl = self.viewer.add_labels(
            np.zeros(shape, dtype=np.uint32),
            name="Skeletal Preview",
            opacity=0.55,
            colormap=_skeletal_preview_colormap(),
            rendering="translucent",
            **skw,
        )
        _ensure_labels_show_selected(lbl)
        self._preview_layer = lbl
        self._skeletal_preview_vispy_registered = False
        self._register_skeletal_preview_vispy_if_needed(lbl)

    def _clear_skeletal_preview_data(self, shape: tuple) -> None:
        if "Skeletal Preview" not in self.viewer.layers:
            return
        self.viewer.layers["Skeletal Preview"].data = np.zeros(shape, dtype=np.uint32)

    # ----------------------------------------------------------- execution --
    def _branch_plain_upper_threshold_method(self) -> Optional[str]:
        """Read the upper-threshold method from the UI (must run on the GUI thread).

        Returns the method key (``'otsu'`` …) or ``None`` if the upper threshold
        is disabled.  The image-dependent value is computed later in the worker
        via :func:`holvesseg._algorithm.compute_upper_threshold`; this keeps all
        Qt widget access on the main thread.
        """
        if not self.branch_plain_upper_thr_check.isChecked():
            return None
        _method_map = {
            "Otsu": "otsu",
            "Triangle": "triangle",
            "Li": "li",
            "90th percentile": "p90",
            "95th percentile": "p95",
        }
        return _method_map[self.branch_plain_upper_thr_combo.currentText()]

    def _maybe_append_growth_capture_frame(self) -> None:
        if not getattr(self, "_growth_capture_run", False):
            return
        if getattr(self, "_active_branch_job", None) != "grow":
            return
        self._growth_capture_step_counter += 1
        n_sub = max(1, int(self.capture_subsample_spin.value()))
        if (self._growth_capture_step_counter % n_sub) != 0:
            return
        max_f = int(self.capture_max_frames_spin.value())
        if max_f > 0 and len(self._growth_capture_frames) >= max_f:
            if not self._growth_capture_hit_max:
                self._growth_capture_hit_max = True
            return
        if self._growth_capture_bytes >= _MAX_GIF_CAPTURE_BYTES:
            # RAM ceiling reached: stop buffering frames but let the grow finish.
            if not self._growth_capture_hit_max:
                self._growth_capture_hit_max = True
            return
        QApplication.processEvents()
        canvas_only = (
            self.capture_region_combo.currentText() == "Viewer canvas"
        )
        scale_pct = float(self.capture_scale_spin.value())
        from ._growth_capture import capture_viewer_frame

        img = capture_viewer_frame(
            self.viewer,
            canvas_only=canvas_only,
            scale_percent=scale_pct,
        )
        if img is not None:
            self._growth_capture_frames.append(img)
            self._growth_capture_bytes += int(getattr(img, "nbytes", 0))

    def _finalize_growth_capture(self, had_error: bool) -> bool:
        """Return True if a background GIF encode was started."""
        if getattr(self, "_growth_capture_finalized", False):
            return False
        if not getattr(self, "_growth_capture_run", False):
            return False
        self._growth_capture_finalized = True
        self._growth_capture_run = False
        frames = list(self._growth_capture_frames)
        self._growth_capture_frames.clear()
        self._growth_capture_bytes = 0
        hit_max = self._growth_capture_hit_max
        self._growth_capture_hit_max = False

        combine = getattr(self, "_gif_capture_combine_this_run", False)

        if not frames:
            if had_error:
                return False
            msg = (
                "GIF capture: no frames captured — enable Animate growth or set "
                "subsample to 1."
            )
            self.status_label.setText(msg)
            return False

        if combine:
            if had_error:
                return False
            self._gif_capture_pending_segment = list(frames)
            extra_hit = (
                " (max frame cap reached; grow continued)" if hit_max else ""
            )
            self.status_label.setText(
                f'Preview written to "{DRAFT_BRANCH_LAYER_NAME}". '
                "Combined GIF: last grow buffered — Merge to append, or Reset branch "
                f"preview to discard.{extra_hit}"
            )
            return False

        default_name = (
            f"vessel_growth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        )
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Save growth GIF",
            default_name,
            "GIF (*.gif);;All files (*)",
        )
        if not path:
            self.status_label.setText("GIF capture cancelled (no file chosen).")
            return False
        if not path.lower().endswith(".gif"):
            path = path + ".gif"

        fps = float(self.capture_output_fps_spin.value())
        self._start_gif_encode_worker(
            frames, path, fps, hit_max, had_error, clear_combined_after=False
        )
        return True

    def _start_gif_encode_worker(
        self,
        frames: List[np.ndarray],
        path: str,
        fps: float,
        hit_max: bool,
        had_error: bool,
        *,
        clear_combined_after: bool = False,
    ) -> None:
        err_prefix = "Partial GIF (grow had errors): " if had_error else ""
        ch = int(self.capture_gif_canvas_h_spin.value())
        cw = int(self.capture_gif_canvas_w_spin.value())

        @thread_worker
        def _encode():
            from ._growth_capture import save_gif

            save_gif(frames, path, fps, canvas_height=ch, canvas_width=cw)
            yield path

        worker = _encode()

        def _on_enc_done(result):
            p = result if isinstance(result, str) else str(result)
            extra = " — max frame cap reached; grow may have continued" if hit_max else ""
            if clear_combined_after:
                self.status_label.setText(
                    f"Saved combined GIF ({len(frames)} frames): {p}"
                )
                self._gif_capture_combined_frames.clear()
                self._sync_save_combined_gif_button()
            else:
                self.status_label.setText(
                    f"{err_prefix}Saved GIF ({len(frames)} frames){extra}: {p}"
                )

        def _on_enc_err(exc):
            self.status_label.setText(
                f"GIF save failed: {self._worker_exception_message(exc)}"
            )

        worker.yielded.connect(_on_enc_done)
        worker.errored.connect(_on_enc_err)
        worker.finished.connect(lambda: setattr(self, "_gif_encode_worker", None))
        worker.start()
        self._gif_encode_worker = worker

    def _grow_branches_into_result(self):
        image_layer = self._get_image_layer()
        if image_layer is None:
            return

        validation_issues = self._collect_image_validation_issues()
        if has_blocking_validation_issues(validation_issues):
            msg = summarize_validation_issues(validation_issues)
            self.status_label.setText(msg)
            self._update_image_validation_ui(show_dialog=True)
            return

        branch_name = self.branch_combo.currentText()
        if not branch_name or branch_name not in self.viewer.layers:
            self.status_label.setText(
                'Pick a branch points layer (e.g. "BranchPoints").'
            )
            return
        branch_layer = self.viewer.layers[branch_name]
        if len(branch_layer.data) < 1:
            self.status_label.setText("Place at least one branch tip.")
            return

        tgt = self._branch_trunk_labels_layer()
        if tgt is None:
            tgt = self._ensure_trunk_when_missing()
        if tgt is None:
            self.status_label.setText(
                "Select a segmentation mask (labels, same shape as Image)."
            )
            return
        if _is_branch_draft_labels_name(tgt.name):
            self.status_label.setText(
                f'Cannot grow using "{tgt.name}" as mask — select a Segmentation_* layer.'
            )
            return

        level = self._selected_pyramid_level()

        shape_fine = image_finest_shape(image_layer)
        shape_work = image_level_shape(image_layer, level)
        if len(shape_work) != 3:
            self.status_label.setText("Image must be 3-D.")
            return

        if layer_data_shape(tgt) != shape_work:
            self.status_label.setText(
                "Segmentation mask shape must match the selected image pyramid level."
            )
            return

        branch_idx = _branch_points_to_working_grid_zyx(
            image_layer, branch_layer, level, shape_work
        )
        if branch_idx.shape[0] < 2:
            self.status_label.setText(
                "Need at least two branch points in click order along one branch."
            )
            return

        method = self.branch_method_combo.currentText()

        bname = self.blocker_combo.currentText()
        if bname and bname != "(none)":
            if bname not in self.viewer.layers:
                self.status_label.setText("Selected blocker layer is missing — pick another or (none).")
                return
            bl = self.viewer.layers[bname]
            if not isinstance(bl, napari.layers.Labels):
                self.status_label.setText("Blocker mask must be a labels layer.")
                return
            if layer_data_shape(bl) != shape_work:
                self.status_label.setText(
                    "Blocker labels must match the selected image pyramid level shape."
                )
                return

        lazy_img = image_level_is_lazy(image_layer, level)
        if lazy_img:
            QApplication.processEvents()
            self.status_label.setText(
                "Compute Branch: loading the full pyramid level from Zarr/Dask. "
                "If it is still too large, pick a coarser pyramid level."
            )

        # Full working level is materialized in the worker (no polyline crop).

        self._image_working_metadata[image_layer.name] = {
            "finest_shape": tuple(shape_fine),
            "working_shape": tuple(shape_work),
            "level": int(level),
            "base_image_name": image_layer.name,
        }

        self._ensure_skeletal_preview_layer(image_layer, shape_work, level)

        # If a previous preview exists and wasn't reset/merged, archive it so
        # a new Compute Branch run starts with a fresh Draft_Branch layer.
        self._maybe_archive_nonempty_branch_preview(image_layer, shape_work)

        draft_layer = self._ensure_draft_branch_layer(image_layer, shape_work, level)
        draft_layer.data = np.zeros(shape_work, dtype=np.int32)
        try:
            draft_layer.visible = True
        except Exception:
            pass
        self._branch_step_target_layer = draft_layer
        self._pending_branch_grow_cleanup = True
        self._last_connect_shape = shape_work
        self._active_branch_job = "grow"
        self._branch_grow_image_layer = image_layer
        self._branch_grow_pyramid_level = int(level)

        self._growth_capture_finalized = False
        self._growth_capture_run = self.capture_growth_check.isChecked()
        self._gif_capture_combine_this_run = (
            self._growth_capture_run
            and self.capture_combine_grows_check.isChecked()
        )
        if self._growth_capture_run:
            self._growth_capture_frames = []
            self._growth_capture_step_counter = 0
            self._growth_capture_hit_max = False
            self._growth_capture_bytes = 0
        if self._gif_capture_combine_this_run:
            self._gif_capture_pending_segment.clear()

        self.btn_stop.setEnabled(True)
        self.btn_grow_branches.setEnabled(False)
        self.progress_bar.show()

        spacing = voxel_spacing_zyx_for_level(image_layer, level, shape_work)
        radius_ui = float(self.branch_ac_radius_spin.value())
        radius_work = tube_radius_voxels_for_work_level(
            radius_ui, image_layer, level, shape_work
        )
        b_margin_ui = float(self.branch_plain_margin_spin.value())
        b_margin = axis_margin_voxels_for_work_level(
            b_margin_ui, image_layer, level, shape_work
        )
        ac_margin_ui = float(self.branch_ac_margin_spin.value())
        ac_margin = axis_margin_voxels_for_work_level(
            ac_margin_ui, image_layer, level, shape_work
        )
        want_animate = (
            self.animate_check.isChecked() or self.capture_growth_check.isChecked()
        )
        plain_yield = (
            self.branch_plain_step_spin.value()
            if want_animate
            else 10**9
        )
        b_cost = self.branch_plain_cost_budget_spin.value()
        rg_params = dict(
            sigma=self.branch_plain_sigma_spin.value(),
            spacing=spacing,
            cost_budget=b_cost if b_cost > 0 else None,
            flux_weight=self.branch_plain_flux_spin.value(),
            intensity_tolerance=self.branch_plain_intensity_tol_spin.value(),
            margin=b_margin,
            upper_threshold=None,
        )
        _bac_total = self.branch_ac_total_iter_spin.value()
        _bac_yield = (
            self.branch_ac_yield_spin.value()
            if want_animate
            else _bac_total
        )
        # MGAC branch corridor / effective margin use branch_ac_margin_spin.
        ac_params = dict(
            radius=radius_work,
            spacing=spacing,
            sigma=self.branch_ac_sigma_spin.value(),
            low_intensity_equalize_below=self.branch_ac_low_clip_spin.value(),
            balloon=self.branch_ac_balloon_spin.value(),
            smoothing=self.branch_ac_smoothing_spin.value(),
            total_iter=_bac_total,
            yield_every=_bac_yield,
            margin=ac_margin,
            early_stop_stable_yields=int(self.branch_ac_early_stop_slider.value()),
        )
        bn = self.blocker_combo.currentText()
        blocker_full = None
        if bn and bn != "(none)" and bn in self.viewer.layers:
            bl = self.viewer.layers[bn]
            if isinstance(bl, napari.layers.Labels):
                blocker_full = np.asarray(bl.data) > 0

        _image_layer = image_layer
        _level = int(level)
        _shape_work = tuple(int(x) for x in shape_work)
        _spacing = spacing
        _branch_idx_arr = branch_idx
        _method = method
        _plain_yield = plain_yield
        _rg_params = rg_params
        _ac_params = ac_params
        _blocker_full = blocker_full
        _branch_radius = float(radius_work)
        _ac_margin = float(ac_margin)
        _ac_sigma = float(ac_params["sigma"])
        _grow_use_roi = bool(self.grow_use_roi_check.isChecked())
        _use_level_cache = bool(self.grow_cache_level_check.isChecked())
        # Read all Qt widget state on the GUI thread; the worker must not touch
        # widgets. The image-dependent threshold value is computed in the worker.
        _upper_thr_method = self._branch_plain_upper_threshold_method()

        @thread_worker
        def _work():
            from ._algorithm import (
                compute_upper_threshold,
                polyline_corridor_mask,
                polyline_to_line_mask,
                polyline_tube_mask,
                region_grow,
            )
            from ._active_contour import active_contour_grow

            poly = _branch_idx_arr.astype(np.int64)
            margin_for_roi = (
                _ac_margin if _method == "3D Active Contour" else _rg_params["margin"]
            )
            sz, sy, sx = _shape_work
            if _grow_use_roi:
                roi_sl = grow_roi_slices_zyx(
                    _shape_work,
                    poly,
                    _spacing,
                    _branch_radius,
                    margin_for_roi,
                    edge_sigma_phys=_ac_sigma if _method == "3D Active Contour" else 0.0,
                )
                load_slices = roi_sl
                budget_context = "Compute Branch (polyline ROI)"
            else:
                roi_sl = (slice(0, sz), slice(0, sy), slice(0, sx))
                load_slices = None
                budget_context = "Compute Branch (full pyramid level)"
            roi_shp = roi_shape_from_slices(roi_sl)

            msg = check_materialization_budget(
                roi_shp,
                _GROW_WORK_DTYPE,
                max_bytes=_MAX_MATERIALIZE_BYTES,
                copies=3.5,
                context=budget_context,
            )
            if msg:
                hint = (
                    " (Enable polyline ROI under Layers, or pick a coarser pyramid level.)"
                    if not _grow_use_roi
                    else " (Pick a coarser pyramid level or fewer points.)"
                )
                yield ("error", msg + hint)
                return

            img_roi = materialize_image_level_cached(
                _image_layer,
                _level,
                dtype=_GROW_WORK_DTYPE,
                slices=load_slices,
                use_cache=_use_level_cache,
            )
            if not np.any(np.isfinite(img_roi)):
                yield (
                    "error",
                    "Error: Loaded image has no finite values in the grow region "
                    "(NaN/inf). Check the Zarr level / contrast or point placement.",
                )
                return

            if _grow_use_roi:
                poly_loc = polyline_to_roi_local(poly, roi_sl)
                blocker_sub = (
                    crop_bool_mask_to_roi(_blocker_full, roi_sl)
                    if _blocker_full is not None
                    else None
                )
            else:
                poly_loc = poly.astype(np.int64)
                blocker_sub = (
                    np.asarray(_blocker_full, dtype=bool)
                    if _blocker_full is not None
                    else None
                )

            tube_preview_roi = polyline_tube_mask(
                img_roi.shape, poly_loc, _branch_radius, _spacing
            )
            if _grow_use_roi:
                tube_preview_out = paste_roi_mask_into_full(
                    _shape_work, roi_sl, tube_preview_roi
                )
            else:
                tube_preview_out = tube_preview_roi
            yield -1, tube_preview_out

            line_m = polyline_to_line_mask(img_roi.shape, poly_loc)

            if not np.any(tube_preview_roi):
                yield ("error", "Seed tube is empty — increase tube radius or check points.")
                return

            start_f = poly_loc[0].astype(np.float64)
            end_f = poly_loc[-1].astype(np.float64)

            def _embed(mask_roi: np.ndarray) -> np.ndarray:
                m = np.asarray(mask_roi, dtype=bool)
                if _grow_use_roi:
                    return paste_roi_mask_into_full(_shape_work, roi_sl, m)
                return m

            if _method == "Plain Region Growing":
                bm_rg = _branch_effective_margin(line_m, _rg_params["margin"], _spacing)
                upper_thr = (
                    compute_upper_threshold(img_roi, _upper_thr_method)
                    if _upper_thr_method is not None
                    else None
                )
                rg_local = {**_rg_params, "margin": bm_rg, "upper_threshold": upper_thr}
                fb = blocker_sub if (blocker_sub is not None and np.any(blocker_sub)) else None
                for step, m in region_grow(
                    img_roi,
                    tube_preview_roi,
                    start_f,
                    end_f,
                    yield_every=_plain_yield,
                    stats_seed_mask=line_m,
                    forbidden_mask=fb,
                    **rg_local,
                ):
                    yield step, _embed(np.asarray(m, dtype=bool))
                return

            bm_ac = _branch_effective_margin(line_m, _ac_params["margin"], _spacing)
            ac_local = {**_ac_params, "margin": bm_ac}
            corridor = polyline_corridor_mask(
                img_roi.shape,
                poly_loc,
                _spacing,
                bm_ac,
                _branch_radius,
            )
            dummy_seed = np.zeros(img_roi.shape, dtype=bool)
            init_ls = tube_preview_roi & corridor
            gen = active_contour_grow(
                img_roi,
                dummy_seed,
                start_f,
                end_f,
                blocker_mask=blocker_sub,
                init_level_set=init_ls,
                corridor_mask=corridor,
                **ac_local,
            )
            for it, m in gen:
                yield it, _embed(np.asarray(m, dtype=bool))

        worker = _work()
        worker.yielded.connect(self._on_step)
        worker.errored.connect(self._on_worker_error)
        worker.finished.connect(self._on_finished)
        worker.start()
        self._worker = worker
        self.status_label.setText("Growing…")

    def _on_worker_error(self, exc):
        self._clear_grow_view_transition_state()
        self._finish_branch_grow_preview_if_needed()
        self.btn_stop.setEnabled(False)
        self.btn_grow_branches.setEnabled(True)
        self.progress_bar.hide()
        self._worker = None
        self.status_label.setText(f"Error: {self._worker_exception_message(exc)}")
        self._branch_step_target_layer = None

    def _on_step(self, result):
        if getattr(self, "_grow_view_transition", False):
            self._pending_grow_step_result = result
            return
        self._apply_grow_step_result(result)

    def _apply_grow_step_result(self, result: Any) -> None:
        if isinstance(result, tuple) and len(result) == 2 and result[0] == "error":
            self.status_label.setText(str(result[1]))
            return
        iteration, mask = result
        if iteration >= 0:
            dl = (
                self._branch_step_target_layer
                if getattr(self, "_active_branch_job", None) == "grow"
                else self._result_layer
            )
            if dl is None:
                return
            if isinstance(mask, tuple) and len(mask) >= 1:
                draft_b = mask[0]
            else:
                draft_b = mask
            _update_labels_layer_data(dl, draft_b)
            n_voxels = int(np.asarray(draft_b, dtype=bool).sum())
            if getattr(self, "_active_branch_job", None) == "grow":
                self.status_label.setText(
                    f"Step {iteration} — {n_voxels:,} voxels (preview)"
                )
                self._maybe_append_growth_capture_frame()
            else:
                self.status_label.setText(
                    f"Step {iteration} — {n_voxels:,} voxels segmented"
                )
            return
        if iteration < 0:
            if "Skeletal Preview" in self.viewer.layers:
                sp = self.viewer.layers["Skeletal Preview"]
                sp.data = (np.asarray(mask) > 0).astype(np.uint32)
                self._apply_skeletal_preview_style(sp)
                sp.visible = True
            self.status_label.setText(
                f"Branch tube preview — {int(mask.sum()):,} voxels"
            )
            if getattr(self, "_active_branch_job", None) == "grow":
                self._maybe_append_growth_capture_frame()
            return

    def _on_finished(self):
        self._clear_grow_view_transition_state()
        self.btn_stop.setEnabled(False)
        self.btn_grow_branches.setEnabled(True)
        self.progress_bar.hide()
        self._finish_branch_grow_preview_if_needed()
        st = self.status_label.text()
        job = getattr(self, "_active_branch_job", None)
        self._active_branch_job = None
        self._branch_step_target_layer = None
        err = st.startswith("Error:") or st.startswith("Branch step")
        combine_run = getattr(self, "_gif_capture_combine_this_run", False)
        enc_started = False
        if job == "grow":
            enc_started = self._finalize_growth_capture(had_error=err)
        if job == "grow" and not err:
            self._refresh_draft_branch_combo()
            _select_combo_layer(self.draft_branch_combo, DRAFT_BRANCH_LAYER_NAME)
            self._sync_draft_branch_labels_colors()
            if enc_started:
                self.status_label.setText(
                    f'Preview written to "{DRAFT_BRANCH_LAYER_NAME}". Encoding GIF in background…'
                )
            elif not combine_run:
                self.status_label.setText(
                    f'Preview written to "{DRAFT_BRANCH_LAYER_NAME}". Use Merge Branch when satisfied.'
                )
        elif st.startswith("Step"):
            self.status_label.setText(st + "  ✓ Done")
        elif not err:
            self.status_label.setText("Done")
        if job == "grow":
            self._gif_capture_combine_this_run = False
        self._update_postprocess_button()
        self._worker = None

    def _stop(self):
        if self._worker is not None:
            self._worker.quit()
        self._clear_grow_view_transition_state()
        self._finish_branch_grow_preview_if_needed()
        self.btn_stop.setEnabled(False)
        self.btn_grow_branches.setEnabled(True)
        self.progress_bar.hide()
        self.status_label.setText("Stopped by user")
        self._worker = None
        self._branch_step_target_layer = None

    def closeEvent(self, event) -> None:
        """Release viewer callbacks and session RAM when the dock widget closes."""
        try:
            self._camera_points_timer.stop()
            self._refresh_layers_timer.stop()
            self._autosave_timer.stop()
            self._ndisplay_sync_timer.stop()
            self._merge_postprocess_timer.stop()
        except Exception:
            pass
        if self._worker is not None:
            try:
                self._worker.quit()
            except Exception:
                pass
            self._worker = None
        try:
            self.viewer.layers.events.inserted.disconnect(self._on_layer_inserted)
        except Exception:
            pass
        try:
            self.viewer.layers.events.removed.disconnect(self._on_layer_removed)
        except Exception:
            pass
        try:
            self.viewer.dims.events.ndisplay.disconnect(self._on_ndisplay_changed)
        except Exception:
            pass
        try:
            self.viewer.dims.events.range.disconnect(
                self._on_dims_range_changed_for_pyramid_nav
            )
            self.viewer.dims.events.point.disconnect(
                self._on_dims_point_changed_for_pyramid_nav
            )
            self.viewer.dims.events.current_step.disconnect(
                self._on_dims_point_changed_for_pyramid_nav
            )
        except Exception:
            pass
        try:
            self.viewer.camera.events.zoom.disconnect(self._on_camera_for_branch_points)
            self.viewer.camera.events.center.disconnect(self._on_camera_for_branch_points)
        except Exception:
            pass
        if hasattr(self, "_branch_pts_sync") and self._branch_pts_sync:
            old_pts, old_cb = self._branch_pts_sync
            if old_pts in self.viewer.layers:
                try:
                    old_pts.events.data.disconnect(old_cb)
                except Exception:
                    pass
            self._branch_pts_sync = None
        self._remove_forced_2d_display_layer(restore_display=True)
        self._remove_forced_3d_display_layer(restore_display=True)
        self._reset_pyramid_dims_navigation()
        clear_image_level_cache()
        self._growth_capture_frames.clear()
        self._gif_capture_combined_frames.clear()
        self._gif_capture_pending_segment.clear()
        super().closeEvent(event)
