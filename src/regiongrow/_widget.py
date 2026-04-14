"""Napari widget for interactive 3D vessel segmentation via region growing."""

import numpy as np
from qtpy.QtCore import Qt
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
    QTabWidget,
    QScrollArea,
)
from napari.qt.threading import thread_worker
import napari
from scipy.ndimage import zoom as ndimage_zoom, binary_dilation, binary_erosion, generate_binary_structure


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


class RegionGrowWidget(QWidget):
    """Widget for 3D vessel segmentation: plain region growing or
    morphological geodesic active contour (MGAC)."""

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer
        self._worker = None
        self._result_layer = None
        self._preprocessed_images = {}

        self._build_ui()
        self._connect_signals()
        self._refresh_layers()

        # Keep combos in sync with layer changes
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        root_layout = QVBoxLayout()
        self.setLayout(root_layout)

        # Use a scroll area so the dock widget stays vertically resizable
        # even when the parameter panel grows taller than the window.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root_layout.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)

        # --- Shared: Layer selection ---
        layer_group = QGroupBox("Layers")
        layer_form = QFormLayout()
        layer_group.setLayout(layer_form)

        self.image_combo = QComboBox()
        layer_form.addRow("Image:", self.image_combo)

        self.labels_combo = QComboBox()
        layer_form.addRow("Seed (Labels):", self.labels_combo)

        self.points_combo = QComboBox()
        layer_form.addRow("Start / End (Points):", self.points_combo)

        btn_row = QHBoxLayout()
        self.btn_create_seed = QPushButton("New Seed Layer")
        self.btn_create_points = QPushButton("New Points Layer")
        btn_row.addWidget(self.btn_create_seed)
        btn_row.addWidget(self.btn_create_points)
        layer_form.addRow(btn_row)
        layout.addWidget(layer_group)

        # --- Shared: Geometry (valid for both modes) ---
        geom_group = QGroupBox("Geometry")
        geom_form = QFormLayout()
        geom_group.setLayout(geom_form)

        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0, 200)
        self.margin_spin.setValue(5.0)
        self.margin_spin.setToolTip(
            "Extra margin (voxels) beyond start/end points along the\n"
            "vessel axis.  Applied in both Plain and Active Contour modes."
        )
        geom_form.addRow("Length margin:", _row(self.margin_spin))

        self.prep_downsample_spin = QSpinBox()
        self.prep_downsample_spin.setRange(1, 8)
        self.prep_downsample_spin.setValue(1)
        self.prep_downsample_spin.setToolTip(
            "Preprocess downsample factor (1 = no change).\n"
            "Creates a smaller working image for faster computation.\n"
            "Draw seed and points on the downsampled layer, then\n"
            "upsample the result.  Applies to both modes."
        )
        geom_form.addRow(
            "Preprocess downsample:", _row(self.prep_downsample_spin)
        )

        self.btn_preprocess = QPushButton("Create Downsampled Image")
        self.btn_preprocess.setToolTip(
            "Create a new image layer at the chosen lower resolution.\n"
            "Draw seeds and start/end points on that layer."
        )
        geom_form.addRow(_row(self.btn_preprocess))

        layout.addWidget(geom_group)

        # --- Shared: Visualization (valid for both modes) ---
        vis_group = QGroupBox("Visualization")
        vis_form = QFormLayout()
        vis_group.setLayout(vis_form)

        self.animate_check = QCheckBox("Animate growth")
        self.animate_check.setChecked(True)
        self.animate_check.setToolTip(
            "Show the growing contour at each display step.\n"
            "Disable for maximum speed.\n"
            "The update frequency is configured per mode in the tab below."
        )
        vis_form.addRow(_row(self.animate_check))

        layout.addWidget(vis_group)

        # --- Mode tabs ---
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_plain_tab(), "Plain Region Growing")
        self.tabs.addTab(self._build_ac_tab(), "3D Active Contour")
        layout.addWidget(self.tabs)

        # --- Shared: Run / Stop / Reset ---
        ctrl = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_run.setStyleSheet("font-weight: bold;")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_reset = QPushButton("Reset")
        ctrl.addWidget(self.btn_run)
        ctrl.addWidget(self.btn_stop)
        ctrl.addWidget(self.btn_reset)
        layout.addLayout(ctrl)

        # --- Shared: Post-Processing (after first result review) ---
        post_group = QGroupBox("Post-Processing (Optional)")
        post_form = QFormLayout()
        post_group.setLayout(post_form)

        self.btn_postprocess = QPushButton("Upsample Result to Original Size")
        self.btn_postprocess.setToolTip(
            "Enabled after segmentation when downsample factor > 1.\n"
            "Creates a full-resolution mask from the downsampled result."
        )
        self.btn_postprocess.setEnabled(False)
        post_form.addRow(_row(self.btn_postprocess))

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
            "Apply the selected operation to the current segmentation result\n"
            "after visual inspection. Creates a new refined result layer."
        )
        post_form.addRow(_row(self.btn_apply_morph))

        layout.addWidget(post_group)

        # --- Shared: Status ---
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        layout.addStretch()

    def _build_plain_tab(self):
        """Build and return the Plain Region Growing parameter panel."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Smoothing
        smooth_group = QGroupBox("Smoothing")
        smooth_form = QFormLayout()
        smooth_group.setLayout(smooth_form)
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 20.0)
        self.sigma_spin.setValue(2.0)
        self.sigma_spin.setSingleStep(0.5)
        self.sigma_spin.setToolTip(
            "Gaussian smoothing sigma for gradient computation.\n"
            "Increase to ignore small surface wrinkles;\n"
            "decrease for finer boundary detail."
        )
        smooth_form.addRow("Smoothing \u03c3:", _row(self.sigma_spin))
        layout.addWidget(smooth_group)

        # Stopping criteria
        stop_group = QGroupBox("Stopping Criteria")
        stop_form = QFormLayout()
        stop_group.setLayout(stop_form)

        self.flux_spin = QDoubleSpinBox()
        self.flux_spin.setRange(0.0, 50.0)
        self.flux_spin.setValue(15.0)
        self.flux_spin.setSingleStep(1.0)
        self.flux_spin.setDecimals(1)
        self.flux_spin.setToolTip(
            "Gradient-flux penalty weight.\n"
            "When the image gradient points outward (wall \u2192 background),\n"
            "the traversal cost is multiplied by 1 + w\u00b7cos\u00b2(\u03b8).\n"
            "Higher \u2192 stronger boundary penalty.\n"
            "Lower  \u2192 more permissive (may leak)."
        )
        stop_form.addRow("Flux penalty:", _row(self.flux_spin))

        self.intensity_tol_spin = QDoubleSpinBox()
        self.intensity_tol_spin.setRange(0.5, 10.0)
        self.intensity_tol_spin.setValue(3.0)
        self.intensity_tol_spin.setSingleStep(0.5)
        self.intensity_tol_spin.setToolTip(
            "Intensity gate: reject any candidate whose voxel intensity\n"
            "is more than N standard deviations below the running\n"
            "region mean (Welford online statistics)."
        )
        stop_form.addRow(
            "Intensity tolerance:", _row(self.intensity_tol_spin)
        )

        self.cost_budget_spin = QDoubleSpinBox()
        self.cost_budget_spin.setRange(0.0, 100000.0)
        self.cost_budget_spin.setValue(0.0)
        self.cost_budget_spin.setDecimals(1)
        self.cost_budget_spin.setToolTip(
            "Maximum accumulated geodesic cost a voxel may have to be\n"
            "accepted.  0 = auto-calibrate from image statistics.\n"
            "Increase if the region stops too early."
        )
        stop_form.addRow("Cost budget (0=auto):", _row(self.cost_budget_spin))

        # Optional upper-threshold hard stop
        self.upper_thr_check = QCheckBox("Enable upper threshold")
        self.upper_thr_check.setChecked(False)
        self.upper_thr_check.setToolTip(
            "Hard stop: voxels with intensity above the computed threshold\n"
            "are unconditionally rejected, regardless of other criteria.\n"
            "Useful to prevent leaking into hyper-bright structures."
        )
        stop_form.addRow(_row(self.upper_thr_check))

        self.upper_thr_combo = QComboBox()
        self.upper_thr_combo.addItems([
            "Otsu",
            "Triangle",
            "Li",
            "90th percentile",
            "95th percentile",
        ])
        self.upper_thr_combo.setEnabled(False)
        self.upper_thr_combo.setToolTip(
            "Algorithm used to auto-compute the upper intensity threshold\n"
            "from the whole image before growing starts."
        )
        stop_form.addRow("Threshold method:", _row(self.upper_thr_combo))

        layout.addWidget(stop_group)

        # Display frequency  (unit = accepted voxels, plain-mode specific)
        disp_group = QGroupBox("Display Frequency")
        disp_form = QFormLayout()
        disp_group.setLayout(disp_form)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 10000)
        self.step_spin.setValue(500)
        self.step_spin.setToolTip(
            "Refresh the napari display every N accepted voxels.\n"
            "Increase for faster (but coarser) animation.\n"
            "Only active when \u2018Animate growth\u2019 is checked above."
        )
        disp_form.addRow("Every N voxels:", _row(self.step_spin))

        layout.addWidget(disp_group)
        layout.addStretch()
        return tab

    def _build_ac_tab(self):
        """Build and return the 3D Active Contour parameter panel."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Initialization
        init_group = QGroupBox("Initialization")
        init_form = QFormLayout()
        init_group.setLayout(init_form)

        self.ac_radius_spin = QDoubleSpinBox()
        self.ac_radius_spin.setRange(1.0, 500.0)
        self.ac_radius_spin.setValue(10.0)
        self.ac_radius_spin.setSingleStep(1.0)
        self.ac_radius_spin.setToolTip(
            "Radius (voxels) of the initial cylindrical tube seeded from centerline.\n\n"
            "Initial tube should slightly overlap the vessel boundary:\n"
            "  • Too small: contour must grow far, may miss fine features\n"
            "  • Too large: contour must shrink far, may include background\n\n"
            "Tip: Start with ~50% of vessel diameter estimate, adjust if needed."
        )
        init_form.addRow("Initial radius (vox):", _row(self.ac_radius_spin))
        layout.addWidget(init_group)

        # Algorithm parameters
        algo_group = QGroupBox("Algorithm Parameters")
        algo_form = QFormLayout()
        algo_group.setLayout(algo_form)

        self.ac_sigma_spin = QDoubleSpinBox()
        self.ac_sigma_spin.setRange(0.1, 20.0)
        self.ac_sigma_spin.setValue(2.0)
        self.ac_sigma_spin.setSingleStep(0.5)
        self.ac_sigma_spin.setToolTip(
            "Gaussian sigma for computing the inverse-Gaussian-gradient edge image.\n\n"
            "Controls edge sensitivity:\n"
            "  • Lower σ (0.5–1.0): sensitive to fine details, noisy at edges\n"
            "  • Medium σ (1.5–3.0): typical range for most vessels\n"
            "  • Higher σ (3.0–10.0): robust to texture, may miss thin vessels\n\n"
            "Choose based on noise level and boundary definition of your image."
        )
        algo_form.addRow("Smoothing \u03c3:", _row(self.ac_sigma_spin))

        self.ac_balloon_spin = QDoubleSpinBox()
        self.ac_balloon_spin.setRange(-5.0, 5.0)
        self.ac_balloon_spin.setValue(0.5)
        self.ac_balloon_spin.setSingleStep(0.1)
        self.ac_balloon_spin.setDecimals(2)
        self.ac_balloon_spin.setToolTip(
            "Balloon (inflation) coefficient that drives contour evolution.\n\n"
            "Controls how strongly the contour expands through the image:\n"
            "  • Positive values: inflates the contour, pushing it outward.\n"
            "  • Negative values: deflates (rarely used).\n\n"
            "Recommended ranges depend on vessel characteristics:\n"
            "  • Weak edges or smooth interior: 0.5–1.0 (strong inflation)\n"
            "  • Strong edges or thin vessels: 0.1–0.3 (gentle inflation)\n"
            "  • High SNR, well-defined boundaries: 0.1–0.5\n\n"
            "Too high → contour expands too far; too low → slow growth."
        )
        algo_form.addRow("Balloon:", _row(self.ac_balloon_spin))

        self.ac_smoothing_spin = QSpinBox()
        self.ac_smoothing_spin.setRange(0, 10)
        self.ac_smoothing_spin.setValue(1)
        self.ac_smoothing_spin.setToolTip(
            "Morphological smoothing iterations applied after each evolution step.\n\n"
            "Controls surface smoothness:\n"
            "  • 0: no smoothing, follows all contours\n"
            "  • 1–2: moderate smoothing (recommended start)\n"
            "  • 3+: aggressive smoothing, removes fine features\n\n"
            "Increase for cleaner, smoother segmentation if result is too bumpy."
        )
        algo_form.addRow("Smoothing steps:", _row(self.ac_smoothing_spin))

        self.ac_total_iter_spin = QSpinBox()
        self.ac_total_iter_spin.setRange(1, 5000)
        self.ac_total_iter_spin.setValue(200)
        self.ac_total_iter_spin.setToolTip(
            "Total morphological evolution iterations to run.\n\n"
            "Controls how many steps the contour evolves:\n"
            "  • Typical range: 100–500 iterations\n"
            "  • Larger values allow growth in complex shapes but may leak\n"
            "  • Start with 200, adjust if segmentation reaches boundary well"
        )
        algo_form.addRow("Total iterations:", _row(self.ac_total_iter_spin))

        self.ac_yield_spin = QSpinBox()
        self.ac_yield_spin.setRange(1, 500)
        self.ac_yield_spin.setValue(5)
        self.ac_yield_spin.setToolTip(
            "Refresh the napari display every N evolution iterations.\n"
            "Increase for faster (but coarser) animation.\n"
            "Only active when \u2018Animate growth\u2019 is checked above."
        )
        algo_form.addRow("Every N iterations:", _row(self.ac_yield_spin))

        layout.addWidget(algo_group)

        layout.addStretch()
        return tab

    # ------------------------------------------------------------- signals --
    def _connect_signals(self):
        self.btn_create_seed.clicked.connect(self._create_seed_layer)
        self.btn_create_points.clicked.connect(self._create_points_layer)
        self.btn_preprocess.clicked.connect(self._create_downsampled_image)
        self.btn_postprocess.clicked.connect(self._upsample_result_to_original)
        self.btn_apply_morph.clicked.connect(self._apply_morphological_operation)
        self.btn_run.clicked.connect(self._run)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_reset.clicked.connect(self._reset)
        self.image_combo.currentTextChanged.connect(
            self._update_postprocess_button
        )
        # Enable/disable threshold method combo with checkbox
        self.upper_thr_check.toggled.connect(self.upper_thr_combo.setEnabled)

    # -------------------------------------------------------- layer helpers --
    def _refresh_layers(self, event=None):
        for combo, layer_type in [
            (self.image_combo, napari.layers.Image),
            (self.labels_combo, napari.layers.Labels),
            (self.points_combo, napari.layers.Points),
        ]:
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            for layer in self.viewer.layers:
                if isinstance(layer, layer_type):
                    combo.addItem(layer.name)
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        self._update_postprocess_button()

    def _update_postprocess_button(self):
        name = self.image_combo.currentText()
        info = self._preprocessed_images.get(name)
        enabled = info is not None and int(info.get("factor", 1)) > 1
        self.btn_postprocess.setEnabled(enabled)

    def _create_downsampled_image(self):
        name = self.image_combo.currentText()
        if not name:
            self.status_label.setText("Select an image layer first.")
            return

        layer = self.viewer.layers[name]
        image_data = np.asarray(layer.data, dtype=np.float64)
        if image_data.ndim != 3:
            self.status_label.setText("Image must be 3-D.")
            return

        factor = int(self.prep_downsample_spin.value())
        if factor <= 1:
            self.status_label.setText("Downsample factor is 1. Use original image directly.")
            self._update_postprocess_button()
            return

        zoom_factor = 1.0 / factor
        ds_image = ndimage_zoom(image_data, zoom_factor, order=1)
        ds_name = f"{name} (ds x{factor})"

        if ds_name in self.viewer.layers:
            self.viewer.layers.remove(ds_name)

        self.viewer.add_image(ds_image, name=ds_name)
        self._preprocessed_images[ds_name] = {
            "original_name": name,
            "original_shape": tuple(image_data.shape),
            "factor": factor,
        }

        self._refresh_layers()
        self.image_combo.setCurrentText(ds_name)
        self.status_label.setText(
            f"Created downsampled image: {ds_name}. Draw seed and points on this layer."
        )

    def _upsample_result_to_original(self):
        if self._result_layer is None or self._result_layer not in self.viewer.layers:
            self.status_label.setText("Run segmentation first.")
            return

        image_name = self.image_combo.currentText()
        info = self._preprocessed_images.get(image_name)
        if info is None:
            self.status_label.setText("Current image is not a preprocessed layer.")
            self._update_postprocess_button()
            return

        factor = int(info.get("factor", 1))
        if factor <= 1:
            self.status_label.setText("Downsample factor is 1. Upsampling is not needed.")
            self._update_postprocess_button()
            return

        mask = np.asarray(self._result_layer.data) > 0
        target_shape = tuple(info["original_shape"])
        zoom = [o / s for o, s in zip(target_shape, mask.shape)]
        upsampled = ndimage_zoom(mask.astype(np.float64), zoom, order=0) > 0.5

        result_name = "Segmentation Result (Original Size)"
        if result_name in self.viewer.layers:
            self.viewer.layers[result_name].data = upsampled.astype(np.int32)
        else:
            self.viewer.add_labels(
                upsampled.astype(np.int32),
                name=result_name,
                opacity=0.5,
            )

        self.status_label.setText("Postprocessing complete: upsampled result created.")

    def _apply_morphological_operation(self):
        """Apply morphological dilation or erosion to the result layer."""
        if self._result_layer is None or self._result_layer not in self.viewer.layers:
            self.status_label.setText("Run segmentation first.")
            return

        operation = self.morph_op_combo.currentText()
        if operation == "None":
            self.status_label.setText("Select an operation (Dilation or Erosion).")
            return

        radius = self.morph_radius_spin.value()
        mask = np.asarray(self._result_layer.data) > 0

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
            self.viewer.layers[result_name].data = result.astype(np.int32)
        else:
            self.viewer.add_labels(
                result.astype(np.int32),
                name=result_name,
                opacity=0.5,
            )

        self.status_label.setText(
            f"Postprocessing complete: {op_name} (radius={radius}) applied."
        )

    def _create_seed_layer(self):
        name = self.image_combo.currentText()
        if not name:
            self.status_label.setText("Select an image layer first.")
            return
        shape = self.viewer.layers[name].data.shape
        lbl = self.viewer.add_labels(
            np.zeros(shape, dtype=np.int32), name="Vessel Seed"
        )
        lbl.mode = "paint"
        lbl.brush_size = 3
        self._refresh_layers()
        self.labels_combo.setCurrentText("Vessel Seed")

    def _create_points_layer(self):
        pts = self.viewer.add_points(
            np.empty((0, 3)),
            ndim=3,
            name="Start/End Points",
            size=5,
            face_color="magenta",
        )
        pts.mode = "add"
        self._refresh_layers()
        self.points_combo.setCurrentText("Start/End Points")

    def _get_layers(self):
        """Return (image_layer, labels_layer, points_layer) or None."""
        names = (
            self.image_combo.currentText(),
            self.labels_combo.currentText(),
            self.points_combo.currentText(),
        )
        if not all(names):
            self.status_label.setText("Select all required layers.")
            return None
        try:
            return tuple(self.viewer.layers[n] for n in names)
        except KeyError as exc:
            self.status_label.setText(f"Layer not found: {exc}")
            return None

    # ----------------------------------------------------------- execution --
    def _run(self):
        layers = self._get_layers()
        if layers is None:
            return
        image_layer, labels_layer, points_layer = layers

        image_data = np.asarray(image_layer.data, dtype=np.float64)
        if image_data.ndim != 3:
            self.status_label.setText("Image must be 3-D.")
            return

        seed_mask = np.asarray(labels_layer.data) > 0
        if not seed_mask.any():
            self.status_label.setText(
                "Draw a seed region first (use the brush)."
            )
            return

        points = np.asarray(points_layer.data)
        if len(points) < 2:
            self.status_label.setText(
                "Mark at least 2 points (start & end)."
            )
            return
        start_point = points[0]
        end_point = points[-1]

        # Prepare result layer
        if (
            self._result_layer is None
            or self._result_layer not in self.viewer.layers
        ):
            self._result_layer = self.viewer.add_labels(
                np.zeros(image_data.shape, dtype=np.int32),
                name="Segmentation Result",
                opacity=0.5,
            )

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.show()

        if self.tabs.currentIndex() == 0:
            self._run_plain(image_data, seed_mask, start_point, end_point)
        else:
            self._run_ac(image_data, seed_mask, start_point, end_point)

    def _run_plain(self, image_data, seed_mask, start_point, end_point):
        """Start a plain region-growing worker."""
        self.status_label.setText(
            "Computing gradient (may take a moment)..."
        )

        upper_thr = None
        if self.upper_thr_check.isChecked():
            from ._algorithm import compute_upper_threshold
            _method_map = {
                "Otsu": "otsu",
                "Triangle": "triangle",
                "Li": "li",
                "90th percentile": "p90",
                "95th percentile": "p95",
            }
            method = _method_map[self.upper_thr_combo.currentText()]
            upper_thr = compute_upper_threshold(image_data, method)
            self.status_label.setText(
                f"Upper threshold ({self.upper_thr_combo.currentText()})"
                f": {upper_thr:.2f}  — growing..."
            )

        cost_val = self.cost_budget_spin.value()
        params = dict(
            sigma=self.sigma_spin.value(),
            cost_budget=cost_val if cost_val > 0 else None,
            flux_weight=self.flux_spin.value(),
            intensity_tolerance=self.intensity_tol_spin.value(),
            margin=self.margin_spin.value(),
            upper_threshold=upper_thr,
        )
        animate = self.animate_check.isChecked()
        yield_every = self.step_spin.value() if animate else 10**9

        from ._algorithm import region_grow
        _image = image_data
        _seed = seed_mask
        _start = start_point
        _end = end_point
        _yield_every = yield_every

        @thread_worker
        def _work():
            for result in region_grow(
                _image, _seed, _start, _end,
                yield_every=_yield_every, **params
            ):
                yield result

        worker = _work()
        worker.yielded.connect(self._on_step)
        worker.finished.connect(self._on_finished)
        worker.start()
        self._worker = worker

    def _run_ac(self, image_data, seed_mask, start_point, end_point):
        """Start a 3-D morphological geodesic active contour worker."""
        self.status_label.setText("Initializing active contour tube...")

        _ac_total = self.ac_total_iter_spin.value()
        _ac_yield = (
            self.ac_yield_spin.value()
            if self.animate_check.isChecked()
            else _ac_total
        )
        params = dict(
            radius=self.ac_radius_spin.value(),
            sigma=self.ac_sigma_spin.value(),
            balloon=self.ac_balloon_spin.value(),
            smoothing=self.ac_smoothing_spin.value(),
            total_iter=_ac_total,
            yield_every=_ac_yield,
            margin=self.margin_spin.value(),
        )

        from ._active_contour import active_contour_grow
        _image = image_data
        _seed = seed_mask
        _start = start_point
        _end = end_point

        @thread_worker
        def _work():
            for result in active_contour_grow(
                _image, _seed, _start, _end, **params
            ):
                yield result

        worker = _work()
        worker.yielded.connect(self._on_step)
        worker.finished.connect(self._on_finished)
        worker.start()
        self._worker = worker

    def _on_step(self, result):
        iteration, mask = result
        self._result_layer.data = mask.astype(np.int32)
        n_voxels = int(mask.sum())
        self.status_label.setText(
            f"Step {iteration} — {n_voxels:,} voxels segmented"
        )

    def _on_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.hide()
        if self.status_label.text().startswith("Step"):
            self.status_label.setText(self.status_label.text() + "  ✓ Done")
        else:
            self.status_label.setText("Done")
        self._update_postprocess_button()
        self._worker = None

    def _stop(self):
        if self._worker is not None:
            self._worker.quit()
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.hide()
        self.status_label.setText("Stopped by user")
        self._worker = None

    def _reset(self):
        if self._result_layer is not None and self._result_layer in self.viewer.layers:
            self._result_layer.data = np.zeros_like(self._result_layer.data)
        self.status_label.setText("Ready")
