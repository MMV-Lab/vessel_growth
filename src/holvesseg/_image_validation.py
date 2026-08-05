"""Validate napari Image layers for hollow-vessel segmentation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, List, Optional, Sequence, Tuple

from ._volume_utils import image_level_shape

_CHANNEL_LAYER_RE = re.compile(r"^C:\d+(\[\d+\])?$")
_SPATIAL_AXIS_LETTERS = frozenset("zyx")


@dataclass(frozen=True)
class ImageValidationIssue:
    code: str
    message: str
    blocking: bool = True


def looks_like_channel_layer_name(name: str) -> bool:
    """True for napari-ome-zarr style per-channel layer names (``C:0``, ``C:1[0]``, …)."""
    return bool(_CHANNEL_LAYER_RE.match(str(name).strip()))


def _axis_names_from_metadata(metadata: dict) -> List[str]:
    axes = metadata.get("axes")
    names: List[str] = []
    if isinstance(axes, list):
        for entry in axes:
            if isinstance(entry, dict):
                names.append(str(entry.get("name", "")).lower())
            elif isinstance(entry, str):
                names.append(entry.lower())
    elif isinstance(axes, str):
        names = list(axes.lower())
    ome_axes = metadata.get("ome_axes_original")
    if isinstance(ome_axes, str) and not names:
        names = list(ome_axes.lower())
    return names


def ngff_axis_size(
    metadata: dict, shape: Tuple[int, ...], axis_letter: str
) -> Optional[int]:
    """Size of *axis_letter* (``c``, ``t``, …) from NGFF / OME metadata and *shape*."""
    letter = axis_letter.lower()
    names = _axis_names_from_metadata(metadata)
    if letter in names and len(names) == len(shape):
        return int(shape[names.index(letter)])
    return None


def ngff_channel_count(metadata: dict, shape: Tuple[int, ...]) -> Optional[int]:
    """Number of channels described by metadata, if known."""
    n = ngff_axis_size(metadata, shape, "c")
    if n is not None:
        return n
    chn = metadata.get("channel_names")
    if isinstance(chn, list) and chn:
        return len(chn)
    return None


def count_channel_split_image_layers(layers: Sequence[Any]) -> int:
    """Count image layers opened as separate channels (``C:0``, ``C:1``, …)."""
    import napari.layers

    n = 0
    for lyr in layers:
        if isinstance(lyr, napari.layers.Image) and looks_like_channel_layer_name(
            str(getattr(lyr, "name", ""))
        ):
            n += 1
    return n


def validate_image_layer(
    layer: Any,
    *,
    pyramid_level: int = 0,
    channel_split_layer_count: int = 0,
    omzarr_store_channels: Optional[int] = None,
) -> List[ImageValidationIssue]:
    """Return user-facing issues for the selected image layer."""
    import napari.layers

    if layer is None:
        return [
            ImageValidationIssue(
                "no_layer", "Select an image layer in the Image dropdown.", blocking=True
            )
        ]

    if not isinstance(layer, napari.layers.Image):
        return [
            ImageValidationIssue(
                "not_image",
                "The selected layer is not an Image layer.",
                blocking=True,
            )
        ]

    issues: List[ImageValidationIssue] = []

    if bool(getattr(layer, "rgb", False)):
        issues.append(
            ImageValidationIssue(
                "rgb",
                "RGB colour images are not supported. Use a single-channel "
                "(grayscale) 3D volume with shape Z×Y×X.",
                blocking=True,
            )
        )

    try:
        shape = tuple(int(x) for x in image_level_shape(layer, int(pyramid_level)))
    except (TypeError, ValueError, IndexError):
        shape = tuple()

    if not shape:
        issues.append(
            ImageValidationIssue(
                "no_shape",
                "Could not read the image shape at the selected pyramid level.",
                blocking=True,
            )
        )
        return issues

    if any(int(s) <= 0 for s in shape):
        issues.append(
            ImageValidationIssue(
                "empty",
                f"Image has an invalid shape {shape}.",
                blocking=True,
            )
        )
        return issues

    md = dict(getattr(layer, "metadata", {}) or {})

    n_ch = ngff_channel_count(md, shape)
    if n_ch is not None and n_ch > 1:
        issues.append(
            ImageValidationIssue(
                "multi_channel_metadata",
                f"This volume has {n_ch} channels in its file metadata. "
                "Segmentation expects a single-channel 3D stack (Z×Y×X). "
                "Preprocess to one channel (see README) before using Compute Branch.",
                blocking=True,
            )
        )

    if int(channel_split_layer_count) > 1:
        issues.append(
            ImageValidationIssue(
                "multi_channel_layers",
                f"This OME-Zarr was opened as {channel_split_layer_count} separate channel "
                "layers (C:0, C:1, …). Pick one channel in the Image dropdown, or "
                "re-export a single-channel Zarr before segmenting.",
                blocking=True,
            )
        )
    elif omzarr_store_channels is not None and int(omzarr_store_channels) > 1:
        if len(shape) == 3:
            issues.append(
                ImageValidationIssue(
                    "multi_channel_store",
                    f"The OME-Zarr store lists {int(omzarr_store_channels)} channels. "
                    "Confirm the selected image is the channel you want, or preprocess "
                    "to a single-channel Z×Y×X volume (see README).",
                    blocking=False,
                )
            )
        else:
            issues.append(
                ImageValidationIssue(
                    "multi_channel_store",
                    f"The OME-Zarr store lists {int(omzarr_store_channels)} channels. "
                    "Preprocess to one channel (Z×Y×X) before segmenting.",
                    blocking=True,
                )
            )

    n_time = ngff_axis_size(md, shape, "t")
    if n_time is not None and n_time > 1:
        issues.append(
            ImageValidationIssue(
                "multi_time",
                f"This volume has {n_time} time points. Use a single time point "
                "(shape Z×Y×X) before segmenting.",
                blocking=True,
            )
        )

    if len(shape) == 2:
        issues.append(
            ImageValidationIssue(
                "2d",
                f"Image is 2D (shape {shape}). This plugin expects a 3D volume (Z×Y×X).",
                blocking=True,
            )
        )
    elif len(shape) == 3:
        axis_names = _axis_names_from_metadata(md)
        if len(axis_names) == 3:
            spatial = sum(1 for a in axis_names if a in _SPATIAL_AXIS_LETTERS)
            if spatial < 3 and "c" not in axis_names and "t" not in axis_names:
                issues.append(
                    ImageValidationIssue(
                        "axes_mismatch",
                        f"Image axes {axis_names!r} do not look like Z×Y×X. "
                        "Check that the volume is a single-channel 3D stack.",
                        blocking=True,
                    )
                )
    elif len(shape) > 3:
        issues.append(
            ImageValidationIssue(
                "extra_dims",
                f"Image shape {shape} has {len(shape)} dimensions; expected single-channel "
                "3D (Z×Y×X). If this is multi-channel or time-series data, preprocess "
                "to one channel and one time point (see README).",
                blocking=True,
            )
        )

    ndim = getattr(layer, "ndim", None)
    if ndim is not None and int(ndim) != 3 and len(shape) == 3:
        issues.append(
            ImageValidationIssue(
                "ndim_mismatch",
                f"Image layer reports ndim={int(ndim)} but the working grid is 3D. "
                "Extra dimensions (channel/time) may still be present — verify the data.",
                blocking=False,
            )
        )

    return issues


def summarize_validation_issues(issues: Sequence[ImageValidationIssue]) -> str:
    if not issues:
        return ""
    return " ".join(i.message for i in issues)


def has_blocking_validation_issues(issues: Sequence[ImageValidationIssue]) -> bool:
    return any(i.blocking for i in issues)


def warn_if_multichannel_omezarr(
    metadata: dict,
    finest_shape: Tuple[int, ...],
    *,
    path_label: str = "OME-Zarr",
) -> Optional[str]:
    """Return a warning message when an OME-Zarr image is not single-channel 3D."""
    shp = tuple(int(x) for x in finest_shape)
    n_ch = ngff_channel_count(metadata, shp)
    if n_ch is not None and n_ch > 1:
        return (
            f"{path_label}: {n_ch} channels detected in metadata; this plugin expects "
            "a single-channel Z×Y×X volume. Preprocess to one channel before segmenting."
        )
    n_time = ngff_axis_size(metadata, shp, "t")
    if n_time is not None and n_time > 1:
        return (
            f"{path_label}: {n_time} time points detected; use a single time point "
            "(Z×Y×X) before segmenting."
        )
    if len(shp) > 3:
        return (
            f"{path_label}: finest level shape {shp} is not 3D. Expected single-channel "
            "Z×Y×X — preprocess multi-channel or time-series stacks first."
        )
    if len(shp) == 2:
        return (
            f"{path_label}: finest level is 2D (shape {shp}). "
            "This plugin expects a 3D volume (Z×Y×X)."
        )
    return None
