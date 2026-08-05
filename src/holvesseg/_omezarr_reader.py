"""OME-Zarr reader for napari (image pyramid only).

Saved segmentations under ``labels/`` are **not** opened automatically — use the
plugin's **Load saved segmentation from OME-Zarr…** button to import a mask into
the empty ``Segmentation`` layer on the current pyramid grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

LayerDataTuple = Tuple[Any, Dict[str, Any], str]


def resolve_omezarr_store_root(path: str | Path) -> Path:
    """If *path* points inside an ``.ome.zarr`` tree, return that store directory."""
    p = Path(path).expanduser()
    try:
        cur = p.resolve()
    except OSError:
        cur = p
    if cur.is_file():
        cur = cur.parent
    for _ in range(40):
        if cur.name.lower().endswith(".ome.zarr"):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    try:
        return p.resolve()
    except OSError:
        return p


def resolve_label_load_target(path: str | Path) -> Tuple[Path, Optional[str]]:
    """Map a file-dialog path to ``(store_root, label_group_name)``.

    - ``…/dataset.ome.zarr`` or ``…/dataset.ome.zarr/labels`` → ``(store, None)``
      (caller should show the label-group picker).
    - ``…/dataset.ome.zarr/labels/segmentation_v7`` → ``(store, "segmentation_v7")``
      (load that group directly, no picker).
    """
    p = Path(path).expanduser()
    try:
        selected = p.resolve()
    except OSError:
        selected = p
    store = resolve_omezarr_store_root(selected)
    try:
        store_resolved = store.resolve()
    except OSError:
        store_resolved = store
    try:
        rel = selected.relative_to(store_resolved)
    except ValueError:
        return store, None
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "labels":
        name = str(parts[1])
        if name and name not in {".", ".."}:
            return store, name
    return store, None


def _label_group_level_shapes(
    store: str | Path, name: str
) -> Dict[str, Tuple[int, ...]]:
    """Map pyramid keys → shapes for ``labels/<name>`` without decoding arrays."""
    import zarr

    from holvesseg._zarr_compat import zarr_array_keys

    path = resolve_omezarr_store_root(Path(store))
    try:
        root = zarr.open_group(str(path), mode="r")
        lg = root["labels"][name]
        keys = sorted(
            zarr_array_keys(lg),
            key=lambda k: int(k) if str(k).isdigit() else k,
        )
    except _ZARR_IO_ERRORS:
        return {}
    shapes: Dict[str, Tuple[int, ...]] = {}
    for key in keys:
        try:
            shapes[str(key)] = tuple(int(s) for s in lg[key].shape)
        except _ZARR_IO_ERRORS:
            continue
    return shapes


def _pick_label_array_key_for_shape(
    level_shapes: Dict[str, Tuple[int, ...]],
    target_shape: Tuple[int, int, int],
) -> Optional[str]:
    """Choose the pyramid level whose shape best matches *target_shape* for NN resample."""
    if not level_shapes:
        return None
    tgt = tuple(int(x) for x in target_shape)
    for key, shp in level_shapes.items():
        if tuple(int(x) for x in shp) == tgt:
            return key
    best_key: Optional[str] = None
    best_score = float("inf")
    for key, shp in level_shapes.items():
        if len(shp) != 3:
            continue
        # Prefer the level closest in log-space (minimizes resample distortion and I/O).
        score = sum(
            abs(float(np.log(max(int(s), 1))) - float(np.log(max(int(t), 1))))
            for s, t in zip(shp, tgt)
        )
        if score < best_score:
            best_score = score
            best_key = key
    return best_key or next(iter(level_shapes.keys()))


def _path_if_under_omezarr(uri: str) -> Optional[Path]:
    """Return the ``…/*.ome.zarr`` prefix if *uri* contains that segment."""
    if not uri or ".ome.zarr" not in uri.lower():
        return None
    parts = Path(uri).parts
    for i, part in enumerate(parts):
        if part.lower().endswith(".ome.zarr"):
            return Path(*parts[: i + 1])
    return None


# Exceptions that mean "store/group/array is missing or unreadable" rather than a
# bug we want to hide. Catching these specifically (instead of bare ``Exception``)
# keeps real programming errors visible.
_ZARR_IO_ERRORS = (KeyError, ValueError, OSError, FileNotFoundError)


def omzarr_store_channel_count(store: str | Path) -> Optional[int]:
    """Return the channel count declared in an OME-Zarr store, if any."""
    import zarr

    path = resolve_omezarr_store_root(Path(store))
    try:
        root = zarr.open_group(str(path), mode="r")
    except _ZARR_IO_ERRORS:
        return None

    def _from_block(block: Any) -> Optional[int]:
        if not isinstance(block, dict):
            return None
        chn = block.get("channel_names")
        if isinstance(chn, list) and len(chn) > 1:
            return len(chn)
        axes = block.get("axes")
        names: List[str] = []
        if isinstance(axes, list):
            for entry in axes:
                if isinstance(entry, dict):
                    names.append(str(entry.get("name", "")).lower())
                elif isinstance(entry, str):
                    names.append(entry.lower())
        if "c" not in names:
            return None
        datasets = block.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            return None
        d0 = datasets[0]
        if not isinstance(d0, dict):
            return None
        rel = d0.get("path")
        if not isinstance(rel, str) or not rel.strip():
            return None
        try:
            arr = root[rel.strip("/")]
            shp = tuple(int(x) for x in arr.shape)
        except _ZARR_IO_ERRORS:
            return None
        if len(names) != len(shp):
            return None
        return int(shp[names.index("c")])

    for key in ("multiscales",):
        blocks = root.attrs.get(key)
        if isinstance(blocks, list):
            for block in blocks:
                n = _from_block(block)
                if n is not None and n > 1:
                    return n
    ome = root.attrs.get("ome")
    if isinstance(ome, dict):
        ms = ome.get("multiscales")
        if isinstance(ms, list):
            for block in ms:
                n = _from_block(block)
                if n is not None and n > 1:
                    return n
    return None


def _array_has_foreground_chunked(a0: Any) -> bool:
    """True if any voxel of zarr array *a0* is > 0, scanning in bounded slabs.

    Reads one chunk-sized slab along the first axis at a time and early-exits on
    the first non-zero voxel, so this never materializes the whole label volume
    (a multi-GB label could otherwise be loaded just to answer a yes/no).
    """
    shape = tuple(int(s) for s in getattr(a0, "shape", ()) or ())
    if not shape:
        return False
    chunks = getattr(a0, "chunks", None)
    step = max(1, int(chunks[0])) if chunks else shape[0]
    step = max(1, min(step, shape[0]))
    for z0 in range(0, shape[0], step):
        z1 = min(shape[0], z0 + step)
        if np.any(np.asarray(a0[z0:z1]) > 0):
            return True
    return False


def _is_ephemeral_label_name(name: str) -> bool:
    """True for staging / rollback folders, not user-facing label groups."""
    n = str(name)
    return "__tmp_" in n or ".old_" in n


def _label_group_names_on_disk(store: str | Path) -> List[str]:
    """Subgroup names under ``labels/`` that look like readable NGFF label groups."""
    import zarr

    from holvesseg._zarr_compat import zarr_subgroup_keys

    path = resolve_omezarr_store_root(Path(store))
    try:
        root = zarr.open_group(str(path), mode="r")
        labels_grp = root.get("labels")
    except _ZARR_IO_ERRORS:
        return []
    if labels_grp is None:
        return []
    try:
        keys = zarr_subgroup_keys(labels_grp)
    except _ZARR_IO_ERRORS:
        return []
    out: List[str] = []
    for name in sorted(str(k) for k in keys):
        if _is_ephemeral_label_name(name):
            continue
        try:
            lg = labels_grp[name]
            if "0" in lg or lg.attrs.get("multiscales"):
                out.append(name)
        except _ZARR_IO_ERRORS:
            continue
    return out


def _ngff_label_names_from_store(store: Path) -> List[str]:
    """Label group names that exist on disk (attrs list may be stale after deletes)."""
    import zarr

    on_disk = _label_group_names_on_disk(store)
    if not on_disk:
        return []
    on_disk_set = set(on_disk)
    try:
        root = zarr.open_group(str(store), mode="r")
    except _ZARR_IO_ERRORS:
        return on_disk
    attrs = dict(root.attrs)
    lab = attrs.get("labels")
    if lab is None and isinstance(attrs.get("ome"), dict):
        lab = attrs["ome"].get("labels")
    if isinstance(lab, list):
        ordered: List[str] = []
        seen: set[str] = set()
        for raw in lab:
            name = str(raw)
            if _is_ephemeral_label_name(name) or name not in on_disk_set:
                continue
            if name not in seen:
                ordered.append(name)
                seen.add(name)
        for name in on_disk:
            if name not in seen:
                ordered.append(name)
        return ordered
    return on_disk


def _napari_sort_multiscale_list(
    data: List[Any], metadata: Optional[Dict[str, Any]] = None
) -> Tuple[List[Any], Dict[str, Any]]:
    """Order pyramid levels largest→smallest and drop duplicate shapes (napari rules).

    ``ome_zarr`` may return datasets in multiscales JSON order, which is not always
    strictly decreasing voxel count; napari's ``guess_multiscale`` rejects that.
    """
    if not isinstance(data, list) or len(data) <= 1:
        return data, dict(metadata or {})
    meta = dict(metadata or {})

    indexed = list(enumerate(data))
    indexed.sort(key=lambda iv: int(iv[1].size), reverse=True)
    order: List[int] = []
    seen_shapes: set[tuple[int, ...]] = set()
    new_data: List[Any] = []
    for i, arr in indexed:
        sh = tuple(int(x) for x in arr.shape)
        if sh in seen_shapes:
            continue
        seen_shapes.add(sh)
        order.append(i)
        new_data.append(arr)

    n0 = len(data)

    def _permute(key: str) -> None:
        val = meta.get(key)
        if isinstance(val, list) and len(val) == n0:
            meta[key] = [val[j] for j in order]

    _permute("coordinateTransformations")
    chn = meta.get("channel_names")
    if isinstance(chn, list) and len(chn) == n0:
        meta["channel_names"] = [chn[j] for j in order]
    vis = meta.get("visible")
    if isinstance(vis, list) and len(vis) == n0:
        meta["visible"] = [vis[j] for j in order]
    cl = meta.get("contrast_limits")
    if isinstance(cl, list) and len(cl) == n0:
        meta["contrast_limits"] = [cl[j] for j in order]
    cm = meta.get("colormap")
    if isinstance(cm, list) and len(cm) == n0:
        meta["colormap"] = [cm[j] for j in order]

    return new_data, meta


def _pyramid_data_ok(node: Any) -> bool:
    """True if *node* carries at least one array-like resolution for napari."""
    data = getattr(node, "data", None)
    if not isinstance(data, list) or len(data) == 0:
        return False
    for level in data:
        if not hasattr(level, "shape"):
            return False
        try:
            if int(level.shape[0]) < 1:  # type: ignore[arg-type]
                return False
        except Exception:
            return False
    return True


def _zarr_path_str(node: Any) -> str:
    z = getattr(node, "zarr", None)
    if z is None:
        return ""
    return str(getattr(z, "path", z)).replace("\\", "/")


def _pick_main_image_node(nodes: List[Any]) -> Optional[Any]:
    """Prefer the primary multiscale image, not label placeholders or linked refs."""
    best: Optional[tuple[int, int, Any]] = None
    for n in nodes:
        if not _pyramid_data_ok(n):
            continue
        zp = _zarr_path_str(n)
        if "/labels/" in zp:
            continue
        meta = getattr(n, "metadata", {}) or {}
        score = 0
        if meta.get("axes") is not None:
            score += 2
        if meta.get("coordinateTransformations") is not None:
            score += 1
        cand = (score, len(n.data), n)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is not None:
        return best[2]
    for n in nodes:
        if _pyramid_data_ok(n) and "/labels/" not in _zarr_path_str(n):
            return n
    return None


def _pick_label_pyramid_node(nodes: List[Any], chosen: str) -> Optional[Any]:
    """Pick the labels group node, not an empty prepended *image-label* parent."""
    needle = f"/labels/{chosen}".replace("\\", "/")
    scored: List[tuple[int, int, Any]] = []
    for n in nodes:
        if not _pyramid_data_ok(n):
            continue
        zp = _zarr_path_str(n)
        if needle not in zp and not zp.rstrip("/").endswith(f"/labels/{chosen}"):
            continue
        meta = getattr(n, "metadata", {}) or {}
        score = len(n.data)
        if meta.get("coordinateTransformations"):
            score += 1
        scored.append((score, len(zp), n))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][2]


def _pick_latest_segmentation_label(label_names: List[str]) -> Optional[str]:
    # Prefer "segmentation_vN" with highest N, else "segmentation", else last.
    best = None
    best_n = -1
    for n in label_names:
        if n == "segmentation":
            best = best or n
            continue
        if n.startswith("segmentation_v"):
            try:
                k = int(n.split("_v", 1)[1])
            except Exception:
                continue
            if k > best_n:
                best_n = k
                best = n
    if best is not None:
        return best
    return label_names[-1] if label_names else None


def labels_group_has_foreground(store: str | Path, name: str) -> bool:
    """True if ``labels/<name>/0`` contains any non-zero voxel."""
    import zarr

    path = resolve_omezarr_store_root(Path(store))
    try:
        root = zarr.open_group(str(path), mode="r")
        a0 = root["labels"][name]["0"]
    except _ZARR_IO_ERRORS:
        return False
    try:
        return _array_has_foreground_chunked(a0)
    except _ZARR_IO_ERRORS:
        return False


def label_group_level0_shape(store: str | Path, name: str) -> Optional[Tuple[int, ...]]:
    """Shape of ``labels/<name>/0`` if present."""
    import zarr

    path = resolve_omezarr_store_root(Path(store))
    try:
        root = zarr.open_group(str(path), mode="r")
        a0 = root["labels"][name]["0"]
        return tuple(int(x) for x in a0.shape)
    except _ZARR_IO_ERRORS:
        return None


def list_segmentation_label_groups(
    store: str | Path,
    *,
    check_foreground: bool = False,
) -> List[Dict[str, Any]]:
    """Summarize each NGFF label group under ``labels/`` (for load/save pickers).

    By default only reads names and level-0 shapes (fast).  Set
    ``check_foreground=True`` to scan arrays for non-zero voxels (slow on large
    finest-resolution saves).
    """
    import zarr

    path = resolve_omezarr_store_root(Path(store))
    names = _ngff_label_names_from_store(path)
    if not names:
        return []
    out: List[Dict[str, Any]] = []
    try:
        root = zarr.open_group(str(path), mode="r")
        labels_grp = root.get("labels")
    except _ZARR_IO_ERRORS:
        labels_grp = None
    for name in names:
        if _is_ephemeral_label_name(name):
            continue
        shp: Optional[Tuple[int, ...]] = None
        has_fg = False
        if labels_grp is None:
            continue
        try:
            lg = labels_grp[name]
            a0 = lg["0"]
            shp = tuple(int(x) for x in a0.shape)
            if check_foreground:
                has_fg = _array_has_foreground_chunked(a0)
        except _ZARR_IO_ERRORS:
            continue
        out.append(
            {
                "name": name,
                "shape": shp,
                "has_foreground": has_fg if check_foreground else None,
            }
        )
    return out


def format_label_group_choice(entry: Dict[str, Any]) -> str:
    """One-line label for UI lists."""
    name = str(entry.get("name", ""))
    shp = entry.get("shape")
    fg = entry.get("has_foreground")
    if isinstance(shp, tuple) and len(shp) == 3:
        shape_s = f"Z×Y×X={shp[0]}×{shp[1]}×{shp[2]}"
    else:
        shape_s = "shape unknown"
    if fg is None:
        return f"{name} ({shape_s})"
    empty_s = "" if fg else ", empty"
    return f"{name} ({shape_s}{empty_s})"


def _pick_saved_segmentation_label(
    label_names: List[str],
    store: str | Path,
    *,
    check_foreground: bool = True,
) -> Optional[str]:
    """Label group to load (non-empty autosave, else latest non-empty manual save).

    When ``check_foreground`` is False (picker UI), use name heuristics only so
    opening the dialog does not scan every label array on disk.
    """
    if not label_names:
        return None
    if not check_foreground:
        if "segmentation_autosave" in label_names:
            return "segmentation_autosave"
        seg_like = [
            n
            for n in label_names
            if n == "segmentation"
            or n.startswith("segmentation_v")
            or n == "segmentation_autosave"
        ]
        candidates = seg_like if seg_like else list(label_names)
        best = _pick_latest_segmentation_label(candidates)
        return best if best is not None else candidates[-1]

    path = resolve_omezarr_store_root(Path(store))
    if "segmentation_autosave" in label_names:
        if labels_group_has_foreground(path, "segmentation_autosave"):
            return "segmentation_autosave"
    seg_like = [
        n
        for n in label_names
        if n == "segmentation"
        or n.startswith("segmentation_v")
        or n == "segmentation_autosave"
    ]
    candidates = seg_like if seg_like else list(label_names)
    ordered: List[str] = []
    best = _pick_latest_segmentation_label(candidates)
    if best is not None:
        ordered.append(best)
    for n in reversed(candidates):
        if n not in ordered:
            ordered.append(n)
    for n in ordered:
        if labels_group_has_foreground(path, n):
            return n
    return ordered[0] if ordered else None


def materialize_saved_labels_at_shape(
    store: str | Path,
    target_shape: Tuple[int, ...],
    *,
    label_name: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, str]]:
    """Load saved NGFF labels and resample to *target_shape* (Z,Y,X) for napari."""
    from holvesseg._save_segmentation_zarr import read_zarr_labels_at_shape

    path = resolve_omezarr_store_root(Path(store))
    label_names = _ngff_label_names_from_store(path)
    if not label_names:
        import zarr

        try:
            root = zarr.open_group(str(path), mode="r")
            raw = root.attrs.get("labels", [])
            if isinstance(raw, list):
                label_names = [str(x) for x in raw]
        except _ZARR_IO_ERRORS:
            label_names = []
    chosen = label_name or _pick_saved_segmentation_label(label_names, path)
    if not chosen:
        return None

    import zarr

    from holvesseg._zarr_compat import zarr_array_keys

    try:
        root = zarr.open_group(str(path), mode="r")
        lg = root["labels"][chosen]
    except _ZARR_IO_ERRORS:
        return None

    tgt = tuple(int(x) for x in target_shape)
    if len(tgt) != 3:
        raise ValueError(f"target_shape must be 3-D (Z,Y,X); got {target_shape!r}")

    level_shapes = _label_group_level_shapes(path, chosen)
    if not level_shapes:
        try:
            keys = sorted(
                zarr_array_keys(lg),
                key=lambda k: int(k) if str(k).isdigit() else k,
            )
        except _ZARR_IO_ERRORS:
            keys = ["0"]
        for key in keys:
            try:
                level_shapes[str(key)] = tuple(int(s) for s in lg[key].shape)
            except _ZARR_IO_ERRORS:
                continue

    src_key = _pick_label_array_key_for_shape(level_shapes, tgt)
    if src_key is None:
        return None
    try:
        src_arr = lg[src_key]
    except _ZARR_IO_ERRORS:
        return None

    out_u8 = read_zarr_labels_at_shape(src_arr, tgt)
    return (out_u8.astype(np.int32, copy=False), chosen)


def load_saved_labels_layerdata(
    store: str | Path,
    *,
    label_name: Optional[str] = None,
) -> Optional[LayerDataTuple]:
    """Load one NGFF labels group as lazy napari layer data (multiscale when present).

    Returns a tuple ``(data, metadata_dict, "labels")`` suitable for
    ``viewer._add_layer_from_data`` / ``viewer.add_labels``.  *data* is a list of
    dask (or zarr-backed) arrays per pyramid level — the same representation the
    built-in OME-Zarr reader uses for images, so napari can swap levels on zoom
    instead of materializing the finest volume into RAM.
    """
    path = resolve_omezarr_store_root(store)
    from ome_zarr.io import ZarrLocation
    from ome_zarr.reader import Reader

    loc = ZarrLocation(str(path), mode="r")
    if label_name is not None:
        chosen = str(label_name)
    else:
        label_names = _ngff_label_names_from_store(path)
        if not label_names:
            raw = loc.root_attrs.get("labels", [])
            label_names = [str(x) for x in raw] if isinstance(raw, list) else []
        chosen = _pick_saved_segmentation_label(label_names, path)
    if not chosen:
        return None
    lab_loc = loc.create(f"labels/{chosen}")
    lab_nodes = list(Reader(lab_loc)())
    ln = _pick_label_pyramid_node(lab_nodes, chosen)
    if ln is None:
        return None
    ldata, lmeta = _napari_sort_multiscale_list(list(ln.data), dict(ln.metadata or {}))
    return (ldata, {"name": chosen, "metadata": lmeta}, "labels")


def labels_pyramid_level_count(layerdata: LayerDataTuple) -> int:
    """Number of pyramid levels in a labels ``LayerDataTuple`` from :func:`load_saved_labels_layerdata`."""
    data = layerdata[0]
    if isinstance(data, (list, tuple)):
        return len(data)
    return 1


def load_latest_saved_labels_layerdata(store: str | Path) -> Optional[LayerDataTuple]:
    """Return a single ``labels`` layer tuple for the newest ``labels/segmentation*`` group.

    Use this when the image was opened with another reader (e.g. napari-ome-zarr) and
    only the segmentation should be added.
    """
    return load_saved_labels_layerdata(store)


def read_omezarr_image(path: str | Path) -> List[LayerDataTuple]:
    """Return only the main image pyramid from an OME-Zarr store."""
    import warnings

    from holvesseg._image_validation import warn_if_multichannel_omezarr

    path = resolve_omezarr_store_root(Path(path))
    from ome_zarr.io import ZarrLocation
    from ome_zarr.reader import Reader

    loc = ZarrLocation(str(path), mode="r")
    nodes = list(Reader(loc)())
    out: List[LayerDataTuple] = []

    image_node = _pick_main_image_node(nodes)
    if image_node is not None:
        name = (image_node.metadata or {}).get("name") or path.name
        idata, imeta = _napari_sort_multiscale_list(
            list(image_node.data), dict(image_node.metadata or {})
        )
        finest_arr = idata[0] if isinstance(idata, list) and idata else idata
        finest_shape = tuple(int(x) for x in getattr(finest_arr, "shape", ()))
        warn_msg = warn_if_multichannel_omezarr(
            imeta, finest_shape, path_label=path.name
        )
        if warn_msg:
            warnings.warn(warn_msg, stacklevel=2)

        layer_kw: Dict[str, Any] = {"name": name, "metadata": imeta}
        from holvesseg._volume_utils import ngff_finest_voxel_spacing_zyx

        finest_scale = ngff_finest_voxel_spacing_zyx(imeta)
        if finest_scale is not None:
            layer_kw["scale"] = finest_scale
        out.append((idata, layer_kw, "image"))

    return out


def read_omezarr_with_segmentation(path: str | Path) -> List[LayerDataTuple]:
    """Backward-compatible alias — returns the image only (no ``labels`` layers)."""
    return read_omezarr_image(path)


def napari_get_reader(path: Any) -> Optional[Callable[[str], List[LayerDataTuple]]]:
    """npe2 reader hook for ``*.ome.zarr`` directories."""
    if isinstance(path, (list, tuple)):
        if not path:
            return None
        path = path[0]
    p = str(path)
    lower = p.lower()
    # Match only OME-Zarr-style directories so we do not compete with generic ``.zarr``
    # readers (and unrelated plugins) on arbitrary Zarr stores.
    if not lower.endswith(".ome.zarr"):
        return None
    # Only accept directories for this plugin reader.
    if not Path(p).is_dir():
        return None

    def _reader(_path: str) -> List[LayerDataTuple]:
        return read_omezarr_image(_path)

    return _reader

