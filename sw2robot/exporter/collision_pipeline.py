"""Shared collision-geometry generation for export backends."""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import xml.etree.ElementTree as ET

# CoACD parameters per quality preset.
_COACD_PRESETS = {
    "balanced": {"threshold": 0.2, "max_convex_hull": 6,
                 "preprocess_resolution": 30, "mcts_iterations": 30},
    "fine": {"threshold": 0.1, "max_convex_hull": 8,
             "preprocess_resolution": 40, "mcts_iterations": 60},
}
_COACD_CACHE_VERSION = 1

_PRIMITIVE_MODES = {"primitive": None, "box": "box",
                    "cylinder": "cylinder", "sphere": "sphere"}
COLLISION_MODES = ("copy", "hull", "coacd", *_PRIMITIVE_MODES)
PREVIEW_COLLISION_MODES = ("hull", "coacd", *_PRIMITIVE_MODES)

_coacd_cache_locks = {}
_coacd_cache_locks_guard = threading.Lock()


def _coacd_key_lock(key):
    with _coacd_cache_locks_guard:
        lk = _coacd_cache_locks.get(key)
        if lk is None:
            lk = _coacd_cache_locks[key] = threading.Lock()
        return lk


def coacd_available():
    import importlib.util
    return importlib.util.find_spec("coacd") is not None


def run_coacd(vertices, faces, params):
    import coacd

    coacd.set_log_level("error")
    mesh = coacd.Mesh(vertices, faces)
    return coacd.run_coacd(mesh, merge=True, **params)


def coacd_part_stls(src, quality, cache_dir, *, load_mesh_metres, run_coacd_fn=None):
    import numpy as np
    import trimesh

    params = _COACD_PRESETS[quality]
    with open(src, "rb") as f:
        src_bytes = f.read()
    h = hashlib.sha1(src_bytes)
    h.update(json.dumps([quality, params, _COACD_CACHE_VERSION],
                        sort_keys=True).encode())
    key = h.hexdigest()[:16]
    part_dir = os.path.join(cache_dir, key)
    manifest = os.path.join(part_dir, "parts.json")

    def _read_cache():
        try:
            with open(manifest) as f:
                names = json.load(f)
            return [open(os.path.join(part_dir, n), "rb").read() for n in names]
        except (OSError, ValueError):
            return None

    cached = _read_cache()
    if cached is not None:
        return cached

    runner = run_coacd if run_coacd_fn is None else run_coacd_fn
    with _coacd_key_lock(key):
        cached = _read_cache()
        if cached is not None:
            return cached
        mesh = load_mesh_metres(src)
        parts = runner(mesh.vertices, mesh.faces, params)
        os.makedirs(part_dir, exist_ok=True)
        out, names = [], []
        for i, (verts, faces) in enumerate(parts):
            part = trimesh.Trimesh(vertices=np.asarray(verts),
                                   faces=np.asarray(faces),
                                   process=False).convex_hull
            part.units = "meter"
            data = part.export(file_type="stl")
            name = f"part_{i}.stl"
            with open(os.path.join(part_dir, name), "wb") as f:
                f.write(data)
            out.append(data)
            names.append(name)
        with open(manifest, "w") as f:
            json.dump(names, f)
        return out


def hull_part_stls(src, *, load_mesh_metres):
    hull = load_mesh_metres(src).convex_hull
    hull.units = "meter"
    return [hull.export(file_type="stl")]


def collision_part_stls(src, mode, quality, cache_dir, *, load_mesh_metres,
                        coacd_part_stls_fn=None):
    if mode == "hull":
        return hull_part_stls(src, load_mesh_metres=load_mesh_metres)
    builder = coacd_part_stls if coacd_part_stls_fn is None else coacd_part_stls_fn
    return builder(src, quality, cache_dir, load_mesh_metres=load_mesh_metres)


def skrobot_primitive_available():
    import importlib.util
    try:
        return importlib.util.find_spec(
            "skrobot.utils.primitive_fitting") is not None
    except (ImportError, ValueError):
        return False


def fit_collision_primitive(src, primitive_type, *, load_mesh_metres):
    import trimesh
    from skrobot.utils import fit_primitive_to_mesh, primitive_params_to_origin

    mesh = load_mesh_metres(src)
    params = fit_primitive_to_mesh(mesh, primitive_type=primitive_type,
                                   oriented=None)
    xyz, rpy = primitive_params_to_origin(params)
    M = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
    M[:3, 3] = xyz
    return params, M


def primitive_geometry_el(params, *, fmt_num):
    geo = ET.Element("geometry")
    t = params["type"]
    if t == "box":
        ET.SubElement(geo, "box").set(
            "size", " ".join(fmt_num(v) for v in params["extents"]))
    elif t == "sphere":
        ET.SubElement(geo, "sphere").set("radius", fmt_num(params["radius"]))
    elif t in ("cylinder", "capsule"):
        cyl = ET.SubElement(geo, "cylinder")
        cyl.set("radius", fmt_num(params["radius"]))
        length = float(params["height"])
        if t == "capsule":
            length += 2 * float(params["radius"])
        cyl.set("length", fmt_num(length))
    else:
        raise ValueError(f"unsupported primitive type: {t!r}")
    return geo


def collision_preview_meshes(src, mode, quality, cache_dir, *, load_mesh_metres,
                             coacd_part_stls_fn=None, fit_collision_primitive_fn=None):
    import trimesh

    if mode in _PRIMITIVE_MODES:
        from skrobot.utils.primitive_fitting import create_primitive_mesh

        fitter = (fit_collision_primitive if fit_collision_primitive_fn is None
                  else fit_collision_primitive_fn)
        params, _M = fitter(src, _PRIMITIVE_MODES[mode],
                            load_mesh_metres=load_mesh_metres)
        return [create_primitive_mesh(params)]
    blobs = collision_part_stls(
        src, mode, quality, cache_dir, load_mesh_metres=load_mesh_metres,
        coacd_part_stls_fn=coacd_part_stls_fn)
    return [trimesh.load(io.BytesIO(d), file_type="stl") for d in blobs]
