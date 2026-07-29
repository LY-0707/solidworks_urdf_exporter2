"""Shared geometry-preparation utilities for simulator backends.

This layer is intentionally backend-agnostic: it knows how to load meshes,
normalise units, bake local origins into vertices, and cache converted mesh
artifacts, but it does not know anything about ROS package layouts or MJCF.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import xml.etree.ElementTree as ET


def load_mesh_metres(src):
    """Load ``src`` and pin its geometry to metres.

    SolidWorks ``.3dxml`` tessellations are in millimetres; other supported mesh
    formats are already treated as metres by the current exporter pipeline.
    """
    import trimesh

    loaded = trimesh.load(src)
    if isinstance(loaded, trimesh.Scene):
        if hasattr(loaded, "to_geometry"):
            mesh = loaded.to_geometry()
        elif hasattr(loaded, "to_mesh"):
            mesh = loaded.to_mesh()
        else:
            mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if src.lower().endswith(".3dxml"):
        mesh.apply_scale(0.001)
    mesh.units = "meter"
    return mesh


def hex_to_rgba(hex_color):
    """``'#RRGGBB'`` / ``'#RRGGBBAA'`` -> uint8 ``[R, G, B, A]`` or ``None``."""
    if not hex_color:
        return None
    import numpy as np

    h = str(hex_color).strip().lstrip("#")
    if len(h) == 6:
        h += "ff"
    if len(h) != 8:
        return None
    try:
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4, 6)],
                        dtype=np.uint8)
    except ValueError:
        return None


def apply_uniform_color(mesh, hex_color):
    """Repaint ``mesh`` one solid colour when a per-link override is given."""
    rgba = hex_to_rgba(hex_color)
    if rgba is None:
        return mesh
    import numpy as np
    import trimesh

    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, face_colors=np.tile(rgba, (len(mesh.faces), 1)))
    return mesh


def baked_mesh(mesh, transform):
    """Apply a 4x4 transform to mesh vertices when one is provided."""
    if transform is not None:
        import numpy as np

        mesh.apply_transform(np.asarray(transform, float))
    return mesh


def mesh_to_dae_bytes(src, collada_meshes, color=None, transform=None, mesh=None):
    """COLLADA bytes in metres, with optional colour/origin baking."""
    m = mesh if mesh is not None else load_mesh_metres(src)
    meshes = collada_meshes(apply_uniform_color(baked_mesh(m, transform), color))
    if len(meshes) == 1:
        return meshes[0].export(file_type="dae")
    from trimesh.exchange.dae import export_collada
    return export_collada(meshes)


def mesh_to_stl_bytes(src, color=None, transform=None, mesh=None):
    """STL bytes in metres, with optional origin baking."""
    del color
    m = mesh if mesh is not None else load_mesh_metres(src)
    return baked_mesh(m, transform).export(file_type="stl")


def mesh_to_glb_bytes(src, color=None, transform=None, mesh=None):
    """GLB bytes in metres, with optional colour/origin baking."""
    m = mesh if mesh is not None else load_mesh_metres(src)
    return apply_uniform_color(baked_mesh(m, transform), color).export(
        file_type="glb")


def _atomic_write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


class MeshCache:
    """Load-once, convert-many cache for mesh preparation."""

    _CACHE_VERSION = 1

    def __init__(self, cache_dir, converters):
        self._cache_dir = cache_dir
        self._converters = dict(converters)
        self._meshes = {}
        self._shas = {}

    def sha(self, src):
        key = os.path.abspath(src)
        h = self._shas.get(key)
        if h is None:
            with open(src, "rb") as f:
                h = hashlib.sha1(f.read()).hexdigest()
            self._shas[key] = h
        return h

    def _ensure(self, src):
        key = os.path.abspath(src)
        mesh = self._meshes.get(key)
        if mesh is None:
            mesh = self._meshes[key] = load_mesh_metres(src)
        return mesh

    def copy(self, src):
        return self._ensure(src).copy()

    def measure(self, src):
        import numpy as np

        path = os.path.join(
            self._cache_dir,
            f"{self.sha(src)}.measure{self._CACHE_VERSION}.json")
        try:
            with open(path) as f:
                data = json.load(f)
            return float(data[0]), np.asarray(data[1:4], float)
        except (OSError, ValueError, IndexError, TypeError):
            pass
        try:
            mesh = self._ensure(src)
        except Exception:
            return None
        size = float(np.linalg.norm(mesh.extents))
        centroid = np.asarray(mesh.centroid, float)
        try:
            _atomic_write(
                path,
                json.dumps([size, *centroid.tolist()]).encode("utf-8"))
        except OSError:
            pass
        return size, centroid

    def convert(self, src, fmt, color, transform):
        tkey = None if transform is None else tuple(
            round(float(x), 7) for x in transform.flatten())
        h = hashlib.sha1(self.sha(src).encode())
        h.update(json.dumps([fmt, color, tkey, self._CACHE_VERSION],
                            sort_keys=True).encode())
        path = os.path.join(self._cache_dir, f"{h.hexdigest()[:16]}.{fmt}")
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            pass
        data = self._converters[fmt](
            src, color=color, transform=transform, mesh=self.copy(src))
        try:
            _atomic_write(path, data)
        except OSError:
            pass
        return data


def origin_matrix(origin_el):
    """4x4 transform for a URDF ``<origin>`` element, or identity when absent."""
    import numpy as np
    import trimesh

    if origin_el is None:
        return np.eye(4)
    xyz = [float(v) for v in (origin_el.get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (origin_el.get("rpy") or "0 0 0").split()]
    m = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
    m[:3, 3] = xyz[:3]
    return m


def fmt_num(v):
    """Format a float for URDF attributes, snapping FP noise to exact 0."""
    v = float(v)
    if abs(v) < 1e-9:
        return "0"
    return repr(round(v, 9))


def set_origin_el(el, matrix):
    import numpy as np

    from .geometry import matrix_to_xyz_rpy

    origin = el.find("origin")
    if origin is None:
        origin = ET.SubElement(el, "origin")
    xyz, rpy = matrix_to_xyz_rpy(np.asarray(matrix, float))
    origin.set("xyz", " ".join(fmt_num(v) for v in xyz))
    origin.set("rpy", " ".join(fmt_num(v) for v in rpy))


def tf_marker_scale(root, pkg_dir, own_pkgs, cache, resolve_mesh):
    """Median visual-mesh size, capped for TF marker display."""
    sizes = []
    for link in root.findall("link"):
        vis = link.find("visual")
        mesh_el = vis.find(".//mesh") if vis is not None else None
        if mesh_el is None:
            continue
        _base, src, _ext = resolve_mesh(
            pkg_dir, mesh_el.get("filename"), own_pkgs=own_pkgs)
        if src is None:
            continue
        res = cache.measure(src)
        if res is not None and res[0] > 0.0:
            sizes.append(res[0])
    if not sizes:
        return 1.0
    sizes.sort()
    return round(min(1.0, max(0.02, sizes[len(sizes) // 2])), 4)


def bake_origins(root, pkg_dir, own_pkgs, cache, resolve_mesh):
    """Bake visual/collision origins into vertices and recenter fixed links."""
    import numpy as np

    bake = {}
    sizes = []
    child_joint, child_lists = {}, {}
    for joint in root.findall("joint"):
        child = joint.find("child").get("link")
        parent = joint.find("parent").get("link")
        child_joint[child] = joint
        child_lists.setdefault(parent, []).append(joint)

    for link in root.findall("link"):
        name = link.get("name")
        parent_joint = child_joint.get(name)
        fixed = parent_joint is not None and parent_joint.get("type") == "fixed"
        vis = link.find("visual")
        vis_matrix = origin_matrix(vis.find("origin")) if vis is not None else np.eye(4)
        delta = np.zeros(3)
        if vis is not None:
            mesh_el = vis.find(".//mesh")
            if mesh_el is not None:
                _base, src, _ext = resolve_mesh(
                    pkg_dir, mesh_el.get("filename"), own_pkgs=own_pkgs)
                if src is not None:
                    res = cache.measure(src)
                    if res is not None:
                        size, centroid = res
                        sizes.append(size)
                        if fixed:
                            delta = (vis_matrix @ np.append(centroid, 1.0))[:3]
        moved = float(np.linalg.norm(delta)) > 1e-9
        t_neg = np.eye(4)
        t_neg[:3, 3] = -delta
        if moved:
            t_pos = np.eye(4)
            t_pos[:3, 3] = delta
            set_origin_el(parent_joint, origin_matrix(parent_joint.find("origin")) @ t_pos)
            for child in child_lists.get(name, []):
                set_origin_el(child, t_neg @ origin_matrix(child.find("origin")))
            inertial = link.find("inertial")
            if inertial is not None:
                set_origin_el(inertial, t_neg @ origin_matrix(inertial.find("origin")))
        for ctx in ("visual", "collision"):
            for block in link.findall(ctx):
                block_matrix = origin_matrix(block.find("origin"))
                mesh_bake = t_neg @ block_matrix
                if not np.allclose(mesh_bake, np.eye(4), atol=1e-7):
                    for mesh_el in block.iter("mesh"):
                        bake[id(mesh_el)] = mesh_bake
                origin = block.find("origin")
                if origin is None:
                    origin = ET.SubElement(block, "origin")
                origin.set("xyz", "0 0 0")
                origin.set("rpy", "0 0 0")
    sizes = [s for s in sizes if s > 0.0]
    tf_scale = 1.0
    if sizes:
        sizes.sort()
        tf_scale = round(min(1.0, max(0.02, sizes[len(sizes) // 2])), 4)
    return bake, tf_scale
