"""MuJoCo-oriented backend export.

This backend keeps the current project on a URDF-centric workflow, but emits a
package shaped for MuJoCo import rather than ROS distribution:

* fixed links merged by default,
* visual/collision origins baked to zero,
* collision geometry separated from visuals,
* only the URDF + mesh artifacts are written (no ROS manifests/launch files).
"""

from __future__ import annotations

import copy
import os
import re
import xml.etree.ElementTree as ET

from .backend import ExportContext, ExportResult
from . import collision_pipeline as _collision
from .geometry_prepare import (
    MeshCache,
    bake_origins,
    fmt_num,
    load_mesh_metres,
    origin_matrix,
    set_origin_el,
)
from .export_report import (
    build_report as _build_report,
    make_step as _make_step,
    relpath_or_abs as _relpath_or_abs,
    write_report as _write_report,
)
from .urdf_common import (
    iter_sources,
    load_urdf_root,
    own_pkg_names,
    read_mass_only,
    resolve_mesh,
)

_VALID_COLLISION = {"auto", "copy", "hull", "coacd",
                    "primitive", "box", "cylinder", "sphere"}
_PRIMITIVE_MODES = _collision._PRIMITIVE_MODES
_COACD_PRESETS = _collision._COACD_PRESETS
_MESH_CONVERT = {"stl": lambda src, color=None, transform=None, mesh=None:
                 _mesh_to_stl_bytes(src, color=color, transform=transform,
                                    mesh=mesh)}


def _mesh_to_stl_bytes(src, color=None, transform=None, mesh=None):
    del color
    if mesh is None:
        mesh = load_mesh_metres(src)
    if transform is not None:
        mesh.apply_transform(transform)
    return mesh.export(file_type="stl")


def _resolve_collision_mode(collision):
    if collision == "auto":
        return "coacd" if _collision.coacd_available() else "hull"
    if collision not in _VALID_COLLISION:
        raise ValueError(f"unsupported MuJoCo collision mode: {collision!r}")
    return collision


def coacd_available():
    """Backward-compatible availability helper for callers/tests."""
    return _collision.coacd_available()


def _mujoco_pkg_name(robot_name, package_name=None):
    if package_name and package_name.strip():
        pkg = package_name.strip()
    else:
        base = re.sub(r"[^A-Za-z0-9_.-]", "_", robot_name).strip("_")
        pkg = (base or "robot") + "_mujoco"
    if pkg in (".", "..") or "/" in pkg or "\\" in pkg:
        raise ValueError(f"invalid MuJoCo package name: {package_name!r}")
    return pkg


def _urdf_stem(robot_name, urdf_name=None):
    if not urdf_name:
        return robot_name
    stem = urdf_name.strip()
    if stem.lower().endswith(".urdf"):
        stem = stem[:-5]
    if not stem or "/" in stem or "\\" in stem:
        raise ValueError(f"invalid MuJoCo URDF name: {urdf_name!r}")
    return stem


def _rel_mesh(name):
    return f"../meshes/{name}"


def _validate_mode(mode, coacd_quality):
    if mode == "coacd":
        if coacd_quality not in _COACD_PRESETS:
            raise ValueError(
                f"unsupported coacd_quality: {coacd_quality!r} "
                f"(use one of {sorted(_COACD_PRESETS)})")
        if not _collision.coacd_available():
            raise ValueError(
                "CoACD collision decomposition needs the optional 'coacd' "
                "package -- install it with: pip install coacd")
    if mode in _PRIMITIVE_MODES and not _collision.skrobot_primitive_available():
        raise ValueError(
            "primitive collision fitting needs scikit-robot with the per-mesh "
            "primitive API -- install/upgrade it with: pip install -U "
            "scikit-robot")


def _verbatim(job):
    return (job["src"].lower().endswith(".stl") and job["transform"] is None)


def _report_sidecars(pkg_dir):
    return {
        "mass_only": os.path.isfile(os.path.join(pkg_dir, "mass_only.yaml")),
        "loop_closures": os.path.isfile(
            os.path.join(pkg_dir, "loop_closures.yaml")),
    }


def _round_matrix(matrix):
    if matrix is None:
        return None
    return [[round(float(v), 9) for v in row] for row in matrix.tolist()]


def _build_mujoco_package_plan(pkg_dir, robot_name, package_name=None,
                               urdf_name=None, collision="auto",
                               coacd_quality="balanced", merge_fixed=True,
                               zero_origins=True, progress=None,
                               should_cancel=None, export_surface="backend"):
    """Return ``(files, report)`` for a MuJoCo-oriented package export."""
    import numpy as np

    mode = _resolve_collision_mode(collision)
    _validate_mode(mode, coacd_quality)
    pkg = _mujoco_pkg_name(robot_name, package_name)
    urdf_stem = _urdf_stem(robot_name, urdf_name)
    own_pkgs = own_pkg_names(pkg_dir)
    root = load_urdf_root(pkg_dir, robot_name)

    from .merge import merge_fixed_links

    mass_only = read_mass_only(pkg_dir)
    pipeline = [
        _make_step("load_urdf", "sw2robot.exporter.urdf_common",
                   "load_urdf_root",
                   urdf=f"urdf/{robot_name}.urdf"),
        _make_step("resolve_mesh_refs", "sw2robot.exporter.urdf_common",
                   "resolve_mesh", own_package_names=sorted(own_pkgs)),
    ]
    if merge_fixed:
        merge_fixed_links(root, force_merge=mass_only)
        pipeline.append(_make_step(
            "merge_fixed_links", "sw2robot.exporter.merge",
            "merge_fixed_links",
            mode="force_merge", forced_links=sorted(mass_only)))
    elif mass_only:
        merge_fixed_links(root, only=mass_only)
        pipeline.append(_make_step(
            "merge_mass_only_links", "sw2robot.exporter.merge",
            "merge_fixed_links", mode="only",
            forced_links=sorted(mass_only)))

    cache = MeshCache(os.path.join(pkg_dir, "meshes", ".mesh_cache"),
                      _MESH_CONVERT)
    if zero_origins:
        bake, _tf_scale = bake_origins(root, pkg_dir, own_pkgs, cache,
                                       resolve_mesh)
        pipeline.append(_make_step(
            "bake_visual_collision_origins",
            "sw2robot.exporter.geometry_prepare", "bake_origins"))
    else:
        bake = {}

    coacd_cache_dir = os.path.join(pkg_dir, "meshes", ".coacd_cache")
    files = []
    used_names = set()
    errors = []
    coll_done = {}
    prim_done = {}
    jobs = {}
    job_order = []
    collision_report = []
    if mode == "copy":
        collision_srcs = []
    else:
        collision_srcs = list(set(
            iter_sources(root, pkg_dir, own_pkgs, ("collision",),
                         resolve_mesh_fn=resolve_mesh)))
    coll_prog = {"done": 0}

    def _report(stage, done, total, detail=""):
        if progress:
            try:
                progress(stage, done, total, detail)
            except Exception:
                pass

    def _ckcancel():
        if should_cancel and should_cancel():
            from .urdf_common import ExportCancelled
            raise ExportCancelled()

    if collision_srcs:
        _report("collision", 0, len(collision_srcs))

    def _emit_collision(base, src):
        key = (os.path.abspath(src), mode, coacd_quality)
        if key in coll_done:
            return coll_done[key]
        try:
            blobs = _collision.collision_part_stls(
                src, mode, coacd_quality, coacd_cache_dir,
                load_mesh_metres=load_mesh_metres)
        except Exception as e:
            errors.append(f"{base}: {mode} collision failed ({e!r})")
            return None
        names = []
        for i, data in enumerate(blobs):
            stem = (f"{base}_collision_hull" if mode == "hull"
                    else f"{base}_collision_{i}")
            name, j = f"{stem}.stl", 1
            while name in used_names:
                j += 1
                name = f"{stem}_{j}.stl"
            used_names.add(name)
            files.append((f"{pkg}/meshes/{name}", data))
            names.append(name)
        coll_done[key] = names
        coll_prog["done"] += 1
        _report("collision", coll_prog["done"], len(collision_srcs), base)
        return names

    def _expand_collision(link):
        for block in list(link.findall("collision")):
            _ckcancel()
            meshes = list(block.iter("mesh"))
            if len(meshes) != 1:
                continue
            base, src, _ext = resolve_mesh(pkg_dir, meshes[0].get("filename"),
                                           own_pkgs=own_pkgs)
            if base is None:
                continue
            if src is None:
                errors.append(f"no source mesh for '{base}'")
                continue
            names = _emit_collision(base, src)
            if not names:
                continue
            m_bake = bake.get(id(meshes[0]))
            idx = list(link).index(block)
            link.remove(block)
            for k, name in enumerate(names):
                nb = copy.deepcopy(block)
                if m_bake is not None:
                    set_origin_el(nb, m_bake)
                next(nb.iter("mesh")).set("filename", _rel_mesh(name))
                link.insert(idx + k, nb)
            collision_report.append({
                "link": link.get("name"),
                "kind": "mesh_parts",
                "mode": mode,
                "source_mesh": _relpath_or_abs(src, pkg_dir),
                "outputs": [f"meshes/{name}" for name in names],
                "origin_baked": m_bake is not None,
                "origin_matrix": _round_matrix(m_bake),
                "call_chain": [
                    "sw2robot.exporter.mujoco_export._expand_collision",
                    "sw2robot.exporter.collision_pipeline.collision_part_stls",
                ],
            })

    def _fit_primitive_cached(base, src):
        key = (os.path.abspath(src), mode)
        if key in prim_done:
            return prim_done[key]
        try:
            res = _collision.fit_collision_primitive(
                src, _PRIMITIVE_MODES[mode], load_mesh_metres=load_mesh_metres)
        except Exception as e:
            errors.append(f"{base}: {mode} collision failed ({e!r})")
            return None
        prim_done[key] = res
        coll_prog["done"] += 1
        _report("collision", coll_prog["done"], len(collision_srcs), base)
        return res

    def _expand_collision_primitive(link):
        for block in list(link.findall("collision")):
            _ckcancel()
            meshes = list(block.iter("mesh"))
            if len(meshes) != 1:
                continue
            base, src, _ext = resolve_mesh(pkg_dir, meshes[0].get("filename"),
                                           own_pkgs=own_pkgs)
            if base is None:
                continue
            if src is None:
                errors.append(f"no source mesh for '{base}'")
                continue
            res = _fit_primitive_cached(base, src)
            if res is None:
                continue
            params, matrix = res
            m_bake = bake.get(id(meshes[0]))
            base_matrix = (m_bake if m_bake is not None
                           else origin_matrix(block.find("origin")))
            geo = block.find("geometry")
            new_geo = _collision.primitive_geometry_el(
                params, fmt_num=fmt_num)
            if geo is not None:
                gi = list(block).index(geo)
                block.remove(geo)
                block.insert(gi, new_geo)
            else:
                block.append(new_geo)
            final_matrix = np.asarray(base_matrix, float) @ matrix
            set_origin_el(block, final_matrix)
            collision_report.append({
                "link": link.get("name"),
                "kind": "primitive",
                "mode": mode,
                "source_mesh": _relpath_or_abs(src, pkg_dir),
                "primitive_type": params.get("type"),
                "origin_baked": m_bake is not None,
                "origin_matrix": _round_matrix(final_matrix),
                "call_chain": [
                    "sw2robot.exporter.mujoco_export._expand_collision_primitive",
                    "sw2robot.exporter.collision_pipeline.fit_collision_primitive",
                    "sw2robot.exporter.collision_pipeline.primitive_geometry_el",
                ],
            })

    for link in root.findall("link"):
        _ckcancel()
        if mode in ("coacd", "hull"):
            _expand_collision(link)
        elif mode in _PRIMITIVE_MODES:
            _expand_collision_primitive(link)
        for ctx in ("visual", "collision"):
            if ctx == "collision" and mode != "copy":
                continue
            for block in link.findall(ctx):
                for mesh in block.iter("mesh"):
                    base, src, _ext = resolve_mesh(pkg_dir, mesh.get("filename"),
                                                   own_pkgs=own_pkgs)
                    if base is None:
                        continue
                    if src is None:
                        errors.append(f"no source mesh for '{base}'")
                        continue
                    transform = bake.get(id(mesh))
                    tkey = None if transform is None else tuple(
                        round(float(x), 7) for x in transform.flatten())
                    key = (os.path.abspath(src), "stl", tkey)
                    job = jobs.get(key)
                    if job is None:
                        name, i = f"{base}.stl", 1
                        while name in used_names:
                            i += 1
                            name = f"{base}_{i}.stl"
                        used_names.add(name)
                        job = jobs[key] = {
                            "base": base,
                            "src": src,
                            "transform": transform,
                            "name": name,
                            "contexts": [],
                        }
                        job_order.append(key)
                    job["contexts"].append({
                        "link": link.get("name"),
                        "role": ctx,
                        "origin_baked": transform is not None,
                    })
                    mesh.set("filename", _rel_mesh(job["name"]))

    if errors:
        raise RuntimeError("MuJoCo export failed: " + "; ".join(errors))

    mesh_report = []
    _report("meshes", 0, len(job_order))
    for key in job_order:
        _ckcancel()
        job = jobs[key]
        _report("meshes", len(mesh_report), len(job_order), job["base"])
        try:
            if _verbatim(job):
                with open(job["src"], "rb") as f:
                    data = f.read()
            else:
                data = cache.convert(job["src"], "stl", None, job["transform"])
        except Exception as e:
            errors.append(f"{job['base']}: stl convert failed ({e!r})")
            continue
        files.append((f"{pkg}/meshes/{job['name']}", data))
        mesh_report.append({
            "source_mesh": _relpath_or_abs(job["src"], pkg_dir),
            "output_mesh": f"meshes/{job['name']}",
            "contexts": job["contexts"],
            "origin_baked": job["transform"] is not None,
            "origin_matrix": _round_matrix(job["transform"]),
            "verbatim_copy": _verbatim(job),
            "call_chain": [
                "sw2robot.exporter.mujoco_export.build_mujoco_package",
                "sw2robot.exporter.geometry_prepare.MeshCache.convert",
            ],
        })
    _report("meshes", len(job_order), len(job_order))

    if errors:
        raise RuntimeError("MuJoCo export failed: " + "; ".join(errors))

    new_urdf = '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode")
    if not new_urdf.endswith("\n"):
        new_urdf += "\n"
    files.append((f"{pkg}/urdf/{urdf_stem}.urdf", new_urdf.encode("utf-8")))
    pipeline.append(_make_step(
        "emit_urdf_and_meshes", "sw2robot.exporter.mujoco_export",
        "build_mujoco_package",
        collision_mode=mode,
        visual_mesh_format="stl",
        collision_mesh_format="stl" if mode != "primitive" else "primitive",
        merge_fixed=merge_fixed,
        zero_origins=zero_origins))
    report = {
        "backend_options": {
            "package_name": pkg,
            "urdf_name": urdf_stem,
            "collision": collision,
            "resolved_collision": mode,
            "coacd_quality": coacd_quality,
            "merge_fixed": merge_fixed,
            "zero_origins": zero_origins,
            "export_surface": export_surface,
        },
        "sidecars": _report_sidecars(pkg_dir),
        "pipeline": pipeline,
        "generated": {
            "urdf": f"urdf/{urdf_stem}.urdf",
            "mesh_count": len(mesh_report),
            "collision_item_count": len(collision_report),
            "meshes": mesh_report,
            "collision_pipeline": collision_report,
        },
    }
    return files, report


def build_mujoco_package(pkg_dir, robot_name, package_name=None, urdf_name=None,
                         collision="auto", coacd_quality="balanced",
                         merge_fixed=True, zero_origins=True, progress=None,
                         should_cancel=None, export_surface="backend"):
    """``pkg_dir`` -> ``[(arcname, bytes), ...]`` for a MuJoCo-oriented URDF pkg."""
    files, _report = _build_mujoco_package_plan(
        pkg_dir, robot_name, package_name=package_name, urdf_name=urdf_name,
        collision=collision, coacd_quality=coacd_quality,
        merge_fixed=merge_fixed, zero_origins=zero_origins,
        progress=progress, should_cancel=should_cancel,
        export_surface=export_surface)
    return files


def write_mujoco_package(pkg_dir, robot_name, dest_dir, package_name=None,
                         urdf_name=None, collision="auto",
                         coacd_quality="balanced", merge_fixed=True,
                         zero_origins=True, progress=None,
                         should_cancel=None, export_surface="backend"):
    """Write the MuJoCo-oriented URDF package under ``dest_dir``."""
    pkg = _mujoco_pkg_name(robot_name, package_name)
    files, report_bits = _build_mujoco_package_plan(
        pkg_dir, robot_name, package_name=pkg, urdf_name=urdf_name,
        collision=collision, coacd_quality=coacd_quality,
        merge_fixed=merge_fixed, zero_origins=zero_origins,
        progress=progress, should_cancel=should_cancel,
        export_surface=export_surface)
    root = os.path.abspath(dest_dir)
    for arc, data in files:
        dst = os.path.join(root, *arc.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(data)
    out_dir = os.path.join(root, pkg)
    report = _build_report(
        backend_name="mujoco_urdf",
        source_pkg_dir=pkg_dir,
        robot_name=robot_name,
        urdf_path=os.path.join(pkg_dir, "urdf", robot_name + ".urdf"),
        output_path=out_dir,
        backend_options=report_bits["backend_options"],
        sidecars=report_bits["sidecars"],
        pipeline=report_bits["pipeline"],
        generated=report_bits["generated"])
    _write_report(out_dir, report)
    return out_dir


class MujocoBackend:
    """Backend adapter that writes a MuJoCo-oriented URDF package."""

    name = "mujoco_urdf"

    def __init__(self, *, dest_dir=None, package_name=None, urdf_name=None,
                 collision="auto", coacd_quality="balanced",
                 merge_fixed=True, zero_origins=True,
                 export_surface="backend"):
        self.dest_dir = dest_dir
        self.package_name = package_name
        self.urdf_name = urdf_name
        self.collision = collision
        self.coacd_quality = coacd_quality
        self.merge_fixed = merge_fixed
        self.zero_origins = zero_origins
        self.export_surface = export_surface

    def export(self, ctx: ExportContext) -> ExportResult:
        dest_dir = (self.dest_dir if self.dest_dir is not None
                    else os.path.dirname(os.path.abspath(ctx.pkg_dir)))
        out_dir = write_mujoco_package(
            ctx.pkg_dir, ctx.robot_name, dest_dir,
            package_name=self.package_name, urdf_name=self.urdf_name,
            collision=self.collision, coacd_quality=self.coacd_quality,
            merge_fixed=self.merge_fixed, zero_origins=self.zero_origins,
            progress=ctx.sidecars.get("_export_progress"),
            should_cancel=ctx.sidecars.get("_export_should_cancel"),
            export_surface=self.export_surface)
        return ExportResult(
            backend_name=self.name,
            output_path=out_dir,
            artifacts={
                "package_dir": out_dir,
                "report_path": os.path.join(
                    out_dir, "_sw2robot_export.json"),
            })
