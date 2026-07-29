"""URDF/package helpers shared by simulator backends."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

CONVERTIBLE_EXTS = (".3dxml", ".glb", ".stl", ".dae", ".obj")


def load_urdf_root(pkg_dir, robot_name):
    """Parse ``urdf/<robot_name>.urdf`` while preserving XML comments."""
    urdf_path = os.path.join(pkg_dir, "urdf", robot_name + ".urdf")
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(urdf_path, parser=parser).getroot()


def read_mass_only(pkg_dir):
    """The mass-only link names build() persisted beside the working package."""
    side = os.path.join(pkg_dir, "mass_only.yaml")
    if not os.path.isfile(side):
        return set()
    try:
        import yaml
        with open(side, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return set(data or ())
    except Exception:
        return set()


def own_pkg_names(pkg_dir):
    """Package names that should be treated as self-references."""
    names = {os.path.basename(os.path.normpath(pkg_dir))}
    mani = os.path.join(pkg_dir, "package.xml")
    if os.path.exists(mani):
        try:
            name = ET.parse(mani).getroot().findtext("name")
            if name and name.strip():
                names.add(name.strip())
        except Exception:
            pass
    return names


def resolve_mesh(pkg_dir, ref, own_pkgs=None, convertible_exts=CONVERTIBLE_EXTS):
    """Resolve a URDF mesh ref to ``(base, source_path, ext)``.

    Returns ``(None, None, None)`` for external or unsupported refs, and
    ``(base, None, ext)`` when the ref is ours but the source file is missing.
    """
    if not ref:
        return None, None, None
    ref = ref.replace("\\", "/")
    is_pkg = "://" in ref
    root = os.path.normpath(pkg_dir)
    if is_pkg:
        if not ref.startswith("package://"):
            return None, None, None
        ref_pkg, _, rel = ref.split("://", 1)[1].partition("/")
        if own_pkgs is not None and ref_pkg not in own_pkgs:
            return None, None, None
        base_dir = root
    else:
        if ref.startswith("/") or re.match(r"^[A-Za-z]:", ref):
            return None, None, None
        rel = ref
        base_dir = os.path.join(root, "urdf")
    base, ext = os.path.splitext(os.path.basename(rel))
    if ext.lower() not in convertible_exts:
        return None, None, None
    src = os.path.normpath(os.path.join(base_dir, *rel.split("/")))
    try:
        escapes = os.path.commonpath([root, src]) != root
    except ValueError:
        escapes = True
    if escapes:
        return None, None, None
    if not os.path.exists(src):
        stem = os.path.splitext(src)[0]
        for alt_ext in convertible_exts:
            if os.path.exists(stem + alt_ext):
                src = stem + alt_ext
                break
        else:
            return base, None, ext
    return base, src, ext


def iter_sources(root, pkg_dir, own_pkgs, contexts, *, resolve_mesh_fn=resolve_mesh):
    """Yield local source meshes for the requested URDF contexts."""
    for link in root.findall("link"):
        for ctx in contexts:
            for block in link.findall(ctx):
                for mesh in block.iter("mesh"):
                    _base, src, _ext = resolve_mesh_fn(
                        pkg_dir, mesh.get("filename"), own_pkgs=own_pkgs)
                    if src is not None:
                        yield src


class ExportCancelled(Exception):
    """Raised when an export is cancelled cooperatively."""

