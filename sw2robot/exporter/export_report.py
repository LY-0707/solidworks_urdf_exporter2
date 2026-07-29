"""Backend export report helpers.

The goal is traceability, not validation logic: every backend can leave a
machine-readable report describing which pipeline stages ran, which options were
used, and which output artifacts were materialized.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os

REPORT_BASENAME = "_sw2robot_export.json"
REPORT_SCHEMA_VERSION = 1


def relpath_or_abs(path, root):
    """Return ``path`` relative to ``root`` when it stays underneath it."""
    if not path:
        return path
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        return os.path.abspath(path)
    if rel.startswith(".."):
        return os.path.abspath(path)
    return rel.replace("\\", "/")


def make_step(step, module, function, **details):
    entry = {"step": step, "module": module, "function": function}
    if details:
        entry["details"] = details
    return entry


def build_report(*, backend_name, source_pkg_dir, robot_name, urdf_path,
                 output_path, backend_options=None, pipeline=None,
                 generated=None, sidecars=None, notes=None):
    """Build a JSON-serialisable export report payload."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "backend": backend_name,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "pkg_dir": os.path.abspath(source_pkg_dir),
            "robot_name": robot_name,
            "urdf_path": os.path.abspath(urdf_path),
        },
        "output": {
            "package_dir": os.path.abspath(output_path),
            "report_file": os.path.join(os.path.abspath(output_path),
                                         REPORT_BASENAME),
        },
        "backend_options": backend_options or {},
        "sidecars": sidecars or {},
        "pipeline": pipeline or [],
        "generated": generated or {},
        "notes": notes or [],
    }


def write_report(output_dir, report):
    """Write ``report`` under ``output_dir`` and return its path."""
    path = os.path.join(os.path.abspath(output_dir), REPORT_BASENAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def load_report(path):
    """Load a previously-written export report."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def update_report_validation(path, validation):
    """Attach validation data to an existing export report."""
    report = load_report(path)
    report["validation"] = validation
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    return path
