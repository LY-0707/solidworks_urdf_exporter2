"""Optional post-export validator integration.

This layer is intentionally thin: sw2robot does not implement the user's
validator, but it can invoke one after a backend export, persist the result, and
attach it to the backend's export report for traceability.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess

from .export_report import REPORT_BASENAME, update_report_validation

VALIDATION_REPORT_BASENAME = "_sw2robot_validation.json"


def validation_report_path(output_path):
    return os.path.join(os.path.abspath(output_path), VALIDATION_REPORT_BASENAME)


def _export_report_path(result):
    report_path = result.artifacts.get("report_path")
    if report_path:
        return report_path
    return os.path.join(os.path.abspath(result.output_path), REPORT_BASENAME)


def _template_fields(result, ctx):
    return {
        "backend": result.backend_name,
        "backend_name": result.backend_name,
        "output_path": os.path.abspath(result.output_path),
        "package_dir": os.path.abspath(result.output_path),
        "source_pkg_dir": os.path.abspath(ctx.pkg_dir),
        "robot_name": ctx.robot_name,
        "urdf_path": os.path.abspath(ctx.urdf_path),
        "report_path": _export_report_path(result),
        "validation_report_path": validation_report_path(result.output_path),
    }


def _parse_json_maybe(text):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _write_validation_report(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def run_external_validator(validate_cmd, result, ctx):
    """Run a user-provided validator command against a backend export.

    ``validate_cmd`` is a shell-style command template. Supported placeholders:
    ``{backend}``, ``{output_path}``, ``{package_dir}``, ``{source_pkg_dir}``,
    ``{robot_name}``, ``{urdf_path}``, ``{report_path}``,
    ``{validation_report_path}``.
    """
    fields = _template_fields(result, ctx)
    rendered = validate_cmd.format_map(fields)
    argv = shlex.split(rendered, posix=(os.name != "nt"))
    payload = {
        "status": "error",
        "backend": result.backend_name,
        "command": argv,
        "rendered_command": rendered,
        "report_path": fields["report_path"],
        "validation_report_path": fields["validation_report_path"],
    }
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, check=False)
    except OSError as e:
        payload.update({
            "error": f"{type(e).__name__}: {e}",
            "exit_code": None,
            "stdout": "",
            "stderr": "",
        })
    else:
        payload.update({
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        })
        parsed = _parse_json_maybe(proc.stdout)
        if parsed is not None:
            payload["result"] = parsed
        payload["status"] = "ok" if proc.returncode == 0 else "failed"
    _write_validation_report(fields["validation_report_path"], payload)
    if os.path.isfile(fields["report_path"]):
        update_report_validation(fields["report_path"], payload)
    result.artifacts["validation_report_path"] = fields["validation_report_path"]
    result.artifacts["validation"] = payload
    return payload
