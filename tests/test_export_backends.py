from sw2robot.exporter.backend import ExportContext
from sw2robot.exporter.ros_export import RosDescriptionBackend
from sw2robot.exporter.export_validation import run_external_validator
import json
from pathlib import Path

_TRI_STL = ("solid t\n facet normal 0 0 1\n  outer loop\n"
            "   vertex 0 0 0\n   vertex 0.1 0 0\n   vertex 0 0.1 0\n"
            "  endloop\n endfacet\nendsolid t\n")


def test_ros_description_backend_uses_context_sidecars(tmp_path, monkeypatch):
    seen = {}

    def fake_write(pkg_dir, robot_name, dest_dir, **kwargs):
        seen["pkg_dir"] = pkg_dir
        seen["robot_name"] = robot_name
        seen["dest_dir"] = dest_dir
        seen["kwargs"] = kwargs
        return str(tmp_path / "out" / "robot_description")

    monkeypatch.setattr(
        "sw2robot.exporter.ros_export.write_ros_description_package", fake_write)

    ctx = ExportContext(
        pkg_dir="/tmp/pkg",
        robot_name="robot",
        urdf_path="/tmp/pkg/urdf/robot.urdf",
        sidecars={"loop_closures": {"closures": [{"joint": "a"}]}})
    result = RosDescriptionBackend(ros_version=2, collision="hull").export(ctx)

    assert seen["pkg_dir"] == "/tmp/pkg"
    assert seen["robot_name"] == "robot"
    assert seen["kwargs"]["ros_version"] == 2
    assert seen["kwargs"]["collision"] == "hull"
    assert seen["kwargs"]["loop_closures"] == {"closures": [{"joint": "a"}]}
    assert result.backend_name == "ros_description"
    assert result.output_path.endswith("robot_description")


def test_ros_description_backend_explicit_loop_closures_override_context(
        tmp_path, monkeypatch):
    seen = {}

    def fake_write(pkg_dir, robot_name, dest_dir, **kwargs):
        del pkg_dir, robot_name, dest_dir
        seen["loop_closures"] = kwargs["loop_closures"]
        return str(tmp_path / "out" / "robot_description")

    monkeypatch.setattr(
        "sw2robot.exporter.ros_export.write_ros_description_package", fake_write)

    ctx = ExportContext(
        pkg_dir="/tmp/pkg",
        robot_name="robot",
        urdf_path="/tmp/pkg/urdf/robot.urdf",
        sidecars={"loop_closures": {"closures": [{"joint": "from_ctx"}]}})
    explicit = {"closures": [{"joint": "explicit"}]}
    RosDescriptionBackend(loop_closures=explicit).export(ctx)

    assert seen["loop_closures"] == explicit


def test_build_routes_to_mujoco_backend(tmp_path, monkeypatch):
    from sw2robot.exporter import export as exportmod

    class FakeGraph:
        robot_name = "robot"

    class FakeModel:
        components = ["a"]
        joints = ["b"]
        loop_closures = None

    seen = {}
    pkg = tmp_path / "pkg"
    (pkg / "urdf").mkdir(parents=True)

    monkeypatch.setattr(exportmod.GraphState, "load",
                        staticmethod(lambda _path: FakeGraph()))
    monkeypatch.setattr(exportmod, "build_model",
                        lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(exportmod, "write_urdf",
                        lambda _model, urdf_path, **kwargs: set())
    monkeypatch.setattr(exportmod, "write_ros_package",
                        lambda _model, _pkg_dir: None)
    monkeypatch.setattr(exportmod.jointcfg, "write_template",
                        lambda _model, _tmpl: None)

    class FakeBackend:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def export(self, ctx):
            seen["ctx"] = ctx
            class Result:
                output_path = str(tmp_path / "robot_mujoco")
            return Result()

    monkeypatch.setattr("sw2robot.exporter.mujoco_export.MujocoBackend",
                        FakeBackend)

    urdf = exportmod.build(str(pkg), mujoco_pkg=True, mujoco_collision="hull",
                           mujoco_zero_origins=False)

    assert urdf.endswith("/urdf/robot.urdf")
    assert seen["kwargs"]["collision"] == "hull"
    assert seen["kwargs"]["zero_origins"] is False
    assert seen["ctx"].robot_name == "robot"


def test_ros_description_backend_reports_export_path(tmp_path):
    from sw2robot.exporter.ros_export import write_ros_description_package

    pkg = tmp_path / "pkg"
    (pkg / "meshes").mkdir(parents=True)
    (pkg / "urdf").mkdir()
    (pkg / "meshes" / "part.stl").write_text(_TRI_STL, encoding="utf-8")
    (pkg / "urdf" / "robot.urdf").write_text(
        '<?xml version="1.0"?><robot name="robot"><link name="base">'
        '<visual><geometry><mesh filename="../meshes/part.stl"/>'
        '</geometry></visual></link></robot>', encoding="utf-8")

    out = write_ros_description_package(str(pkg), "robot", tmp_path / "out")
    report = json.loads(
        (tmp_path / "out" / "robot_description" / "_sw2robot_export.json")
        .read_text(encoding="utf-8"))

    assert out.endswith("robot_description")
    assert report["backend"] == "ros_description"
    assert report["generated"]["urdf"] == "urdf/robot_description.urdf"


def test_external_validator_writes_sidecar_and_updates_export_report(
        tmp_path, monkeypatch):
    from sw2robot.exporter.backend import ExportResult
    from sw2robot.exporter.export_report import build_report, write_report

    out = tmp_path / "robot_mujoco"
    out.mkdir()
    report_path = write_report(
        out,
        build_report(
            backend_name="mujoco_urdf",
            source_pkg_dir=tmp_path / "src",
            robot_name="robot",
            urdf_path=tmp_path / "src" / "urdf" / "robot.urdf",
            output_path=out))
    result = ExportResult(
        backend_name="mujoco_urdf",
        output_path=str(out),
        artifacts={"report_path": report_path})
    ctx = ExportContext(
        pkg_dir=str(tmp_path / "src"),
        robot_name="robot",
        urdf_path=str(tmp_path / "src" / "urdf" / "robot.urdf"))

    class CP:
        returncode = 0
        stdout = '{"issues":[{"code":"mesh_origin"}]}'
        stderr = "ok"

    monkeypatch.setattr(
        "sw2robot.exporter.export_validation.subprocess.run",
        lambda *args, **kwargs: CP())

    payload = run_external_validator(
        'validator --pkg "{output_path}" --report "{report_path}"',
        result, ctx)

    saved = json.loads(
        (out / "_sw2robot_validation.json").read_text(encoding="utf-8"))
    merged = json.loads(Path(report_path).read_text(encoding="utf-8"))

    assert payload["status"] == "ok"
    assert saved["result"]["issues"][0]["code"] == "mesh_origin"
    assert merged["validation"]["status"] == "ok"


def test_build_runs_validator_hook_for_backend(tmp_path, monkeypatch):
    from sw2robot.exporter import export as exportmod

    class FakeGraph:
        robot_name = "robot"

    class FakeModel:
        components = ["a"]
        joints = ["b"]
        loop_closures = None

    seen = {}
    pkg = tmp_path / "pkg"
    (pkg / "urdf").mkdir(parents=True)

    monkeypatch.setattr(exportmod.GraphState, "load",
                        staticmethod(lambda _path: FakeGraph()))
    monkeypatch.setattr(exportmod, "build_model",
                        lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(exportmod, "write_urdf",
                        lambda _model, urdf_path, **kwargs: set())
    monkeypatch.setattr(exportmod, "write_ros_package",
                        lambda _model, _pkg_dir: None)
    monkeypatch.setattr(exportmod.jointcfg, "write_template",
                        lambda _model, _tmpl: None)

    class FakeBackend:
        def __init__(self, **kwargs):
            pass

        def export(self, ctx):
            class Result:
                output_path = str(tmp_path / "robot_mujoco")
                artifacts = {"report_path": str(tmp_path / "report.json")}
                backend_name = "mujoco_urdf"
            seen["ctx"] = ctx
            return Result()

    monkeypatch.setattr("sw2robot.exporter.mujoco_export.MujocoBackend",
                        FakeBackend)
    monkeypatch.setattr(
        "sw2robot.exporter.export._maybe_validate_export",
        lambda result, ctx, cmd: seen.update(
            validate=(result.backend_name, ctx.robot_name, cmd)))

    exportmod.build(str(pkg), mujoco_pkg=True, validate_cmd="validator {output_path}")

    assert seen["validate"] == (
        "mujoco_urdf", "robot", "validator {output_path}")
