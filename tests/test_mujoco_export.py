import os
import shutil
import xml.etree.ElementTree as ET
import json


import pytest

_SAMPLE_MESH = os.path.join("examples", "fingertip", "meshes",
                            "fingertip_back_1.3dxml")


def _make_pkg(tmp_path, robot="robot"):
    if not os.path.exists(_SAMPLE_MESH):
        pytest.skip("sample .3dxml mesh not present")
    (tmp_path / "meshes").mkdir()
    (tmp_path / "urdf").mkdir()
    shutil.copy(_SAMPLE_MESH, tmp_path / "meshes" / "part.3dxml")
    urdf = (
        f'<?xml version="1.0"?>\n<robot name="{robot}">\n'
        '  <link name="base_link">\n'
        '    <visual><geometry>'
        '<mesh filename="../meshes/part.3dxml"/></geometry></visual>\n'
        '    <collision><geometry>'
        '<mesh filename="../meshes/part.3dxml"/></geometry></collision>\n'
        '  </link>\n</robot>\n')
    (tmp_path / "urdf" / f"{robot}.urdf").write_text(urdf, encoding="utf-8")
    return str(tmp_path)


def test_build_mujoco_package_emits_only_urdf_and_meshes(tmp_path):
    from sw2robot.exporter.mujoco_export import build_mujoco_package

    pkg_dir = _make_pkg(tmp_path, robot="fing")
    files = dict(build_mujoco_package(pkg_dir, "fing", collision="hull"))

    assert "fing_mujoco/package.xml" not in files
    assert "fing_mujoco/CMakeLists.txt" not in files
    assert "fing_mujoco/urdf/fing.urdf" in files
    assert any(arc.startswith("fing_mujoco/meshes/") for arc in files)


def test_build_mujoco_package_rewrites_mesh_uris_relative(tmp_path):
    from sw2robot.exporter.mujoco_export import build_mujoco_package

    pkg_dir = _make_pkg(tmp_path, robot="fing")
    files = dict(build_mujoco_package(pkg_dir, "fing", collision="hull"))
    root = ET.fromstring(files["fing_mujoco/urdf/fing.urdf"].decode())
    refs = [m.get("filename") for m in root.iter("mesh")]
    assert refs
    assert all(ref.startswith("../meshes/") for ref in refs)
    assert all("package://" not in ref for ref in refs)


def test_mujoco_backend_auto_collision_falls_back_to_hull(tmp_path, monkeypatch):
    from sw2robot.exporter.backend import ExportContext
    from sw2robot.exporter.mujoco_export import MujocoBackend

    seen = {}

    def fake_write(pkg_dir, robot_name, dest_dir, **kwargs):
        del pkg_dir, robot_name, dest_dir
        seen["collision"] = kwargs["collision"]
        return str(tmp_path / "out" / "fing_mujoco")

    monkeypatch.setattr(
        "sw2robot.exporter.mujoco_export.coacd_available", lambda: False)
    monkeypatch.setattr(
        "sw2robot.exporter.mujoco_export.write_mujoco_package", fake_write)

    ctx = ExportContext(pkg_dir="/tmp/pkg", robot_name="fing", urdf_path="/tmp/u")
    result = MujocoBackend().export(ctx)

    assert seen["collision"] == "auto"
    assert result.backend_name == "mujoco_urdf"
    assert result.output_path.endswith("fing_mujoco")


def test_build_mujoco_package_auto_chooses_hull_when_coacd_missing(
        tmp_path, monkeypatch):
    from sw2robot.exporter import mujoco_export

    pkg_dir = _make_pkg(tmp_path, robot="fing")
    monkeypatch.setattr(mujoco_export, "coacd_available", lambda: False)
    files = dict(mujoco_export.build_mujoco_package(pkg_dir, "fing"))

    root = ET.fromstring(files["fing_mujoco/urdf/fing.urdf"].decode())
    collision_ref = root.find("link").find("collision").find(".//mesh").get("filename")
    assert collision_ref.endswith("_collision_hull.stl")


def test_build_mujoco_package_does_not_route_through_ros_export(
        tmp_path, monkeypatch):
    from sw2robot.exporter import mujoco_export, ros_export

    pkg_dir = _make_pkg(tmp_path, robot="fing")

    def fail(*args, **kwargs):
        raise AssertionError("ROS export path should not be used")

    monkeypatch.setattr(ros_export, "build_ros_description", fail)
    files = dict(mujoco_export.build_mujoco_package(pkg_dir, "fing",
                                                    collision="hull"))

    assert "fing_mujoco/urdf/fing.urdf" in files


def test_write_mujoco_package_writes_export_report(tmp_path):
    from sw2robot.exporter.mujoco_export import write_mujoco_package

    src = tmp_path / "src"
    src.mkdir()
    pkg_dir = _make_pkg(src, robot="fing")
    out_dir = write_mujoco_package(pkg_dir, "fing", tmp_path / "out",
                                   collision="hull")
    report = json.loads(
        (tmp_path / "out" / "fing_mujoco" / "_sw2robot_export.json")
        .read_text(encoding="utf-8"))

    assert out_dir.endswith("fing_mujoco")
    assert report["backend"] == "mujoco_urdf"
    assert report["backend_options"]["resolved_collision"] == "hull"
    assert report["generated"]["urdf"] == "urdf/fing.urdf"
    assert report["generated"]["meshes"]
    assert report["generated"]["collision_pipeline"]
