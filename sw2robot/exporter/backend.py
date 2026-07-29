"""Minimal backend interface for build-time export targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .model import RobotModel
    from .state import GraphState


@dataclass
class ExportContext:
    """Shared build outputs handed to a backend exporter."""

    pkg_dir: str
    robot_name: str
    urdf_path: str
    graph: GraphState | None = None
    model: RobotModel | None = None
    sidecars: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportResult:
    """Result metadata for a backend export."""

    backend_name: str
    output_path: str
    artifacts: dict[str, Any] = field(default_factory=dict)


class ExportBackend(Protocol):
    """Small protocol so existing exports can be routed through backends."""

    name: str

    def export(self, ctx: ExportContext) -> ExportResult:
        ...
