from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return project_root() / "data"


def artifacts_dir() -> Path:
    return project_root() / "artifacts"


def artifacts_demo_dir() -> Path:
    return project_root() / "artifacts_demo"


def resolve_artifacts_dir(required_files: list[str]) -> Path:
    primary = artifacts_dir()
    if all((primary / name).exists() for name in required_files):
        return primary

    demo = artifacts_demo_dir()
    if all((demo / name).exists() for name in required_files):
        return demo

    return primary
