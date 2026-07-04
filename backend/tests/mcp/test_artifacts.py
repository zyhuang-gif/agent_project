import pytest
from pathlib import Path

from app.mcp.artifacts import (
    Artifact,
    ArtifactError,
    artifact_root,
    write_markdown_artifact,
)


def test_write_markdown_artifact_creates_file_inside_root(tmp_path):
    root = tmp_path / "runtime" / "artifacts"
    art = write_markdown_artifact(
        source_answer="# 报告\n\n正文内容",
        base_name="张三-薪酬报告",
        root=root,
    )
    assert isinstance(art, Artifact)
    p = Path(art.path)
    assert p.exists()
    # 必须落在 root/send 下
    assert p.resolve().is_relative_to((root / "send").resolve())
    assert art.mime == "text/markdown"
    assert art.name.endswith(".md")
    assert p.read_text(encoding="utf-8") == "# 报告\n\n正文内容"
    assert art.size == len("# 报告\n\n正文内容".encode("utf-8"))


def test_write_markdown_artifact_sanitizes_dangerous_base_name(tmp_path):
    art = write_markdown_artifact(
        source_answer="x",
        base_name="../../../etc/passwd",
        root=tmp_path,
    )
    p = Path(art.path).resolve()
    assert p.is_relative_to((tmp_path / "send").resolve())
    # 名称里的 ../ 会被 sanitize，绝不会含分隔符
    assert "/" not in art.name and "\\" not in art.name
    assert not art.name.startswith(".")


def test_write_markdown_artifact_rejects_non_markdown_extension(tmp_path):
    with pytest.raises(ArtifactError):
        write_markdown_artifact(
            source_answer="x",
            base_name="report",
            root=tmp_path,
            extension=".exe",
        )


def test_write_markdown_artifact_rejects_oversize(tmp_path):
    huge = "a" * (5 * 1024 * 1024 + 1)
    with pytest.raises(ArtifactError):
        write_markdown_artifact(
            source_answer=huge,
            base_name="big",
            root=tmp_path,
        )


def test_artifact_root_defaults_to_backend_runtime(monkeypatch):
    monkeypatch.delenv("ARTIFACT_ROOT", raising=False)
    r = artifact_root()
    # 允许项目内任意深度；只要末尾是 runtime/artifacts
    assert r.name == "artifacts"
    assert r.parent.name == "runtime"


def test_artifact_root_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    assert artifact_root() == tmp_path.resolve()
