"""受控目录 Markdown artifact 渲染（P1）。

约束（spec §2.4 / §6）：
- 只支持 .md。
- 写入位置严格限制在 artifact_root()/send/ 下，用 Path.resolve() + is_relative_to 校验。
- 大小上限 5 MiB。
- 文件名 sanitize，去掉路径分隔符与前导 dot，防止路径穿越 / 隐藏文件。
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ALLOWED_EXTENSIONS = frozenset({".md"})
MAX_ARTIFACT_BYTES = 5 * 1024 * 1024  # 5 MiB
_SAFE_NAME_RE = re.compile(r"[^\w.\-一-鿿]+", re.UNICODE)


class ArtifactError(ValueError):
    """artifact 生成/校验错误——保证在真正调用外部 API 前失败。"""


@dataclass(frozen=True)
class Artifact:
    path: str
    name: str
    mime: str
    size: int


def artifact_root() -> Path:
    override = os.getenv("ARTIFACT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # backend/app/mcp/artifacts.py → parents[2] = backend/
    return (Path(__file__).resolve().parents[2] / "runtime" / "artifacts").resolve()


def _sanitize_base_name(base_name: str) -> str:
    # 去掉分隔符 → sanitize → 截断
    stem = Path((base_name or "").strip()).name
    stem = _SAFE_NAME_RE.sub("_", stem).strip(" .")
    stem = stem[:80]
    if not stem:
        stem = "artifact-" + hashlib.sha1(
            (base_name or "").encode("utf-8", "ignore")
        ).hexdigest()[:8]
    return stem


def write_markdown_artifact(
    *,
    source_answer: str,
    base_name: str,
    root: Optional[Path] = None,
    extension: str = ".md",
) -> Artifact:
    if extension not in ALLOWED_EXTENSIONS:
        raise ArtifactError(f"unsupported extension: {extension}")

    payload = (source_answer or "").encode("utf-8")
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise ArtifactError(
            f"artifact too large: {len(payload)} bytes > {MAX_ARTIFACT_BYTES}"
        )

    base_root = (root or artifact_root()).resolve()
    send_dir = (base_root / "send").resolve()
    send_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = _sanitize_base_name(base_name)
    name = f"{safe_stem}{extension}"
    target = (send_dir / name).resolve()

    if not target.is_relative_to(send_dir):
        raise ArtifactError("artifact path escapes controlled root")

    # 若同名冲突，追加短哈希，仍在受控目录内。
    if target.exists():
        suffix = hashlib.sha1(payload).hexdigest()[:6]
        name = f"{safe_stem}-{suffix}{extension}"
        target = (send_dir / name).resolve()
        if not target.is_relative_to(send_dir):
            raise ArtifactError("artifact path escapes controlled root")

    target.write_bytes(payload)

    return Artifact(
        path=str(target),
        name=name,
        mime="text/markdown",
        size=len(payload),
    )
