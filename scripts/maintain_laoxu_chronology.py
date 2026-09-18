#!/usr/bin/env python3
"""维护“老徐常识看世界”视频与报告的全局时间顺序编号。

本模块只负责文件名展示层，不下载媒体、不修改文件内容。频道时间轴中的 YouTube
视频 ID 是唯一关联键；最早发布的视频固定为 001，暂时无权访问的视频也保留其
全局序号。工作流分为 freeze、dry-run、apply 三个独立阶段，正式执行只能读取已经
冻结的 manifest，避免确认前后重新扫描目录而改变目标集合。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


VIDEO_ID_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]$")
LEADING_PREFIX_RE = re.compile(r"^\s*(?:\d{1,8}|fallback)\s*-\s*", re.IGNORECASE)
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
CHINA_TZ = timezone(timedelta(hours=8))
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TimelineEntry:
    """表示频道完整列表中的一个稳定位置。

    参数：newest_position 为网页端从新到旧的序号，video_id 为 YouTube ID，
    sequence 为换算后的从旧到新全局序号。
    状态影响：无；仅用于生成冻结计划。
    """

    newest_position: int
    video_id: str
    sequence: int


def _now_text() -> str:
    """返回北京时间，供 manifest、预览和执行收据记录审计时间。"""

    return datetime.now(CHINA_TZ).strftime("%Y-%m-%d %H:%M:%S %z")


def _configure_stdio() -> None:
    """在 Windows 宿主仍使用 GBK 时强制 UTF-8，保证完整文件名可进入审计输出。"""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def _sha256_file(path: Path) -> str:
    """计算文件 SHA-256，用于证明 apply 使用的时间轴快照未发生变化。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _path_key(path: Path) -> str:
    """生成适合 Windows 路径冲突检测的规范化键，不访问或修改文件。"""

    return os.path.normcase(os.path.abspath(path))


def parse_timeline(path: Path) -> list[TimelineEntry]:
    """读取 yt-dlp 的频道倒序快照并换算为从旧到新的全局序号。

    参数：path 为制表符分隔文件，前两列必须是网页序号和 11 位视频 ID。
    返回值：按全局序号升序排列的 TimelineEntry 列表。
    状态影响：只读；标题列即使存在也不参与命名，避免编码污染扩散。
    调用时机：仅在 freeze 阶段构建 manifest 时调用。
    """

    rows: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = raw_line.split("\t", 2)
        if len(fields) < 2:
            raise ValueError(f"时间轴第 {line_number} 行缺少制表符字段")
        try:
            newest_position = int(fields[0])
        except ValueError as exc:
            raise ValueError(f"时间轴第 {line_number} 行序号无效：{fields[0]!r}") from exc
        video_id = fields[1].strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError(f"时间轴第 {line_number} 行视频 ID 无效：{video_id!r}")
        rows.append((newest_position, video_id))

    if not rows:
        raise ValueError("时间轴为空")
    positions = [position for position, _ in rows]
    ids = [video_id for _, video_id in rows]
    expected_positions = list(range(1, len(rows) + 1))
    if sorted(positions) != expected_positions:
        raise ValueError("时间轴网页序号必须从 1 连续到总数")
    if len(set(ids)) != len(ids):
        raise ValueError("时间轴包含重复视频 ID")

    total = len(rows)
    entries = [
        TimelineEntry(
            newest_position=position,
            video_id=video_id,
            sequence=total - position + 1,
        )
        for position, video_id in rows
    ]
    return sorted(entries, key=lambda item: item.sequence)


def extract_video_id(path: Path) -> str:
    """从文件名末尾的 ``[video_id]`` 提取关联键，无法识别时立即拒绝。"""

    match = VIDEO_ID_RE.search(path.stem)
    if not match:
        raise ValueError(f"文件名缺少末尾视频 ID：{path}")
    return match.group(1)


def build_target_name(path: Path, sequence: int, width: int) -> str:
    """保留现有标题与扩展名，仅把混乱前缀替换为全局时间序号。

    参数：path 为当前文件，sequence 为最早到最新的全局序号，width 为补零宽度。
    返回值：同目录目标文件名。
    状态影响：无。
    调用时机：freeze 阶段为视频和 Markdown 报告生成统一名称。
    """

    title_with_id = LEADING_PREFIX_RE.sub("", path.stem, count=1).strip()
    if not title_with_id:
        raise ValueError(f"去除旧前缀后文件名为空：{path}")
    return f"{sequence:0{width}d} - {title_with_id}{path.suffix}"


def _collect_files(directory: Path, suffixes: set[str]) -> list[Path]:
    """读取一个明确目录中的目标文件；不递归，避免扩大批量操作范围。"""

    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在：{directory}")
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes),
        key=lambda path: path.name.casefold(),
    )


def _build_items(
    *,
    paths: Iterable[Path],
    kind: str,
    timeline_by_id: dict[str, TimelineEntry],
    width: int,
) -> list[dict[str, Any]]:
    """把一组本地文件映射为含源路径、目标路径和状态指纹的 manifest 条目。"""

    seen_ids: set[str] = set()
    items: list[dict[str, Any]] = []
    for path in paths:
        video_id = extract_video_id(path)
        if video_id in seen_ids:
            raise ValueError(f"{kind} 目录存在重复视频 ID：{video_id}")
        seen_ids.add(video_id)
        entry = timeline_by_id.get(video_id)
        if entry is None:
            raise ValueError(f"{kind} 文件不在频道时间轴中：{path}")
        target = path.with_name(build_target_name(path, entry.sequence, width))
        stat = path.stat()
        items.append(
            {
                "kind": kind,
                "video_id": video_id,
                "sequence": entry.sequence,
                "source": str(path.resolve()),
                "target": str(target.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "action": "unchanged" if _path_key(path) == _path_key(target) else "rename",
            }
        )
    return sorted(items, key=lambda item: (item["sequence"], item["kind"]))


def _validate_plan_collisions(items: list[dict[str, Any]]) -> None:
    """在冻结前拒绝重复目标和会覆盖非目标文件的计划。"""

    target_keys: dict[str, str] = {}
    source_keys = {_path_key(Path(item["source"])) for item in items}
    for item in items:
        target = Path(item["target"])
        target_key = _path_key(target)
        previous = target_keys.get(target_key)
        if previous is not None and previous != item["source"]:
            raise ValueError(f"多个源文件指向同一目标：{previous}；{item['source']} -> {target}")
        target_keys[target_key] = item["source"]
        if target.exists() and target_key not in source_keys:
            raise FileExistsError(f"目标已被计划外文件占用：{target}")


def build_manifest(
    *, timeline_path: Path, video_dir: Path, report_dir: Path, author: str
) -> dict[str, Any]:
    """扫描明确目录并生成冻结 manifest 数据，不执行任何重命名。

    参数：时间轴快照、视频目录、报告目录和作者标识。
    返回值：可序列化的完整 manifest。
    状态影响：只读目录和文件元数据。
    调用时机：freeze 模式写入 manifest 之前。
    """

    timeline = parse_timeline(timeline_path)
    timeline_by_id = {entry.video_id: entry for entry in timeline}
    width = max(3, len(str(len(timeline))))
    video_paths = _collect_files(video_dir, VIDEO_SUFFIXES)
    report_paths = _collect_files(report_dir, {".md"})
    video_items = _build_items(
        paths=video_paths,
        kind="video",
        timeline_by_id=timeline_by_id,
        width=width,
    )
    report_items = _build_items(
        paths=report_paths,
        kind="report",
        timeline_by_id=timeline_by_id,
        width=width,
    )
    video_ids = {item["video_id"] for item in video_items}
    report_ids = {item["video_id"] for item in report_items}
    if not report_ids.issubset(video_ids):
        missing = sorted(report_ids - video_ids)
        raise ValueError(f"存在报告但没有本地视频：{missing}")

    items = video_items + report_items
    _validate_plan_collisions(items)
    reserved = [
        {
            "sequence": entry.sequence,
            "newest_position": entry.newest_position,
            "video_id": entry.video_id,
        }
        for entry in timeline
        if entry.video_id not in video_ids
    ]
    summary = {
        "channel_total": len(timeline),
        "local_videos": len(video_items),
        "local_reports": len(report_items),
        "reserved_without_local_video": len(reserved),
        "rename_operations": sum(item["action"] == "rename" for item in items),
        "unchanged_items": sum(item["action"] == "unchanged" for item in items),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created": _now_text(),
        "author": author,
        "policy": "频道完整时间轴从旧到新编号；最早=001；未下载视频保留全局序号",
        "timeline_path": str(timeline_path.resolve()),
        "timeline_sha256": _sha256_file(timeline_path),
        "video_dir": str(video_dir.resolve()),
        "report_dir": str(report_dir.resolve()),
        "summary": summary,
        "timeline": [entry.__dict__ for entry in timeline],
        "reserved": reserved,
        "items": sorted(items, key=lambda item: (item["sequence"], item["kind"])),
    }
    identity_source = json.dumps(
        {
            "timeline_sha256": manifest["timeline_sha256"],
            "items": manifest["items"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest["manifest_id"] = hashlib.sha256(identity_source).hexdigest()[:16]
    return manifest


def render_preview_markdown(manifest: dict[str, Any]) -> str:
    """生成便于人工审核的完整映射表，不承担执行职责。"""

    summary = manifest["summary"]
    lines = [
        f"更新时间：{manifest['created']}（{manifest['author']}）",
        "",
        "# 老徐常识看世界｜全局时间顺序重命名预览",
        "",
        "## 规则与数量",
        "",
        f"- 规则：{manifest['policy']}",
        f"- Manifest ID：`{manifest['manifest_id']}`",
        f"- 频道总数：{summary['channel_total']}",
        f"- 本地视频：{summary['local_videos']}",
        f"- 本地报告：{summary['local_reports']}",
        f"- 预留空位：{summary['reserved_without_local_video']}",
        f"- 计划重命名：{summary['rename_operations']}",
        f"- 已符合规则：{summary['unchanged_items']}",
        "",
        "## 文件映射",
        "",
        "| 类型 | 全局序号 | 视频ID | 动作 | 当前文件名 | 目标文件名 |",
        "|---|---:|---|---|---|---|",
    ]
    for item in manifest["items"]:
        source_name = Path(item["source"]).name.replace("|", "\\|")
        target_name = Path(item["target"]).name.replace("|", "\\|")
        lines.append(
            f"| {item['kind']} | {item['sequence']:03d} | `{item['video_id']}` | "
            f"{item['action']} | {source_name} | {target_name} |"
        )
    lines.extend(
        [
            "",
            "## 当前无本地视频的预留序号",
            "",
            "| 全局序号 | 视频ID | 网页倒序位置 |",
            "|---:|---|---:|",
        ]
    )
    for item in manifest["reserved"]:
        lines.append(
            f"| {item['sequence']:03d} | `{item['video_id']}` | {item['newest_position']} |"
        )
    lines.append("")
    return "\n".join(lines)


def freeze_manifest(manifest: dict[str, Any], manifest_path: Path, preview_path: Path) -> None:
    """一次性写入固定 manifest 与人类可读预览；已存在时拒绝覆盖。"""

    if manifest_path.exists() or preview_path.exists():
        raise FileExistsError(f"冻结输出已存在，拒绝覆盖：{manifest_path} 或 {preview_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with preview_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(render_preview_markdown(manifest))


def load_manifest(path: Path) -> dict[str, Any]:
    """读取并校验冻结 manifest 的基础结构，供 dry-run 和 apply 共用。"""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"不支持的 manifest 版本：{manifest.get('schema_version')}")
    if not manifest.get("manifest_id") or not isinstance(manifest.get("items"), list):
        raise ValueError("manifest 缺少 manifest_id 或 items")
    return manifest


def render_dry_run(manifest: dict[str, Any]) -> str:
    """列出同一 manifest 中每个待处理文件的完整路径和总数量，不写文件系统。"""

    operations = [item for item in manifest["items"] if item["action"] == "rename"]
    lines = [
        f"MANIFEST_ID={manifest['manifest_id']}",
        f"PLANNED_RENAMES={len(operations)}",
    ]
    for index, item in enumerate(operations, 1):
        lines.append(
            f"{index:03d}|{item['kind']}|{item['video_id']}|RENAME|"
            f"{item['source']}|{item['target']}"
        )
    return "\n".join(lines)


def _validate_apply_state(operations: list[dict[str, Any]]) -> None:
    """在任何改名发生前校验源指纹、目标占用和临时路径，保证失败早于变更。"""

    source_keys = {_path_key(Path(item["source"])) for item in operations}
    for item in operations:
        source = Path(item["source"])
        target = Path(item["target"])
        if not source.is_file():
            raise FileNotFoundError(f"manifest 源文件不存在：{source}")
        stat = source.stat()
        if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise RuntimeError(f"源文件在预览后发生变化：{source}")
        if target.exists() and _path_key(target) not in source_keys:
            raise FileExistsError(f"目标被 manifest 外文件占用：{target}")


def apply_manifest(manifest: dict[str, Any], receipt_path: Path) -> None:
    """基于固定 manifest 两阶段重命名，并在异常时尽力回滚。

    参数：manifest 为已由用户审核的冻结计划，receipt_path 为成功收据。
    返回值：无。
    状态影响：仅重命名 manifest 中 action=rename 的文件；不覆盖、不删除内容。
    调用时机：用户确认完整 dry-run 后，单独以 --apply 执行。
    """

    operations = [item for item in manifest["items"] if item["action"] == "rename"]
    if receipt_path.exists():
        raise FileExistsError(f"执行收据已存在，拒绝重复应用：{receipt_path}")
    _validate_apply_state(operations)
    staged: list[tuple[Path, Path, Path]] = []
    finalized: list[tuple[Path, Path, Path]] = []
    manifest_id = manifest["manifest_id"]

    # 第一阶段把所有源文件移至同目录唯一临时名，先释放可能互相占用的最终名称。
    try:
        for index, item in enumerate(operations, 1):
            source = Path(item["source"])
            target = Path(item["target"])
            temporary = source.with_name(
                f".__chronology_{manifest_id}_{index:04d}{source.suffix}"
            )
            if temporary.exists():
                raise FileExistsError(f"临时路径已存在：{temporary}")
            source.rename(temporary)
            staged.append((source, temporary, target))
    except Exception:
        for source, temporary, _ in reversed(staged):
            if temporary.exists() and not source.exists():
                temporary.rename(source)
        raise

    # 第二阶段写入最终名称；若失败，先撤回已完成目标，再统一恢复全部源名称。
    try:
        for source, temporary, target in staged:
            temporary.rename(target)
            finalized.append((source, temporary, target))
    except Exception:
        for _, temporary, target in reversed(finalized):
            if target.exists() and not temporary.exists():
                target.rename(temporary)
        for source, temporary, _ in reversed(staged):
            if temporary.exists() and not source.exists():
                temporary.rename(source)
        raise

    receipt = {
        "schema_version": 1,
        "manifest_id": manifest_id,
        "applied_at": _now_text(),
        "renamed": len(operations),
    }
    with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数；三个阶段互斥，避免预览与执行串联。"""

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="扫描并冻结 manifest，不改名")
    mode.add_argument("--dry-run", action="store_true", help="只读取 manifest 并打印预览")
    mode.add_argument("--apply", action="store_true", help="只按 manifest 正式改名")
    parser.add_argument("--manifest", type=Path, required=True, help="冻结 manifest 路径")
    parser.add_argument("--timeline", type=Path, help="freeze 阶段的频道时间轴 TSV")
    parser.add_argument("--video-dir", type=Path, help="freeze 阶段的视频目录")
    parser.add_argument("--report-dir", type=Path, help="freeze 阶段的报告目录")
    parser.add_argument("--preview-md", type=Path, help="freeze 阶段的人类可读预览")
    parser.add_argument("--author", default="codex-app-GPT5.6", help="冻结计划作者")
    parser.add_argument("--receipt", type=Path, help="apply 成功收据路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    """执行 freeze、dry-run 或 apply 单一阶段，并返回进程退出码。"""

    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.freeze:
            if not all((args.timeline, args.video_dir, args.report_dir, args.preview_md)):
                parser.error("--freeze 需要 --timeline、--video-dir、--report-dir、--preview-md")
            manifest = build_manifest(
                timeline_path=args.timeline,
                video_dir=args.video_dir,
                report_dir=args.report_dir,
                author=args.author,
            )
            freeze_manifest(manifest, args.manifest, args.preview_md)
            print(f"MANIFEST={args.manifest.resolve()}")
            print(f"PREVIEW={args.preview_md.resolve()}")
            print(json.dumps(manifest["summary"], ensure_ascii=False))
            return 0

        manifest = load_manifest(args.manifest)
        if args.dry_run:
            print(render_dry_run(manifest))
            return 0
        if args.receipt is None:
            parser.error("--apply 需要 --receipt")
        apply_manifest(manifest, args.receipt)
        print(f"APPLIED_MANIFEST={manifest['manifest_id']}")
        print(f"RECEIPT={args.receipt.resolve()}")
        return 0
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
