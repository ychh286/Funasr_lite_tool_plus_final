from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "maintain_laoxu_chronology.py"
SPEC = importlib.util.spec_from_file_location("maintain_laoxu_chronology", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MaintainLaoxuChronologyTests(unittest.TestCase):
    """验证全局编号、预留空位、冻结 manifest 和两阶段改名。"""

    def test_manifest_and_apply_preserve_global_gap(self) -> None:
        """无本地文件的历史视频应占据序号，已有文件按同一 manifest 改名。"""

        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            video_dir = root / "videos"
            report_dir = root / "reports"
            video_dir.mkdir()
            report_dir.mkdir()
            timeline = root / "timeline.tsv"
            timeline.write_text(
                "1\tCCCCCCCCCCC\tlatest\n"
                "2\tBBBBBBBBBBB\tmiddle\n"
                "3\tXXXXXXXXXXX\treserved\n"
                "4\tAAAAAAAAAAA\toldest\n",
                encoding="utf-8",
            )
            (video_dir / "fallback - 最早视频 [AAAAAAAAAAA].mp4").write_bytes(b"a")
            (video_dir / "20260101 - 中间视频 [BBBBBBBBBBB].mp4").write_bytes(b"bb")
            (video_dir / "001 - 最新视频 [CCCCCCCCCCC].mp4").write_bytes(b"ccc")
            (report_dir / "01 - 最早报告 [AAAAAAAAAAA].md").write_text("a", encoding="utf-8")
            (report_dir / "02 - 中间报告 [BBBBBBBBBBB].md").write_text("b", encoding="utf-8")

            manifest = MODULE.build_manifest(
                timeline_path=timeline,
                video_dir=video_dir,
                report_dir=report_dir,
                author="test",
            )

            self.assertEqual(manifest["summary"]["channel_total"], 4)
            self.assertEqual(manifest["summary"]["reserved_without_local_video"], 1)
            self.assertEqual(manifest["reserved"][0]["sequence"], 2)
            targets = {
                (item["kind"], item["video_id"]): Path(item["target"]).name
                for item in manifest["items"]
            }
            self.assertEqual(
                targets[("report", "AAAAAAAAAAA")],
                "001 - 最早报告 [AAAAAAAAAAA].md",
            )
            self.assertIn("003 -", targets[("video", "BBBBBBBBBBB")])
            self.assertIn("004 -", targets[("video", "CCCCCCCCCCC")])

            receipt = root / "receipt.json"
            MODULE.apply_manifest(manifest, receipt)

            self.assertTrue(receipt.is_file())
            self.assertTrue((video_dir / "001 - 最早视频 [AAAAAAAAAAA].mp4").is_file())
            self.assertTrue((video_dir / "003 - 中间视频 [BBBBBBBBBBB].mp4").is_file())
            self.assertTrue((video_dir / "004 - 最新视频 [CCCCCCCCCCC].mp4").is_file())
            self.assertTrue((report_dir / "001 - 最早报告 [AAAAAAAAAAA].md").is_file())
            self.assertTrue((report_dir / "003 - 中间报告 [BBBBBBBBBBB].md").is_file())

    def test_dry_run_only_renders_frozen_operations(self) -> None:
        """dry-run 应读取既有计划并输出完整路径，不改变任何文件。"""

        manifest = {
            "manifest_id": "example",
            "items": [
                {
                    "kind": "video",
                    "video_id": "AAAAAAAAAAA",
                    "source": "E:/old.mp4",
                    "target": "E:/001 - old [AAAAAAAAAAA].mp4",
                    "action": "rename",
                },
                {
                    "kind": "report",
                    "video_id": "BBBBBBBBBBB",
                    "source": "E:/002 - ok [BBBBBBBBBBB].md",
                    "target": "E:/002 - ok [BBBBBBBBBBB].md",
                    "action": "unchanged",
                },
            ],
        }

        rendered = MODULE.render_dry_run(manifest)

        self.assertIn("PLANNED_RENAMES=1", rendered)
        self.assertIn("E:/old.mp4", rendered)
        self.assertNotIn("002 - ok", rendered)


if __name__ == "__main__":
    unittest.main()
