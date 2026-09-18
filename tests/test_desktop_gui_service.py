from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from funasr_lite.desktop_gui.models import SingleRunRequest
from funasr_lite.desktop_gui.service import DesktopWorkflowService
from funasr_lite.pipeline_workflow import resolve_executable


class DesktopWorkflowServiceTests(unittest.TestCase):
    """覆盖桌面 GUI 服务层中不依赖 PySide6 的关键逻辑。"""

    def test_inspect_environment_returns_cards(self) -> None:
        service = DesktopWorkflowService()
        with patch("funasr_lite.pipeline_workflow.resolve_executable", return_value="/fake/ffmpeg"):
            with patch("funasr_lite.pipeline_workflow.inspect_python_runtime", return_value={"ok": True, "status": "✅ ok", "versions": {}, "errors": []}):
                snapshot = service.inspect_environment(SingleRunRequest(local_media="sample.mp4"))
        self.assertIn("环境探测", snapshot.headline)
        self.assertTrue(any(card.title == "ffmpeg" for card in snapshot.cards))

    def test_inspect_environment_flags_missing_local_media(self) -> None:
        service = DesktopWorkflowService()
        with patch("funasr_lite.pipeline_workflow.resolve_executable", return_value="/fake/ffmpeg"):
            with patch("funasr_lite.pipeline_workflow.inspect_python_runtime", return_value={"ok": True, "status": "✅ ok", "versions": {}, "errors": []}):
                snapshot = service.inspect_environment(SingleRunRequest(local_media="missing.wav"))
        self.assertEqual(snapshot.state, "failed")
        self.assertIn("本地文件不存在", snapshot.raw["input_status"])

    def test_inspect_environment_accepts_local_transcript_without_media_tools(self) -> None:
        service = DesktopWorkflowService()
        with tempfile.TemporaryDirectory(prefix="funasr_desktop_transcript_") as temp_dir_str:
            transcript_file = Path(temp_dir_str) / "meeting.txt"
            transcript_file.write_text("已有转写稿", encoding="utf-8")
            with patch("funasr_lite.pipeline_workflow.resolve_executable", side_effect=AssertionError("ffmpeg should be skipped")):
                with patch("funasr_lite.pipeline_workflow.inspect_python_runtime", side_effect=AssertionError("FunASR runtime should be skipped")):
                    snapshot = service.inspect_environment(SingleRunRequest(local_media=str(transcript_file)))
        self.assertEqual(snapshot.state, "ready")
        self.assertIn("本地转写稿可读取", snapshot.raw["input_status"])
        self.assertIn("跳过 ffmpeg", snapshot.raw["ffmpeg_status"])
        self.assertIn("跳过 FunASR", snapshot.raw["python_runtime_status"])

    def test_inspect_environment_flags_output_root_file(self) -> None:
        service = DesktopWorkflowService()
        with tempfile.TemporaryDirectory(prefix="funasr_desktop_env_") as temp_dir_str:
            output_file = Path(temp_dir_str) / "not_a_directory"
            output_file.write_text("x", encoding="utf-8")
            media_file = Path(temp_dir_str) / "sample.wav"
            media_file.write_bytes(b"fake-wav")
            with patch("funasr_lite.pipeline_workflow.resolve_executable", return_value="/fake/ffmpeg"):
                with patch("funasr_lite.pipeline_workflow.inspect_python_runtime", return_value={"ok": True, "status": "✅ ok", "versions": {}, "errors": []}):
                    snapshot = service.inspect_environment(
                        SingleRunRequest(local_media=str(media_file), output_root=str(output_file))
                    )
        self.assertEqual(snapshot.state, "failed")
        self.assertIn("不是目录", snapshot.raw["output_root_status"])

    def test_resolve_executable_rejects_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="funasr_exe_dir_") as temp_dir_str:
            with self.assertRaisesRegex(RuntimeError, "directory"):
                resolve_executable("ffmpeg", explicit_path=temp_dir_str)

    def test_promote_current_summary_backs_up_final_summary(self) -> None:
        service = DesktopWorkflowService()
        with tempfile.TemporaryDirectory(prefix="funasr_desktop_followup_") as temp_dir_str:
            run_dir = Path(temp_dir_str)
            (run_dir / "final_summary.md").write_text("# 旧总结\n", encoding="utf-8")
            (run_dir / "transcript.txt").write_text("转写", encoding="utf-8")
            (run_dir / "run_manifest.json").write_text(json.dumps({"source_value": "demo"}), encoding="utf-8")
            cache = run_dir / "_followup_cache"
            cache.mkdir()
            (cache / "summary_current.md").write_text("# 新总结\n", encoding="utf-8")

            result = service.promote_current_summary(str(run_dir))

            self.assertIn("写回", result.headline)
            self.assertEqual((run_dir / "final_summary.md").read_text(encoding="utf-8"), "# 新总结\n")
            self.assertTrue(list(run_dir.glob("final_summary.backup.*.md")))
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("promoted_from", manifest["followup"])


if __name__ == "__main__":
    unittest.main()
