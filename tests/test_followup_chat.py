from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from funasr_lite.brave_search import BraveSearchSettings
from funasr_lite.followup_chat import (
    current_summary_path,
    load_followup_session,
    new_session_id,
    run_followup_turn,
)
from funasr_lite.llm_client import LLMSettings


class FollowupChatTests(unittest.TestCase):
    """覆盖追问会话的基础回归。"""

    def _prepare_run_dir(self, temp_dir: Path) -> Path:
        run_dir = temp_dir / "20260422_180000"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "final_summary.md").write_text("# 初始总结\n\n这是第一轮总结。", encoding="utf-8")
        (run_dir / "transcript.txt").write_text("这是原始转写。", encoding="utf-8")
        (run_dir / "run_manifest.json").write_text(
            json.dumps({"source_value": "测试视频", "source_media_path": str(run_dir / "sample.mp4")}, ensure_ascii=False),
            encoding="utf-8",
        )
        return run_dir

    def test_load_followup_session_uses_final_summary_by_default(self) -> None:
        with tempfile.TemporaryDirectory(prefix="followup_session_") as temp_dir_str:
            run_dir = self._prepare_run_dir(Path(temp_dir_str))
            session = load_followup_session(str(run_dir), "main")
            self.assertIn("这是第一轮总结", session["current_summary_text"])
            self.assertIn("这是原始转写", session["transcript_text"])

    def test_run_followup_turn_updates_summary_and_history(self) -> None:
        with tempfile.TemporaryDirectory(prefix="followup_turn_") as temp_dir_str:
            run_dir = self._prepare_run_dir(Path(temp_dir_str))
            with patch(
                "funasr_lite.followup_chat.OpenAICompatibleClient.chat_text",
                return_value="<<<ANSWER>>>\n这是追问回答。\n<<<UPDATED_SUMMARY>>>\n# 新总结\n\n这是更新后的总结。",
            ):
                result = run_followup_turn(
                    run_reference=str(run_dir),
                    session_id=new_session_id(),
                    question="请重写总结。",
                    llm_settings=LLMSettings(enabled=True),
                    update_summary=True,
                    enable_web_search=False,
                    brave_settings=BraveSearchSettings(enabled=False),
                )

            self.assertIn("这是追问回答", result.answer_markdown)
            self.assertIn("更新后的总结", result.current_summary_text)
            self.assertTrue(result.current_summary_path.exists())
            self.assertTrue((run_dir / "_followup_cache").exists())
            self.assertEqual(len(result.chat_history_pairs), 1)

    def test_run_followup_turn_falls_back_when_llm_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="followup_fallback_") as temp_dir_str:
            run_dir = self._prepare_run_dir(Path(temp_dir_str))
            with patch(
                "funasr_lite.followup_chat.OpenAICompatibleClient.chat_text",
                side_effect=RuntimeError("LLM unavailable"),
            ):
                result = run_followup_turn(
                    run_reference=str(run_dir),
                    session_id="main",
                    question="继续追问",
                    llm_settings=LLMSettings(enabled=True),
                    update_summary=True,
                    enable_web_search=False,
                    brave_settings=BraveSearchSettings(enabled=False),
                )

            self.assertIn("追问失败", result.answer_markdown)
            self.assertEqual(current_summary_path(run_dir), run_dir / "final_summary.md")
            self.assertEqual(len(result.chat_history_pairs), 1)
