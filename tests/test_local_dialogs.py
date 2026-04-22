from __future__ import annotations

import unittest
from unittest.mock import patch

from funasr_lite.local_dialogs import choose_directory, format_runtime_status_markdown


class LocalDialogsTests(unittest.TestCase):
    """覆盖目录选择和状态格式化的基础回归。"""

    def test_choose_directory_returns_empty_on_failure(self) -> None:
        with patch("funasr_lite.local_dialogs._run_tk_directory_dialog", side_effect=RuntimeError("no gui")):
            self.assertEqual(choose_directory("C:/", "选择"), "")

    def test_choose_directory_returns_selected_value(self) -> None:
        with patch("funasr_lite.local_dialogs._run_tk_directory_dialog", return_value="D:/data"):
            self.assertEqual(choose_directory("C:/", "选择"), "D:/data")

    def test_format_runtime_status_markdown_contains_key_fields(self) -> None:
        markdown = format_runtime_status_markdown(
            profile_label="peak",
            profile_prompt="prompts/humanities_video.md",
            source_value="https://example.com",
            source_kind="url",
            resolved_output_root="D:/runs",
            resolved_download_root="D:/downloads",
            resolved_note_root="D:/notes",
            ytdlp_status="`ok`",
            ffmpeg_status="`ok`",
            cookie_status="`auto`",
            prompt_defaults=[".codex/agents.md", "prompts/humanities_video.md"],
        )
        self.assertIn("peak", markdown)
        self.assertIn("D:/runs", markdown)
        self.assertIn(".codex/agents.md", markdown)
