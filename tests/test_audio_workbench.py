from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from funasr_lite.core import TranscribeResult
from funasr_lite.llm_client import LLMSettings, OpenAICompatibleClient
from funasr_lite.pipeline_workflow import run_audio_workflow
from funasr_lite.prompting import clean_transcript_text


class PromptingTests(unittest.TestCase):
    """覆盖 Prompting 层最基本的文本清洗回归。"""

    def test_clean_transcript_text_removes_control_lines(self) -> None:
        raw = "# ASR Transcript: demo\n<|zh|>\n第一句。\n- Generated: now\n\n第二句。\n"
        self.assertEqual(clean_transcript_text(raw), "第一句。\n第二句。\n")


class LLMClientTests(unittest.TestCase):
    """覆盖 Pro 版 LLM 客户端的核心提取能力。"""

    def test_resolved_api_key_prefers_explicit_value(self) -> None:
        settings = LLMSettings(api_key="abc", api_key_env="SHOULD_NOT_USE")
        self.assertEqual(settings.resolved_api_key(), "abc")

    def test_extract_content_supports_segment_list(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "第一段"},
                            {"type": "text", "text": "第二段"},
                        ]
                    }
                }
            ]
        }
        self.assertEqual(OpenAICompatibleClient._extract_content(payload), "第一段\n第二段")


class AudioWorkbenchFlowTests(unittest.TestCase):
    """用 mock 验证 Pro 版工作流在不跑真实模型时也能产出完整文件。"""

    def test_run_audio_workflow_falls_back_when_llm_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="funasr_workbench_test_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_media = temp_dir / "sample.mp4"
            source_media.write_bytes(b"fake-media")
            output_root = temp_dir / "runs"

            def fake_resolve_executable(name: str, **kwargs) -> str:
                return str(temp_dir / f"{name}.exe")

            def fake_run_command(cmd: list[str], *, cwd=None, capture_output=True):
                if "ffmpeg" in Path(cmd[0]).name.lower():
                    Path(cmd[-1]).write_bytes(b"fake-wav")
                    return subprocess.CompletedProcess(cmd, 0, "", "")
                self.fail(f"Unexpected command: {cmd}")

            fake_result = TranscribeResult(
                text="这是转写正文。需要后续整理。",
                segments=[],
                meta={"mock": True},
            )

            with patch("funasr_lite.pipeline_workflow.resolve_executable", side_effect=fake_resolve_executable):
                with patch("funasr_lite.pipeline_workflow.run_command", side_effect=fake_run_command):
                    with patch("funasr_lite.pipeline_workflow.transcribe_file", return_value=fake_result):
                        with patch("funasr_lite.pipeline_workflow.OpenAICompatibleClient.chat_text", side_effect=RuntimeError("LLM down")):
                            result = run_audio_workflow(
                                uploaded_media=str(source_media),
                                output_root=str(output_root),
                                ffmpeg_path=str(temp_dir / "ffmpeg.exe"),
                                run_llm=True,
                                split_lines=False,
                            )

            final_path = Path(result.final_markdown_path)
            bundle_path = Path(result.bundle_path)
            self.assertTrue(final_path.exists())
            self.assertTrue(bundle_path.exists())

            final_text = final_path.read_text(encoding="utf-8")
            self.assertIn("未调用大模型或大模型未返回内容", final_text)
            self.assertIn("这是转写正文。需要后续整理。", final_text)
            self.assertEqual(result.meta["llm"]["fallback_to_prompt_bundle"], True)

            with ZipFile(bundle_path) as archive:
                names = set(archive.namelist())
            self.assertIn("final_summary.md", names)
            self.assertIn("summary_prompt.md", names)
            self.assertIn("run_manifest.json", names)


if __name__ == "__main__":
    unittest.main()
