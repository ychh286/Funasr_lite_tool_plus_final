from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from funasr_lite.llm_client import extract_message_text, normalize_chat_completions_url
from funasr_lite.pipeline_workflow import WorkflowRunRequest, run_audio_workbench


class LLMClientTests(unittest.TestCase):
    """覆盖 LLM 响应解析的基础回归。"""

    def test_normalize_chat_completions_url(self) -> None:
        self.assertEqual(
            normalize_chat_completions_url("http://127.0.0.1:4000/v1"),
            "http://127.0.0.1:4000/v1/chat/completions",
        )
        self.assertEqual(
            normalize_chat_completions_url("http://127.0.0.1:4000/v1/chat/completions"),
            "http://127.0.0.1:4000/v1/chat/completions",
        )

    def test_extract_message_text_handles_segment_list(self) -> None:
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
        self.assertEqual(extract_message_text(payload), "第一段\n第二段")


class AudioWorkbenchFlowTests(unittest.TestCase):
    """用 mock 命令覆盖工作台主流程的关键产物输出。"""

    def test_run_audio_workbench_generates_final_summary_and_zip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="funasr_workbench_test_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_media = temp_dir / "sample.mp4"
            source_media.write_bytes(b"fake-media")
            ffmpeg_exe = temp_dir / "ffmpeg.exe"
            ffmpeg_exe.write_bytes(b"fake-ffmpeg")
            extra_prompt = temp_dir / "extra_prompt.md"
            extra_prompt.write_text("# 额外要求\n\n请重点整理行动项。", encoding="utf-8")
            output_root = temp_dir / "runs"

            request = WorkflowRunRequest(
                local_media_path=str(source_media),
                uploaded_prompt_path=str(extra_prompt),
                extra_instruction="请补充一段执行建议。",
                ffmpeg_path=str(ffmpeg_exe),
                output_root=str(output_root),
                default_output_root=str(output_root),
                timestamp="20260422_170000",
                prepared_audio_name="audio_16k_mono.wav",
            )

            def fake_run_command(cmd: list[str], *, cwd=None, capture_output=False):
                if cmd[0] == str(ffmpeg_exe):
                    Path(cmd[-1]).write_bytes(b"fake-wav")
                    return subprocess.CompletedProcess(cmd, 0, "", "")
                if len(cmd) > 1 and cmd[1].endswith("transcribe_local.py"):
                    out_index = cmd.index("--out") + 1
                    transcript_path = Path(cmd[out_index])
                    transcript_path.write_text(
                        "# ASR Transcript: sample\n\n这是转写正文。需要后续整理。\n",
                        encoding="utf-8",
                    )
                    return subprocess.CompletedProcess(cmd, 0, "", "")
                self.fail(f"Unexpected command: {cmd}")

            with patch("funasr_lite.pipeline_workflow.run_command", side_effect=fake_run_command):
                result = run_audio_workbench(request)

            self.assertTrue(result.final_summary_path.exists())
            self.assertTrue(result.workflow_zip_path.exists())
            summary_text = result.final_summary_path.read_text(encoding="utf-8")
            self.assertIn("Final Summary Draft", summary_text)
            self.assertIn("这是转写正文。需要后续整理。", summary_text)
            self.assertIn("请补充一段执行建议。", summary_text)

            with ZipFile(result.workflow_zip_path) as archive:
                names = set(archive.namelist())
            self.assertIn("final_summary.md", names)
            self.assertIn("summary_prompt.md", names)
            self.assertIn("audio_16k_mono.wav", names)
            self.assertIn("run_manifest.json", names)


if __name__ == "__main__":
    unittest.main()
