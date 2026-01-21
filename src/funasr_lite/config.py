from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict


# -----------------------------
# Core configs
# -----------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for selecting and initializing FunASR models."""

    # ASR model id (ModelScope/HuggingFace id depending on your FunASR setup)
    model_id: str = "iic/SenseVoiceSmall"

    # Component model ids
    vad_model_id: str = "fsmn-vad"
    punc_model_id: Optional[str] = None

    # Runtime
    device: Optional[str] = None  # None -> auto (cuda if available else cpu)
    disable_update: bool = True   # keep reproducible behavior if supported


@dataclass(frozen=True)
class DecodeConfig:
    """Configuration that affects inference / decoding behavior."""

    language: str = "auto"
    use_itn: bool = True

    # For streaming-style decoding, FunASR often uses "seconds" to size internal batches.
    # Larger -> potentially higher throughput, but not necessarily higher peak VRAM.
    batch_size_s: int = 10

    # VAD/segment merging behavior
    merge_vad: bool = True
    merge_length_s: int = 15

    # VAD upper bound per segment
    max_single_segment_ms: int = 30000


@dataclass(frozen=True)
class OutputConfig:
    """How to format and save outputs."""

    fmt: str = "md"  # 'md' or 'txt'
    split_lines: bool = False  # experimental punctuation-based line splitting


@dataclass(frozen=True)
class ObsConfig:
    """Observability and logging toggles."""

    verbose: bool = False


# -----------------------------
# Presets (models / scenarios)
# -----------------------------


@dataclass(frozen=True)
class ScenarioPreset:
    name: str
    decode: DecodeConfig


# Notes:
# - Model ids below are "typical" ids seen in ModelScope/HF ecosystems.
# - Your local FunASR install decides which registries are usable.
# - If a preset does not exist in your backend, use the "Custom model id" field.
MODEL_PRESETS: Dict[str, str] = {
    "iic/SenseVoiceSmall": "iic/SenseVoiceSmall",
    "iic/SenseVoiceLarge": "iic/SenseVoiceLarge",
    "paraformer-zh (online)": "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "paraformer-zh (offline)": "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "whisper-large-v3 (HF)": "openai/whisper-large-v3",
}


VAD_PRESETS: Dict[str, str] = {
    "fsmn-vad": "fsmn-vad",
}


PUNC_PRESETS: Dict[str, Optional[str]] = {
    "none": None,
    # A commonly used ModelScope punctuation model id (may vary by FunASR version).
    "ct-punc zh (onnx, realtime)": "damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx",
}


SCENARIO_PRESETS: Dict[str, ScenarioPreset] = {
    "中文会议/普通录音": ScenarioPreset(
        name="中文会议/普通录音",
        decode=DecodeConfig(
            language="auto",
            use_itn=True,
            batch_size_s=10,
            merge_vad=True,
            merge_length_s=15,
            max_single_segment_ms=30000,
        ),
    ),
    "英文视频/混合语种": ScenarioPreset(
        name="英文视频/混合语种",
        decode=DecodeConfig(
            language="auto",
            use_itn=True,
            batch_size_s=20,
            merge_vad=True,
            merge_length_s=20,
            max_single_segment_ms=60000,
        ),
    ),
    "噪声环境/远场": ScenarioPreset(
        name="噪声环境/远场",
        decode=DecodeConfig(
            language="auto",
            use_itn=True,
            batch_size_s=8,
            merge_vad=True,
            merge_length_s=10,
            max_single_segment_ms=20000,
        ),
    ),
}


DEFAULT_SCENARIO: str = "中文会议/普通录音"
