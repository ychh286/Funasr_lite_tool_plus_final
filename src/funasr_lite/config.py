from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


DEFAULT_MODEL_ID = "iic/SenseVoiceSmall"
DEFAULT_SPEAKER_ASR_MODEL_ID = (
    "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)
DEFAULT_SPEAKER_PUNC_MODEL_ID = "ct-punc-c"
DEFAULT_SPEAKER_SPK_MODEL_ID = "cam++"

SPEAKER_COMPATIBLE_MODEL_IDS = {
    DEFAULT_SPEAKER_ASR_MODEL_ID,
    "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "paraformer",
    "paraformer-zh",
}


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for selecting and initializing FunASR models."""

    model_id: str = DEFAULT_MODEL_ID
    vad_model_id: str = "fsmn-vad"
    punc_model_id: Optional[str] = None
    spk_model_id: Optional[str] = None
    device: Optional[str] = None
    disable_update: bool = True


@dataclass(frozen=True)
class DecodeConfig:
    """Configuration that affects inference / decoding behavior."""

    language: str = "auto"
    use_itn: bool = True
    batch_size_s: int = 10
    merge_vad: bool = True
    merge_length_s: int = 15
    max_single_segment_ms: int = 30000
    speaker_mode: str = "punc_segment"
    preset_spk_num: Optional[int] = None


@dataclass(frozen=True)
class OutputConfig:
    """How to format and save outputs."""

    fmt: str = "md"
    split_lines: bool = False


@dataclass(frozen=True)
class ObsConfig:
    """Observability and logging toggles."""

    verbose: bool = False


@dataclass(frozen=True)
class ScenarioPreset:
    name: str
    decode: DecodeConfig


MODEL_PRESETS: Dict[str, str] = {
    "iic/SenseVoiceSmall": "iic/SenseVoiceSmall",
    "iic/SenseVoiceLarge": "iic/SenseVoiceLarge",
    "paraformer-zh (online)": "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "paraformer-zh (offline)": "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "paraformer-zh (speaker diarization)": DEFAULT_SPEAKER_ASR_MODEL_ID,
    "whisper-large-v3 (HF)": "openai/whisper-large-v3",
}


VAD_PRESETS: Dict[str, str] = {
    "fsmn-vad": "fsmn-vad",
}


PUNC_PRESETS: Dict[str, Optional[str]] = {
    "none": None,
    "ct-punc-c": DEFAULT_SPEAKER_PUNC_MODEL_ID,
    "ct-punc zh (onnx, realtime)": "damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx",
}


SPK_PRESETS: Dict[str, Optional[str]] = {
    "none": None,
    "cam++": DEFAULT_SPEAKER_SPK_MODEL_ID,
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


DEFAULT_SCENARIO = "中文会议/普通录音"


def speaker_diarization_enabled(model_cfg: ModelConfig) -> bool:
    return bool(model_cfg.spk_model_id)


def apply_speaker_diarization_defaults(
    model_cfg: ModelConfig,
    *,
    allow_auto_model_swap: bool,
) -> ModelConfig:
    model_id = model_cfg.model_id
    if allow_auto_model_swap and model_id not in SPEAKER_COMPATIBLE_MODEL_IDS:
        model_id = DEFAULT_SPEAKER_ASR_MODEL_ID

    return ModelConfig(
        model_id=model_id,
        vad_model_id=model_cfg.vad_model_id,
        punc_model_id=model_cfg.punc_model_id or DEFAULT_SPEAKER_PUNC_MODEL_ID,
        spk_model_id=model_cfg.spk_model_id or DEFAULT_SPEAKER_SPK_MODEL_ID,
        device=model_cfg.device,
        disable_update=model_cfg.disable_update,
    )


def validate_speaker_diarization_config(model_cfg: ModelConfig) -> None:
    if model_cfg.model_id not in SPEAKER_COMPATIBLE_MODEL_IDS:
        supported = ", ".join(sorted(SPEAKER_COMPATIBLE_MODEL_IDS))
        raise ValueError(
            "Speaker diarization requires a timestamp-capable Paraformer model. "
            f"Current model: {model_cfg.model_id}. Supported values: {supported}."
        )
    if not model_cfg.punc_model_id:
        raise ValueError("Speaker diarization requires a punctuation model.")
    if not model_cfg.spk_model_id:
        raise ValueError("Speaker diarization requires a speaker model such as cam++.")
