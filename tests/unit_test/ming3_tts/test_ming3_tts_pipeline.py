# SPDX-License-Identifier: Apache-2.0
"""Pipeline registration and backbone configuration contracts."""

import torch

from sglang_omni.models.ming3_tts.config import Ming3TTSPipelineConfig
from sglang_omni.models.ming3_tts.model_runner import (
    Ming3TTSRequestData,
    build_stream_output,
)
from sglang_omni.models.ming3_tts.sglang_model import Ming3TTSConfig
from sglang_omni.models.model_capabilities import get_model_capabilities
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.types import RequestOutput

CHECKPOINT_ARCHITECTURE = "LingV3TTSForConditionalGeneration"
HIDDEN_SIZE = 64


def test_checkpoint_architecture_resolves_to_ming3_tts_pipeline() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config(CHECKPOINT_ARCHITECTURE)
        is Ming3TTSPipelineConfig
    )
    capabilities = get_model_capabilities(CHECKPOINT_ARCHITECTURE)
    assert capabilities is not None
    assert capabilities.supports_reference_audio is True


def test_default_pipeline_ends_at_vocoder() -> None:
    config = Ming3TTSPipelineConfig(model_path="inclusionAI/Ming-3.0-tts-tiny")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]
    engine, vocoder = config.stages[1:]
    assert engine.stream_to == ["vocoder"]
    assert vocoder.can_accept_stream_before_payload


def test_stream_output_transfers_owned_latents_once() -> None:
    """Emission must not repeat a previous latent or alias mutable runner storage."""
    payload = StagePayload("speech", OmniRequest("text", params={"stream": True}), {})
    request = Ming3TTSRequestData(
        stage_payload=payload,
        speaker_embedding=torch.zeros(4),
        speaker_position=0,
        sampling_temperature=1.0,
        noise_generator=torch.Generator(),
        latent_history=torch.zeros(1, 4),
        latents=[torch.ones(1, 4)],
    )
    output = RequestOutput("speech")
    first = build_stream_output("speech", request, output)
    assert len(first) == 1
    assert first[0].target == "vocoder"
    assert first[0].metadata == {"modality": "audio_latents", "stream": True}
    request.latents[0].zero_()
    torch.testing.assert_close(first[0].data, torch.ones(1, 4))
    assert build_stream_output("speech", request, output) == []
    request.latents.append(torch.full((1, 4), 2.0))
    second = build_stream_output("speech", request, output)
    assert len(second) == 1
    torch.testing.assert_close(second[0].data, torch.full((1, 4), 2.0))
    payload.request.params["stream"] = False
    request.latents.append(torch.zeros(1, 4))
    assert build_stream_output("speech", request, output) == []


def test_backbone_config_flattens_llm_config_and_keeps_speech_sections() -> None:
    config = Ming3TTSConfig(
        llm_config={
            "hidden_size": HIDDEN_SIZE,
            "model_type": "bailing_moe_v3",
        },
        tts_config={"latent_dim": 4},
        flow_config={"depth": 1},
        aggregator_config={"depth": 1},
    )
    assert config.model_type == "lingv3_tts"
    assert config.hidden_size == HIDDEN_SIZE
    assert config.tts_config == {"latent_dim": 4}
    assert config.flow_config == {"depth": 1}
    assert config.aggregator_config == {"depth": 1}
