# SPDX-License-Identifier: Apache-2.0
"""Ming 3 text-to-speech pipeline configuration."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from sglang_omni.config.schema import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)

PACKAGE = "sglang_omni.models.ming3_tts"


def default_stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="preprocessing",
            gpu=0,
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_concurrency=4, reference_cache_entries=256),
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="tts_engine",
            gpu=0,
            factory_path=f"{PACKAGE}.stages.create_tts_engine_executor",
            factory=FactoryArgs(
                max_seq_len=4096,
                enable_acoustic_graph=True,
                prefill_graph_token_counts=[
                    64,
                    96,
                    128,
                    160,
                    192,
                    224,
                    256,
                    288,
                    320,
                    384,
                    448,
                    512,
                ],
                prefill_graph_max_requests=32,
                enable_async_decode=True,
            ),
            engine=EngineArgs(
                mem_fraction_static=0.75,
                max_running_requests=32,
                cuda_graph_max_bs=32,
                disable_cuda_graph=False,
                enable_torch_compile=False,
            ),
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="vocoder",
            gpu=0,
            factory_path=f"{PACKAGE}.vocoder.create_vocoder_executor",
            factory=FactoryArgs(graph_latent_steps=384),
            terminal=True,
        ),
    ]


DENSE_ARCHITECTURE = "Ming3DenseTTSForConditionalGeneration"


class Ming3TTSPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "LingV3TTSForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = (DENSE_ARCHITECTURE,)
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }
    model_path: str
    entry_stage: str = "preprocessing"
    # note (rsi): the three stage processes share one GPU; MPS runs their
    # kernels concurrently instead of time-slicing the device between them.
    mps: Literal["off", "on", "auto"] = "auto"
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=default_stages)


EntryClass = Ming3TTSPipelineConfig
