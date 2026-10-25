# SPDX-License-Identifier: Apache-2.0
"""Request-local incremental AudioVAE decoding for Ming 3 TTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypedDict

import torch
from transformers.cache_utils import Cache

from sglang_omni.modules.ming_audio.vae.modeling_audio_vae import AudioVAE
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase


class FullDecode(Protocol):
    def __call__(self, payload: StagePayload) -> StagePayload:
        pass


class UpsampleState(TypedDict):
    prev_chunk: torch.Tensor | None
    history_last: torch.Tensor | None
    is_first: bool


@dataclass(kw_only=True)
class Ming3StreamState:
    pending_latents: list[torch.Tensor] = field(default_factory=list)
    past_key_values: Cache | None = None
    codec_state: tuple[
        UpsampleState | None, torch.Tensor | None, torch.Tensor | None
    ] = (
        None,
        None,
        None,
    )
    expected_chunk_id: int = 0
    emitted_samples: int = 0


class Ming3StreamingVocoder(StreamingVocoderBase[Ming3StreamState, None]):
    def __init__(
        self,
        compute: FullDecode,
        *,
        vae: AudioVAE,
        sample_rate: int,
        stream_chunk_steps: int,
    ) -> None:
        if stream_chunk_steps < 1:
            raise ValueError("stream_chunk_steps must be positive")
        else:
            pass
        self.vae = vae
        self.stream_chunk_steps = stream_chunk_steps
        super().__init__(
            compute,
            sample_rate=sample_rate,
            stream_source_hint="Ming-3.0-TTS",
            stream_input_modality="audio_latents",
        )

    def create_stream_state(self, request_id: str) -> Ming3StreamState:
        return Ming3StreamState()

    def ingest_stream_item(
        self, request_id: str, chunk: StreamItem
    ) -> Ming3StreamState | None:
        state = self.get_or_create_stream_state(request_id)
        if state is None:
            return None
        elif chunk.chunk_id != state.expected_chunk_id:
            raise ValueError(f"Out-of-order Ming 3 latent chunk for {request_id}")
        else:
            super().ingest_stream_item(request_id, chunk)
            state.expected_chunk_id += 1
            return state

    def validate_chunk(
        self, request_id: str, state: Ming3StreamState, latents: torch.Tensor
    ) -> torch.Tensor:
        expected_shape = (1, self.vae.decoder.latent_dim)
        if (
            tuple(latents.shape) != expected_shape
            or latents.dtype != torch.float32
            or latents.device.type != "cpu"
        ):
            raise ValueError(
                f"Ming 3 latent chunks must be CPU float32 with shape {expected_shape}"
            )
        else:
            return latents

    def ingest(
        self, request_id: str, state: Ming3StreamState, latents: torch.Tensor
    ) -> None:
        state.pending_latents.append(latents)

    @torch.inference_mode()
    def decode_delta(
        self, request_id: str, state: Ming3StreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        # note (Zhuofeng Li): Keep a final latent to flush interpolation and ISTFT overlap without an empty decoder call.
        if not is_final and len(state.pending_latents) <= self.stream_chunk_steps:
            return None
        elif not state.pending_latents:
            raise RuntimeError(f"Ming 3 stream {request_id} ended without latents")
        else:
            pass
        step_count = len(state.pending_latents) if is_final else self.stream_chunk_steps
        parameter = next(self.vae.parameters())
        latents = torch.cat(state.pending_latents[:step_count], dim=0).to(
            device=parameter.device, dtype=parameter.dtype
        )
        waveform, codec_state, past_key_values = self.vae.decode(
            latents.unsqueeze(0),
            past_key_values=state.past_key_values,
            use_cache=True,
            stream_state=state.codec_state,
            last_chunk=is_final,
        )
        state.pending_latents[:step_count] = []
        state.codec_state = codec_state
        state.past_key_values = past_key_values
        waveform = waveform.detach().float().cpu().reshape(-1).clone()
        state.emitted_samples += waveform.numel()
        if waveform.numel() == 0:
            return None
        else:
            return waveform

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: Ming3StreamState
    ) -> dict[str, int | float | str | dict[str, int]]:
        if state.emitted_samples == 0:
            raise RuntimeError(f"Ming 3 stream {request_id} completed without audio")
        else:
            return {
                "modality": "audio",
                "sample_rate": self.sample_rate,
                "duration_s": state.emitted_samples / self.sample_rate,
                "usage": payload.data["usage"],
            }
