# SPDX-License-Identifier: Apache-2.0
"""Pipeline registration and backbone configuration contracts."""

import pytest
import torch

import sglang_omni.models.ming3_tts.tiny_model_runner as ming3_runner
from sglang_omni.models.ming3_tts.config import Ming3TTSPipelineConfig
from sglang_omni.models.ming3_tts.sglang_model import Ming3TTSConfig
from sglang_omni.models.ming3_tts.vocoder_graph import VocoderGraphRunner
from sglang_omni.models.model_capabilities import get_model_capabilities
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.modules.ming_audio.vae.modeling_audio_vae import AudioVAE

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


class LinearVocoderDecoder(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 1, bias=False, device="cuda")
        self.latent_dim = 4
        with torch.no_grad():
            self.weight.fill_(1)


class LinearAudioVAE(AudioVAE):
    """Use a small decoder to exercise real graph storage without checkpoints."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.decoder = LinearVocoderDecoder()

    def decode(self, latents: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        return self.decoder(latents).squeeze(-1), None, None


@pytest.mark.accelerator
def test_vocoder_replay_owns_results_and_falls_back_for_uncaptured_lengths() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs require a CUDA device")
    else:
        pass
    runner = VocoderGraphRunner(LinearAudioVAE(), maximum_decode_steps=2)
    retained = runner.decode(torch.ones(1, 2, 4, device="cuda"))
    subsequent = runner.decode(torch.zeros(1, 2, 4, device="cuda"))
    torch.testing.assert_close(retained, torch.full((1, 2), 4.0, device="cuda"))
    torch.testing.assert_close(subsequent, torch.zeros(1, 2, device="cuda"))
    fallback = runner.decode(torch.ones(1, 3, 4, device="cuda"))
    runner.decode(torch.zeros(1, 1, 4, device="cuda"))
    torch.testing.assert_close(fallback, torch.full((1, 3), 4.0, device="cuda"))


@pytest.mark.accelerator
def test_kda_startup_warms_both_branches_without_advancing_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold multi-request prefill compiled a second KDA branch during serving."""
    if not torch.cuda.is_available() or ming3_runner.chunk_kda is None:
        pytest.skip("KDA warmup requires CUDA and Triton")
    else:
        pass
    native_prefill = ming3_runner.chunk_kda
    token_counts: list[int] = []

    def observe_prefill(
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        initial_state_indices: torch.Tensor,
        use_qk_l2norm_in_kernel: bool,
        cu_seqlens: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float | None,
        beta_is_raw: bool,
    ) -> torch.Tensor:
        assert A_log.dtype == dt_bias.dtype == torch.float32
        token_counts.append(q.shape[1])
        output = native_prefill(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            beta_is_raw=beta_is_raw,
        )
        assert torch.isfinite(output).all().item()
        return output

    monkeypatch.setattr(ming3_runner, "chunk_kda", observe_prefill)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        ming3_runner.warmup_kda_prefill(
            head_count=16,
            key_channel_count=128,
            value_channel_count=128,
            lower_bound=-5.0,
            activation_dtype=torch.bfloat16,
            state_dtype=torch.float32,
            device=torch.device("cuda"),
        )
    finally:
        torch.set_default_dtype(default_dtype)
    assert token_counts == [64, 1088]
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
