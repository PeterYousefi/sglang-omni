# RFC: Qwen3-TTS on the native C++ MLX path

Status: proposed; this document does not add a serving implementation.

## Problem and decision

[#1960](https://github.com/sgl-project/sglang-omni/pull/1960) implements Qwen3-TTS
through the former Python MLX worker inside `sglang_omni/`. That integration was
removed by [#2725](https://github.com/sgl-project/sglang-omni/pull/2725).
Reintroducing its worker, scheduler adapters or `SGLANG_USE_MLX` dispatch would
violate the [MLX path policy](../mlx/index.md) established in
[#2701](https://github.com/sgl-project/sglang-omni/pull/2701).

Port the model computation and prompt semantics to
`sglang_omni_mlx/native/src/qwen3_tts/`, with `qwen3_tts_server` and
`qwen3_tts_synthesize` executables. Serving must run in C++ with MLX tensors
between model stages, without Python, Torch, SGLang or mlx-audio at runtime.
Python remains available for checkpoint preparation and independent reference
checks. The CUDA and native MLX paths meet at `/v1/audio/speech`.

Start with a complete, non-streaming 0.6B CustomVoice consumer. Add Base voice
cloning and stateful PCM streaming as separately qualified increments. The
Torch/MPS work in #1960 is a separate proposal; it is not part of this port.
VoiceDesign, additional model sizes, quantization, batching, a playground and
Voxt application integration are outside the initial implementation.

## Sources and current integration boundary

This RFC was checked against upstream `main` at
`6b6329afb` on October 24, 2026. References below identify the inspected PR heads;
their validation reports are historical evidence, not native Qwen3-TTS results.

| Source | Inspected revision | Use |
| --- | --- | --- |
| [#1960](https://github.com/sgl-project/sglang-omni/pull/1960) | `f4ed5942633e790f59e57b3f150bbffaa21e395d` | Talker, code predictor, prompts, native MLX speech tokenizer, speaker encoder and streaming semantics. |
| [#2430](https://github.com/sgl-project/sglang-omni/pull/2430) | `08ed7b0ac26d7ac6bd5dc10a6b0f13bb602b866b` | CustomVoice HTTP behavior and real Mac measurements. Its implementation is Python and uses mlx-audio decoding; it is not the C++ runtime foundation. |
| [#2870](https://github.com/sgl-project/sglang-omni/pull/2870) | `2d528de675933dd4b8ed37f81c783fe92aab66e5` | Ming TTS service, typed synthesis jobs, PCM responses and disconnect cancellation. |
| [#2444](https://github.com/sgl-project/sglang-omni/pull/2444) | `cc662c380db67bd4fdaab2c95b8f082ddd948e69` | AuK checkpoint preparation, waveform parity, shared TTS lifecycle and API validation. |

At this base, `common/service.*`, `common/http.*`, `common/layers.*`,
`common/tokenizer.*` and `common/audio.*` already exist. The
`common/tts_service.*` worker proposed by #2444/#2870 is not yet on `main`.
Reuse that implementation after it lands. If Qwen3-TTS must land first, coordinate
the same minimal shared worker change with those authors and preserve attribution;
do not copy a private worker into `qwen3_tts/` or add a backend registry.
The implementation PR must target `main` for Mac CI, rather than stack on an
unmerged TTS branch. Existing native Voxtral TTS also supplies a synthesis CLI
and the `ci/voxtral_golden.py` seeded-code golden pattern.

## First user-visible behavior

Launch `qwen3_tts_server --model-path MODEL_DIRECTORY --port 8000` and send:

```json
{
  "model": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
  "input": "Hello, welcome to the native speech server.",
  "voice": "Ryan",
  "language": "English",
  "response_format": "wav",
  "stream": false,
  "seed": 42,
  "max_new_tokens": 256
}
```

The proposed response is mono 24 kHz signed 16-bit PCM, packaged as WAV or
returned directly for `response_format="pcm"`. `max_new_tokens` counts codec
frames, not the 16 individual codes in each frame. The checkpoint is marketed
as 12 Hz; its 1,920 output samples per frame at 24 kHz give 12.5 frames/second.

Use the shared supervisor lifecycle for standalone and supervised launch,
health reporting, model naming and shutdown. Normalize requests before worker
admission. Accept only a CustomVoice checkpoint with a supported preset voice
in the first milestone. Resolve voice and language identifiers from checkpoint
metadata, following #1960's case normalization and dialect-language behavior.
Do not silently substitute an unknown voice or language.

Support `input`, `model`, `voice`, `language`, `response_format`, `stream`,
`max_new_tokens`, `seed` and the existing Qwen3-TTS semantic/subtalker sampling
fields. Preserve the defaults and field meanings in the pinned #1960 reference;
record them in the eventual cookbook and golden requests. Reject malformed,
non-finite or out-of-range sampling values. Keep `speed=1`; reject other speeds.
Reject cloning fields, style instructions and streaming in the first milestone,
then enable only the fields qualified for each later checkpoint and milestone.

One synthesis job may run at a time, with no waiting queue: busy requests return
429 and a stopping service returns 503, matching the proposed shared TTS worker.
Invalid requests return 400 before synthesis; enforce finite request-body, input,
frame and reference-audio limits. A disconnect or shutdown cancels at computation
boundaries, releases request state and makes the worker available after cleanup.
Unexpected inference failures return an error and never fabricate audio.

## Port map and ownership

All filenames below are proposed additions; they are not existing APIs.

| #1960 component | Native location | Responsibility |
| --- | --- | --- |
| `config.py`, `weights.py`, `model.py` | `src/qwen3_tts/qwen3_tts_model.{h,cpp}` | Config validation, strict checkpoint loading and model ownership. |
| `talker.py` | `src/qwen3_tts/qwen3_tts_talker.{h,cpp}` | Talker and code predictor, attention and cache semantics. |
| `prompt.py`, frame loop in `generate.py`, `sampling.py` | `src/qwen3_tts/qwen3_tts_generator.{h,cpp}` | Concrete prompt construction, sampling, EOS and feedback. |
| `speech_tokenizer.py` decoder and `streaming_vocoder.py` computation | `src/qwen3_tts/qwen3_tts_codec.{h,cpp}` | Codebooks, waveform decoding and later request-owned streaming state. |
| `speaker_encoder.py`, `dsp.py`, speech-tokenizer encoder | `src/qwen3_tts/qwen3_tts_reference.{h,cpp}` | Later Base reference-audio preprocessing, speaker embedding and codec encoding. |
| Request normalization and HTTP response | `src/qwen3_tts/qwen3_tts_service.{h,cpp}` | Validated request values and shared synthesis-worker integration. |
| Executable entry points | `src/qwen3_tts/qwen3_tts_server.cpp`, `tools/qwen3_tts_synthesize.cpp` | Shared server startup and an HTTP-independent parity CLI. |

Do not port `runner.py`, `pending.py`, `scheduler_runner.py`, stage wiring or the
MLX worker registry. They integrate with the deleted scheduler path. Keep only
their model semantics in the native generator. Keep prompt embeddings, codec
frames and waveform intermediates in MLX; copy audio to the host at response
packaging and copy diagnostic tensors only when the parity CLI requests them.

The model owns immutable weights. Each request owns its talker KV cache, code
predictor cache, remaining text embeddings, sampling state and codec state.
The predictor cache resets for every codec frame; the talker cache persists
throughout that request. No cache or RNG state may leak to the next request.

Add a `qwen3_tts` CMake library linked to `common`; link the server to
`omni_service` and the shared TTS worker when available. Reuse checkpoint,
linear, normalization, embedding, tokenizer and audio primitives where their
semantics match. Keep Qwen3-TTS attention and prompt code in its own directory;
do not import a sibling model's internals or move model-specific code into
`common/` merely because another transformer exists.
Install both executables with the existing MLX dylib/metallib packaging rules.

## Numerical behavior that must survive the port

1. **Prompts.** Preserve the exact control-token order, text projection,
   language markers, speaker embedding positions, trailing text and padding.
   Compare prompt IDs and assembled embeddings against #1960 before testing
   sampled speech. Explicit language adds a control position; dialect voices
   can imply their language for `auto`.
2. **One complete frame.** Prefill the talker, sample semantic group 0 with
   the reference special-token suppression and repetition rules, then predict
   groups 1–15. Prime the predictor with the talker hidden state and group-0
   embedding. Sum all group embeddings and add the next text embedding for
   the next talker step. Stop on semantic EOS without sending an EOS/control
   frame to the waveform decoder; preserve the frame-cap finish reason.
3. **Attention and positions.** Preserve talker full attention, Q/K norms,
   RoPE positions and predictor reset offsets. #1960 carries a Metal RoPE
   workaround; check the same shapes on the pinned MLX build before deciding
   whether C++ needs it. CPU-only parity cannot qualify that decision.
4. **Codec window.** The speech-tokenizer decoder uses sliding causal
   attention with window 72, even when that computed configuration field is
   absent from JSON. Keys satisfy `key_position <= query_position` and
   `key_position > query_position - 72`. Include a 96-frame decode to expose
   regressions hidden by short speech. Do not copy mlx-audio's unmasked behavior.
5. **Weights.** Fold codebook sums divided by usage counts floored at `1e-5`
   during loading/preparation, covering both encoder and decoder spellings.
   Distinguish Conv1d `[out,in,kernel]` from ConvTranspose1d `[in,out,kernel]`
   explicitly when converting to MLX `[out,kernel,in]`. Equal channel counts
   make shape-only layout detection ambiguous. Record source versus prepared
   layout and never transpose an already converted artifact again.
6. **Sampling.** Keep semantic and subtalker settings distinct. Use request-owned
   MLX random keys and the reference draw order. Fixed seeds across different
   chips or Torch and MLX do not imply identical sampled codes. Separately
   compare logits with fixed inputs and the code path with greedy sampling.
   Do not change precision, fusion or sampling rules to chase speed before parity.

Preserve the MIT notices on code derived from #1960's mlx-audio ports and carry
applicable source/weight notices into the native distribution.

## Checkpoints and independent parity

The first checkpoint is `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, using the matching
speech tokenizer. Before implementation qualification, choose immutable model,
text-tokenizer and speech-tokenizer revisions and SHA-256 every consumed file in
`ci/model_pins.json`. Resolve these from the actual artifacts; this RFC does not
invent checkpoint revisions or checksums. Validate model kind, dimensions,
code-group counts, special IDs and every required tensor at load. Missing,
unexpected or incorrectly shaped inference weights must fail loading.

Use the Python MLX components at the pinned #1960 revision as the migration
reference. Run them from an isolated reference checkout/environment without
restoring them into `sglang_omni/`. Pin their dependencies, including mlx-audio
source provenance and the same MLX version as native (`0.32.3` at this base).
Use official Torch semantics to resolve known reference discrepancies such as
the codec window, rather than treating an old library output as authoritative.
#2430 supplies additional real-Mac behavior evidence, not an independent oracle
for the unmasked long-codec case.

The CLI should accept the same normalized request fields as HTTP and optionally
write diagnostics: prompt IDs, prompt embeddings, prefill/next-step hidden
states and logits, generated codec frames, finish reason and decoded waveform.
Reference generation must use the frozen Python implementation, not the native
binary being checked. Compare first-frame and post-feedback tensors, all 16
code groups, and fixed-code waveform output. Establish and document tensor and
waveform tolerances from measured same-device differences; retain exact token
and code assertions where applicable. A valid WAV alone is insufficient.

## Later increments

**Base voice cloning.** Add the matching 0.6B Base checkpoint, speech encoder,
ECAPA-TDNN and mel frontend from #1960. Accept bounded inline reference WAVs and
reference text through the established speech-request fields. Decode and
resample through native audio utilities; serving does not fetch URLs or read
caller-supplied filesystem paths. Match reference mel features, speaker embeddings
and encoded codes before end-to-end cloning. Decode reference and generated
codes in context, then exclude the reference waveform from the response.
A CustomVoice server must continue to reject cloning fields. Reference results
remain request-owned initially; no cross-request reference cache is needed.

**Streaming.** Enable `stream=true` only with raw PCM output after the codec
has a request-owned stateful decoder. Preserve causal convolution tails,
transposed-convolution overlap, absolute positions and bounded windowed KV state.
Drop consumed frames; apply bounded output buffering and backpressure so a slow
client cannot retain the full waveform. Flush the final partial chunk once.
For cloning, consume reference frames as context while suppressing their audio.
Compare concatenated output against one-shot decoding for identical fixed codes,
including irregular chunk sizes, more than 72 frames, and a short final chunk.
Exercise cancellation followed by another request to detect stale state.

## Mac CI and acceptance gates

The implementation adds `ci/qwen3_tts_golden.py`, populated model goldens under
`ci/model_golden/`, and focused `ci/test_qwen3_tts_api.py` coverage. Extend
`ci/check_golden.py`, `ci/provision_models.py` and
`.github/workflows/voxt-mac-ci.yaml` using their existing model hooks.
Do not change golden formats or introduce a generic TTS evaluation framework.

- Record English, Chinese and mixed-language requests, at least two voices,
  `auto` and explicit language, EOS and frame-cap cases. Add fixed-code
  96-frame waveform coverage; add clone and streaming cases with those increments.
- Follow the existing seeded TTS golden structure: exact prompt IDs everywhere,
  exact seeded codec frames and finish reasons on chips with recorded outputs.
  Report unrecorded-chip coverage explicitly; it is not exact-code qualification.
  Independently qualify fixed-code waveform parity to catch decoder regressions.
- Import real runner outputs from the `golden-outputs` artifact using the existing
  `--import` mechanism. Empty goldens, missing cases and outputs generated only
  by the implementation under test do not establish reference parity.
- Check WAV/PCM headers, sample count, finite/non-silent waveform, preset and
  language selection, malformed JSON/types, wrong model kind, unsupported fields,
  invalid limits/sampling, busy admission, disconnect cancellation, shutdown and
  recovery. Add streaming and clone failure contracts when supported.
- Target `main`, satisfy the non-draft/`run-mac-ci` gating and obtain real Mac
  CI results required by the MLX policy before claiming qualification. A docs-only
  RFC does not trigger native model qualification.

Report English WER, Chinese CER and mixed MER through a pinned ASR evaluator on
fixed generated-speech cases, plus listening samples; disclose every failed or
unscored sample. The ASR 392-clip transcription corpus is not itself a TTS input
suite. For Base, report speaker similarity separately from intelligibility.
Measure load time, end-to-end latency, realtime factor, throughput and peak
memory against the frozen Python reference on the same named Mac. Record model
and dependency pins, input set, sampling, concurrency, warmups and output modes;
add time to first PCM chunk when streaming lands. Keep performance outside CI.

## Delivery sequence

1. Freeze artifacts/reference and implement strict loading, prompts and talker
   parity through the synthesis CLI. Do not advertise serving before waveform
   decoding is complete.
2. Complete the native codec and CustomVoice HTTP consumer, shared worker reuse,
   populated Mac goldens, API coverage, cookbook and `Voxt/backend/README.md`
   updates. This is the first independently usable implementation PR.
3. Qualify Base voice cloning with its own pinned artifacts and goldens.
4. Qualify streaming with bounded state, waveform parity and cancellation coverage.

Each implementation PR must distinguish local checks, runner checks and pending
qualification. For this RFC, verification is documentation review and repository
pre-commit through the existing venv; unit tests and benchmarks are not run.
