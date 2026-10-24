// SPDX-License-Identifier: Apache-2.0
// Fun-CosyVoice3 0.5B: a Qwen2 decoder that samples 25 Hz speech tokens after
// the text, the DiT flow that turns them into mel frames, and the HiFT
// vocoder that turns those into 24 kHz samples.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/layers.h"
#include "common/tokenizer.h"
#include "cosyvoice3/cosyvoice3_vocoder.h"
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace cosyvoice3 {

inline constexpr int kSpeechTokenSize = 6561;
inline constexpr int kTotalSpeechVocabularySize = kSpeechTokenSize + 200;
inline constexpr int kSosTokenId = kSpeechTokenSize;
inline constexpr int kTaskTokenId = kSpeechTokenSize + 2;
// The most speech tokens one request generates: 2048 tokens, 82 seconds.
inline constexpr int kMaxSpeechTokens = 2048;
// The decoder's context: SOS, text, TASK and speech tokens.
inline constexpr int kMaxContextTokens = 32768;

// The sampler's settings; the defaults are the CUDA server's.
struct SamplingOptions {
  // Below 1e-6 samples greedily.
  float temperature = 0.7f;
  float top_p = 0.8f;
  // -1 keeps every token.
  int top_k = 20;
  float repetition_penalty = 1.21f;
  uint64_t seed = 0;
  // At most this many speech tokens, else 20 per target text token.
  std::optional<int> max_new_tokens;
};

// One request's text: the target, and the instruction that styles it.
struct SpeechText {
  std::string text;
  std::optional<std::string> instructions;
};

struct GeneratedSpeech {
  // As sampled, before silent runs are trimmed.
  std::vector<int> speech_tokens;
  qwen3_asr::FinishReason finish_reason = qwen3_asr::FinishReason::kStop;
};

class CosyVoice3Model {
public:
  explicit CosyVoice3Model(const std::filesystem::path &model_directory);
  CosyVoice3Model(const CosyVoice3Model &) = delete;
  CosyVoice3Model &operator=(const CosyVoice3Model &) = delete;

  // The prompt's text token ids, then the target text's.
  std::pair<std::vector<int>, std::vector<int>>
  EncodeText(const SpeechText &text) const;

  // Samples speech tokens after the text ids; calls on_token with each one
  // and throws TranscriptionCancelled once cancel is set.
  GeneratedSpeech Generate(const std::vector<int> &prompt_text_ids,
                           const std::vector<int> &target_text_ids,
                           const SamplingOptions &options,
                           const std::atomic<bool> &cancel,
                           const std::function<void(int)> &on_token) const;

  // Float32 samples of speech tokens, with no reference speaker; calls
  // on_step before each flow step, which may throw to stop it.
  mlx::core::array Vocode(const std::vector<int> &speech_tokens,
                          const std::function<void()> &on_step) const;
  // Vocode's two halves: the flow's mel [1, 80, frames], and its samples.
  mlx::core::array Mel(const std::vector<int> &speech_tokens,
                       const std::function<void()> &on_step) const;
  mlx::core::array Samples(const mlx::core::array &mel) const;

  // Logits [1, 1, 6761] after embeddings [1, tokens, 896], appended to
  // caches.
  mlx::core::array
  ForwardEmbeddings(const mlx::core::array &embeddings,
                    std::vector<qwen3_asr::KVCache> &caches) const;
  // [SOS, text, TASK, prompt speech] embeddings.
  mlx::core::array
  BuildPromptEmbeddings(const std::vector<int> &text_token_ids,
                        const std::vector<int> &prompt_speech_token_ids) const;
  // Includes the extended control-token vocabulary for teacher forcing.
  mlx::core::array
  SpeechEmbeddings(const std::vector<int> &speech_token_ids) const;
  std::vector<qwen3_asr::KVCache> NewCaches() const;

private:
  // Ids of text with the checkpoint's special tokens split out.
  std::vector<int> Encode(const std::string &text) const;

  qwen3_asr::Checkpoint checkpoint_;
  qwen3_asr::Qwen3Decoder decoder_;
  std::unique_ptr<qwen3_asr::Tokenizer> tokenizer_;
  Flow flow_;
  Hift hift_;
};

// Speech tokens with every run of more than five silent tokens cut to five.
std::vector<int> TrimSilentRuns(const std::vector<int> &speech_tokens);

} // namespace cosyvoice3
