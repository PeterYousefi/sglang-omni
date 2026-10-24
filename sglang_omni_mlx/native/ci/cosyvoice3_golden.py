# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 golden check for check_golden.py: seeded speech requests through
cosyvoice3_synthesize, whose text ids and speech tokens a chip with golden outputs
must give exactly; voxtral_golden.check compares them. Another chip's GPU
arithmetic may sample other tokens, so there only the text ids, which the
tokenizer alone sets, must match.

A golden file of kind cosyvoice3 names its cases (text, optional instructions,
seed and max_new_tokens); each chip's outputs keep one string per speech token, so
a divergence reads as the token it starts at.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


def synthesize(runtime_bin: Path, model_directory: Path, case: dict) -> dict:
    instruction_flags = (
        ["--instructions", case["instructions"]] if "instructions" in case else []
    )
    with tempfile.TemporaryDirectory(prefix="cosyvoice3-golden-") as directory:
        tokens_path = Path(directory) / "tokens.json"
        completed = subprocess.run(
            [
                str(runtime_bin / "cosyvoice3_synthesize"),
                "--model-path",
                str(model_directory),
                "--text",
                case["text"],
                "--seed",
                str(case["seed"]),
                "--max-new-tokens",
                str(case["max_new_tokens"]),
                "--tokens-output",
                str(tokens_path),
                *instruction_flags,
            ],  # fmt: skip
            check=True,
            capture_output=True,
            text=True,
        )
        tokens = json.loads(tokens_path.read_text())
    return {
        "prompt_ids": " ".join(str(value) for value in tokens["text_ids"]),
        "finish_reason": json.loads(completed.stdout.splitlines()[-1])["finish_reason"],
        "codes": [str(token) for token in tokens["speech_tokens"]],
    }


def synthesize_cases(
    runtime_bin: Path, model_directory: Path, cases: dict[str, dict]
) -> dict[str, dict]:
    return {
        case_id: synthesize(runtime_bin, model_directory, case)
        for case_id, case in cases.items()
    }
