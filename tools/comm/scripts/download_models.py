"""モデルを事前ダウンロードするスクリプト．

Usage:
    uv run python tools/comm/scripts/download_models.py --profile balanced
    uv run python tools/comm/scripts/download_models.py --profile all
    uv run python tools/comm/scripts/download_models.py --profile fast

`huggingface_hub` は :func:`download` 内で遅延 import するため，本ファイルの
import 自体は標準ライブラリだけで成立する（設計書 §6.6.3）．
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Profile -> model ids (design §6.5 / §6.6.3 / §16).
PROFILES = {
    "fast": [
        "cl-nagoya/ruri-v3-30m",
        "Mizuiro-sakura/bert-base-japanese-v2-wrime-fine-tune",
        "Formzu/bert-base-japanese-jsnli",
    ],
    "balanced": [
        "cl-nagoya/ruri-v3-130m",
        "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime",
        "Formzu/bert-base-japanese-jsnli",
    ],
    "quality": [
        "cl-nagoya/ruri-v3-310m",
        "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime",
        "sbintuitions/sarashina2.2-3b-instruct-v0.1",
    ],
}


def download(model_id: str) -> Path:
    """単一モデルを HF キャッシュへダウンロードしてパスを返す．"""
    # Lazy import — keeps the module importable with stdlib only.
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    try:
        path = snapshot_download(
            repo_id=model_id,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "*.txt",
                "tokenizer*",
                "special_tokens_map.json",
            ],
            resume_download=True,
            max_workers=4,
        )
        return Path(path)
    except HfHubHTTPError as e:
        print(f"  ERROR: {model_id}: {e}", file=sys.stderr)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=[*PROFILES.keys(), "all"],
        default="balanced",
    )
    args = parser.parse_args()

    profiles = list(PROFILES.keys()) if args.profile == "all" else [args.profile]
    models: list[str] = []
    for p in profiles:
        models.extend(PROFILES[p])
    models = sorted(set(models))  # 重複排除

    print(f"Downloading {len(models)} models for profile(s): {profiles}")
    for i, m in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {m}")
        download(m)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
