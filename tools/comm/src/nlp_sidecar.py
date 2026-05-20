#!/usr/bin/env python3
"""NLP sidecar entry point — JSONL stdin/stdout protocol (design §5.6 / §6.3).

The Rust ``comm-analyzer`` crate spawns this script as a child process and
exchanges one JSON object per line:

    request  -> stdin   (one JSON object per line, discriminated by "task")
    response <- stdout  (one JSON object per line, "task" echoed back)

Tasks: ``stance`` / ``sentiment`` / ``embed`` / ``cluster``.  On EOF the
process exits 0.  A malformed line produces an ``{"task":"error", ...}`` line
and the loop continues (it never crashes the loop).

Modes
-----
* ``--mock``  : deterministic, **stdlib-only** responses (used by tests / CI).
                No ML library is ever imported in this mode.
* real mode   : lazily loads models on first request of each task type;
                heavy imports happen inside :mod:`models` accessors only.  A
                model-load / inference failure produces an ``error`` response
                (the Rust side degrades gracefully) — it does not crash.

Usage:
    python3 tools/comm/src/nlp_sidecar.py --mock
    uv run python tools/comm/src/nlp_sidecar.py --profile balanced --device mps
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

# NOTE: only stdlib imported at module top level.  ALL ML imports
# (torch / transformers / sentence_transformers / hdbscan / umap / numpy /
# scipy) are lazily imported inside the real model code paths in models.py /
# embedding.py / sentiment.py / stance.py / clustering.py.


# --------------------------------------------------------------------------- #
# Mock implementations — deterministic, stdlib only.  The Rust integration
# tests depend on this contract exactly; do not change without updating them.
# --------------------------------------------------------------------------- #
def mock_sentiment(text: str) -> tuple:
    """``polarity in [-1, 1]``, ``magnitude in [0, 1)`` from a char-sum hash."""
    h = sum(ord(c) for c in text)
    polarity = ((h % 201) - 100) / 100.0
    magnitude = (h % 100) / 100.0
    return polarity, magnitude


def mock_embed_one(text: str) -> list:
    """8-float L2-normalized vector from the first 8 sha256 bytes of ``text``."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
    vec = [float(b) for b in digest]
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return [1.0 / math.sqrt(8.0)] * 8
    return [v / norm for v in vec]


def mock_cluster(embeddings: list, min_cluster_size: int) -> tuple:
    """Round-robin labels into ``k = max(1, n // max(1, min_cluster_size))``."""
    n = len(embeddings)
    mcs = max(1, int(min_cluster_size))
    k = max(1, n // mcs)
    labels = [i % k for i in range(n)]
    return labels, k


def handle_mock(req: dict) -> dict:
    """Dispatch one request in mock mode.  Always echoes ``id``."""
    task = req.get("task")
    rid = req.get("id", "")
    if task == "stance":
        return {"task": "stance", "id": rid, "label": "neutral", "score": 0.5}
    if task == "sentiment":
        polarity, magnitude = mock_sentiment(req.get("text", ""))
        return {
            "task": "sentiment",
            "id": rid,
            "polarity": polarity,
            "magnitude": magnitude,
        }
    if task == "embed":
        texts = req.get("texts", [])
        vectors = [mock_embed_one(t) for t in texts]
        return {"task": "embed", "id": rid, "vectors": vectors}
    if task == "cluster":
        labels, k = mock_cluster(
            req.get("embeddings", []), req.get("min_cluster_size", 2)
        )
        return {"task": "cluster", "labels": labels, "num_clusters": k}
    return {
        "task": "error",
        "id": rid,
        "message": f"unknown task: {task!r}",
    }


# --------------------------------------------------------------------------- #
# Real-mode dispatch (lazy model loading; only reached without --mock).
# --------------------------------------------------------------------------- #
class RealDispatcher:
    """Caches per-task models, importing ML libs lazily on first use."""

    def __init__(self, profile: str, device: str, models_cfg: dict):
        self._profile = profile
        self._device = device
        self._models_cfg = models_cfg
        self._registry = None

    def _reg(self):
        if self._registry is None:
            # Lazy import — keeps stdlib-only top level.
            from models import load_models  # noqa: WPS433

            self._registry = load_models(
                self._profile, self._device, self._models_cfg
            )
        return self._registry

    def handle(self, req: dict) -> dict:
        task = req.get("task")
        rid = req.get("id", "")
        try:
            if task == "stance":
                from stance import classify  # noqa: WPS433

                reg = self._reg()
                mode = reg.cfg.get("stance_mode", "nli")
                label, score = classify(
                    req.get("text", ""),
                    req.get("context"),
                    reg.stance(),
                    mode,
                )
                return {
                    "task": "stance",
                    "id": rid,
                    "label": label,
                    "score": float(score),
                }
            if task == "sentiment":
                from sentiment import score as sentiment_score  # noqa: WPS433

                polarity, magnitude = sentiment_score(
                    req.get("text", ""), self._reg().sentiment()
                )
                return {
                    "task": "sentiment",
                    "id": rid,
                    "polarity": float(polarity),
                    "magnitude": float(magnitude),
                }
            if task == "embed":
                from embedding import embed  # noqa: WPS433

                vectors = embed(req.get("texts", []), self._reg().embedder())
                return {"task": "embed", "id": rid, "vectors": vectors}
            if task == "cluster":
                from clustering import cluster  # noqa: WPS433

                labels, k = cluster(
                    req.get("embeddings", []),
                    req.get("min_cluster_size", 2),
                )
                return {
                    "task": "cluster",
                    "labels": labels,
                    "num_clusters": k,
                }
            return {
                "task": "error",
                "id": rid,
                "message": f"unknown task: {task!r}",
            }
        except Exception as exc:  # noqa: BLE001 — never crash the loop.
            return {
                "task": "error",
                "id": rid,
                "message": f"{type(exc).__name__}: {exc}",
            }


def _load_models_cfg(path) -> dict:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="comm-analyzer NLP sidecar")
    parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality"],
        default="balanced",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="cpu",
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--models-json", default=None)
    args = parser.parse_args()

    dispatcher = None
    if not args.mock:
        models_cfg = _load_models_cfg(args.models_json)
        dispatcher = RealDispatcher(args.profile, args.device, models_cfg)

    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("request is not a JSON object")
        except ValueError as exc:
            resp = {"task": "error", "id": "", "message": f"bad request: {exc}"}
            out.write(json.dumps(resp, ensure_ascii=False) + "\n")
            out.flush()
            continue

        if args.mock:
            resp = handle_mock(req)
        else:
            resp = dispatcher.handle(req)
        out.write(json.dumps(resp, ensure_ascii=False) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
