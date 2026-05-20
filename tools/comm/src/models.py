"""Lazy transformer model registry (design §6.5).

:func:`load_models` returns a :class:`ModelRegistry` whose ``embedder()`` /
``sentiment()`` / ``stance()`` accessors lazily build (and then cache) the
underlying models on first use.  This keeps process start-up cheap and means
the heavy ``torch`` / ``transformers`` / ``sentence_transformers`` imports only
happen inside the accessor bodies — never at import time.

The acceptance harness never calls these accessors (it runs ``--mock``).
"""

from __future__ import annotations

from typing import Dict, Optional

# Profile -> default model ids (design §6.5 profile matrix / §6.6.3).
_PROFILE_MODELS: Dict[str, Dict[str, str]] = {
    "fast": {
        "embedding": "cl-nagoya/ruri-v3-30m",
        "sentiment": "Mizuiro-sakura/bert-base-japanese-v2-wrime-fine-tune",
        "stance": "Formzu/bert-base-japanese-jsnli",
        "stance_mode": "nli",
        "stance_llm": "",
    },
    "balanced": {
        "embedding": "cl-nagoya/ruri-v3-130m",
        "sentiment": "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime",
        "stance": "Formzu/bert-base-japanese-jsnli",
        "stance_mode": "nli",
        "stance_llm": "",
    },
    "quality": {
        "embedding": "cl-nagoya/ruri-v3-310m",
        "sentiment": "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime",
        "stance": "Formzu/bert-base-japanese-jsnli",
        "stance_mode": "llm",
        "stance_llm": "sbintuitions/sarashina2.2-3b-instruct-v0.1",
    },
}


class _NliPipeline:
    """Adapter exposing ``model(premise, hypothesis) -> entailment prob``."""

    def __init__(self, pipe):
        self._pipe = pipe

    def __call__(self, premise: str, hypothesis: str) -> float:
        out = self._pipe({"text": premise, "text_pair": hypothesis})
        if isinstance(out, list):
            out = out[0]
        scores = {o["label"].lower(): float(o["score"]) for o in out} \
            if isinstance(out, list) else {out["label"].lower(): float(out["score"])}
        return scores.get("entailment", 0.0)


class _StanceModels:
    """Holds both NLI and (optional) LLM callables for :mod:`stance`."""

    def __init__(self, nli, llm):
        self.nli = nli
        self.llm = llm


class ModelRegistry:
    """Lazily-built model accessors keyed by profile/device/config."""

    def __init__(self, profile: str, device: str, cfg: Dict[str, str]):
        self.profile = profile
        self.device = device
        self.cfg = cfg
        self._embedder = None
        self._sentiment = None
        self._stance: Optional[_StanceModels] = None

    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # noqa: WPS433

            self._embedder = SentenceTransformer(
                self.cfg["embedding"], device=self.device
            )
        return self._embedder

    def sentiment(self):
        if self._sentiment is None:
            from transformers import pipeline  # noqa: WPS433

            self._sentiment = pipeline(
                "text-classification",
                model=self.cfg["sentiment"],
                top_k=None,
                device=-1 if self.device == "cpu" else 0,
            )
        return self._sentiment

    def stance(self) -> _StanceModels:
        if self._stance is None:
            from transformers import pipeline  # noqa: WPS433

            nli = _NliPipeline(
                pipeline(
                    "text-classification",
                    model=self.cfg["stance"],
                    device=-1 if self.device == "cpu" else 0,
                )
            )
            llm = None
            if self.cfg.get("stance_mode") == "llm" and self.cfg.get("stance_llm"):
                from transformers import (  # noqa: WPS433
                    AutoModelForCausalLM,
                    AutoTokenizer,
                )

                tok = AutoTokenizer.from_pretrained(self.cfg["stance_llm"])
                lm = AutoModelForCausalLM.from_pretrained(self.cfg["stance_llm"])

                def _llm(prompt: str) -> str:
                    ids = tok(prompt, return_tensors="pt").to(lm.device)
                    out = lm.generate(**ids, max_new_tokens=128)
                    return tok.decode(out[0], skip_special_tokens=True)

                llm = _llm
            self._stance = _StanceModels(nli, llm)
        return self._stance


def load_models(
    profile: str,
    device: str,
    models_cfg: Optional[Dict[str, str]] = None,
) -> ModelRegistry:
    """Build a :class:`ModelRegistry` for ``profile`` on ``device``.

    ``models_cfg`` (e.g. the resolved Rust ``ModelSet`` passed via
    ``--models-json``) overrides individual model ids.  No model is loaded
    here — accessors load lazily on first use.
    """
    base = dict(_PROFILE_MODELS.get(profile, _PROFILE_MODELS["balanced"]))
    if models_cfg:
        for key, value in models_cfg.items():
            if value:
                base[key] = value
    return ModelRegistry(profile, device, base)
