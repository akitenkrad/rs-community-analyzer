"""Stance classification: {support, neutral, disagree, clarify}.

Two strategies (design §6.4):

* **NLI zero-shot** (default, ``mode="nli"``): premise = parent / thread-root
  message (``context``), hypothesis = a per-label Japanese template; the label
  whose ``entailment`` probability is highest wins.
* **LLM prompting** (``mode="llm"``, used by the ``quality`` profile): a
  ``sbintuitions/sarashina2.2-3b-instruct-v0.1`` instruction prompt asking for
  a JSON ``{"label": ...}`` answer; on parse failure it falls back to NLI.

This module is wired now but is only consumed by Phase 5 (H1).  In Phase 4 it
is exercised solely through the ``--mock`` path.  All heavy imports are lazy.
"""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple

# stance label -> NLI hypothesis template (design §6.4).
_TEMPLATES = {
    "support": "この発言は前の意見に賛成している",
    "disagree": "この発言は前の意見に反対している",
    "neutral": "この発言は前の意見について中立的である",
    "clarify": "この発言は前の意見に対して質問・確認をしている",
}

_LLM_PROMPT = """あなたは組織コミュニケーション分析の専門家です．
以下のスレッド文脈と対象発言を読み，対象発言の stance を以下から 1 つ選んでください．

- support: 賛成・同調
- disagree: 反対・異論
- neutral: 中立・態度保留
- clarify: 質問・確認

[スレッド文脈]
{context}

[対象発言]
{text}

回答は JSON 形式で {{"label": "...", "reason": "..."}} とすること．"""


def _nli_classify(text: str, context: Optional[str], model) -> Tuple[str, float]:
    """Zero-shot NLI: pick the label with the highest entailment probability."""
    premise = context if context else text
    best_label = "neutral"
    best_score = 0.0
    for label, template in _TEMPLATES.items():
        # ``model`` is an NLI pipeline returning entailment probability for the
        # (premise, hypothesis) pair.
        prob = float(model(premise, template))
        if prob > best_score:
            best_score = prob
            best_label = label
    return best_label, best_score


def _llm_classify(text: str, context: Optional[str], model) -> Tuple[str, float]:
    """LLM prompting with JSON parse; raises on unparseable output."""
    prompt = _LLM_PROMPT.format(context=context or "(なし)", text=text)
    raw = model(prompt)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("LLM output had no JSON object")
    parsed = json.loads(match.group(0))
    label = parsed.get("label", "neutral")
    if label not in _TEMPLATES:
        label = "neutral"
    return label, 1.0


def classify(
    text: str,
    context: Optional[str],
    model,
    mode: str = "nli",
) -> Tuple[str, float]:
    """Return ``(label, score)`` with ``label`` in the 4-class stance set.

    ``model`` is the registry entry built by :func:`models.load_models`; for
    ``mode="llm"`` it is a causal-LM callable, otherwise an NLI pipeline.  LLM
    parse failures fall back to NLI (design §6.4).
    """
    if mode == "llm":
        try:
            return _llm_classify(text, context, model.llm)
        except Exception:  # noqa: BLE001 — design §6.4: fall back to NLI.
            return _nli_classify(text, context, model.nli)
    return _nli_classify(text, context, model.nli)
