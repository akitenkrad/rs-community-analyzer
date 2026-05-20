"""WRIME 8-emotion sentiment scoring.

Real implementation runs a LUKE / BERT model fine-tuned on the WRIME dataset
(design §6.5) which yields 8 emotion intensities:

    喜び (joy), 期待 (anticipation), 信頼 (trust),
    悲しみ (sadness), 怒り (anger), 恐れ (fear), 嫌悪 (disgust), 驚き (surprise)

Mapping to the sidecar's ``(polarity, magnitude)`` contract
--------------------------------------------------------
* ``polarity`` = ((喜び + 期待 + 信頼) − (悲しみ + 怒り + 恐れ + 嫌悪))
  normalized to ``[-1, 1]`` by dividing by the sum of those 7 contributing
  intensities (驚き is treated as neutral / ambivalent and excluded from the
  signed sum).  Empty / all-zero → ``0.0``.
* ``magnitude`` = sum of all 8 intensities normalized to ``[0, 1]`` (divided by
  8 since each WRIME intensity is in ``[0, 1]``).

All heavy imports are lazy.  Acceptance uses ``--mock`` (stdlib only).
"""

from __future__ import annotations

from typing import Tuple

# Order of the 8 WRIME emotion labels as emitted by the model head.
_POSITIVE = ("喜び", "期待", "信頼")
_NEGATIVE = ("悲しみ", "怒り", "恐れ", "嫌悪")
_ALL = _POSITIVE + _NEGATIVE + ("驚き",)


def score(text: str, model) -> Tuple[float, float]:
    """Return ``(polarity in [-1, 1], magnitude in [0, 1])`` for ``text``.

    ``model`` is a callable ``transformers`` pipeline (built by
    :func:`models.load_models`) returning a list of ``{"label", "score"}``
    dicts covering the 8 WRIME emotions.
    """
    raw = model(text)
    # ``transformers`` pipelines may wrap the result in an extra list.
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    intensities = {item["label"]: float(item["score"]) for item in raw}

    pos = sum(intensities.get(label, 0.0) for label in _POSITIVE)
    neg = sum(intensities.get(label, 0.0) for label in _NEGATIVE)
    signed_total = pos + neg
    polarity = 0.0 if signed_total == 0.0 else (pos - neg) / signed_total
    polarity = max(-1.0, min(1.0, polarity))

    all_sum = sum(intensities.get(label, 0.0) for label in _ALL)
    magnitude = max(0.0, min(1.0, all_sum / float(len(_ALL))))
    return polarity, magnitude
