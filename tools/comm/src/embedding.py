"""Sentence embedding (Ruri v3) wrapper.

Real implementation uses ``sentence-transformers`` with the Ruri v3 model and a
``文章: `` prefix on every input (design §6.5: messages are compared
symmetrically so both sides use the passage prefix).  All heavy imports are
performed lazily inside :func:`embed` so importing this module needs only the
standard library.

The acceptance harness never exercises the real path — it runs the sidecar in
``--mock`` mode (see ``nlp_sidecar.mock_embed``), which is pure stdlib.
"""

from __future__ import annotations

from typing import List


def embed(texts: List[str], model, prefix: str = "文章: ") -> List[List[float]]:
    """Embed ``texts`` and return one L2-normalized float vector per input.

    ``model`` is a ``sentence_transformers.SentenceTransformer`` instance (built
    by :func:`models.load_models`).  ``prefix`` is prepended to every text
    (Ruri's passage marker).  The returned vectors are L2-normalized; this is
    the contract the Rust client / clustering step relies on.
    """
    # Lazy heavy imports — never at module top level.
    import numpy as np  # noqa: WPS433

    prefixed = [f"{prefix}{t}" for t in texts]
    raw = model.encode(prefixed, convert_to_numpy=True, show_progress_bar=False)
    arr = np.asarray(raw, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = arr / norms
    return normalized.tolist()
