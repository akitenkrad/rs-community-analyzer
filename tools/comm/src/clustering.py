"""HDBSCAN semantic clustering.

Real implementation clusters L2-normalized sentence embeddings with HDBSCAN
(design §6.2 / §4.5).  ``num_clusters`` is the number of distinct *non-noise*
labels (HDBSCAN assigns ``-1`` to noise points).  All heavy imports are lazy;
acceptance uses the stdlib ``--mock`` path only.
"""

from __future__ import annotations

from typing import List, Tuple


def cluster(
    embeddings: List[List[float]],
    min_cluster_size: int,
) -> Tuple[List[int], int]:
    """Return ``(labels, num_clusters)`` for ``embeddings``.

    ``labels[i]`` is the cluster id of point ``i`` (``-1`` = noise).
    ``num_clusters`` counts distinct non-noise labels.  Fewer points than
    ``min_cluster_size`` yields all-noise / zero clusters.
    """
    import hdbscan  # noqa: WPS433 — lazy heavy import.
    import numpy as np  # noqa: WPS433

    if not embeddings:
        return [], 0

    arr = np.asarray(embeddings, dtype="float32")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, int(min_cluster_size)))
    labels = clusterer.fit_predict(arr)
    distinct = {int(label) for label in labels if int(label) != -1}
    return [int(label) for label in labels], len(distinct)
