"""Source-level retrieval metrics for the RAG-vs-OKF benchmark."""
import math
from pathlib import PurePosixPath


def normalize_source(source):
    value = str(source or "").replace("\\\\", "/").strip().rstrip("/")
    name = PurePosixPath(value).name.lower()
    return name[:-3] if name.endswith(".md") else name


def dedupe_sources(sources):
    result, seen = [], set()
    for source in sources:
        name = normalize_source(source)
        if name and name not in seen:
            result.append(name)
            seen.add(name)
    return result


def evaluate_retrieval(retrieved_sources, expected_sources, k=3):
    expected = set(dedupe_sources(expected_sources))
    retrieved = dedupe_sources(retrieved_sources)

    if k <= 0:
        raise ValueError("k must be positive")

    # Recall/MRR/nDCG are undefined when no relevant sources are expected.
    recall = (len(set(retrieved[:k]) & expected) / len(expected)) if expected else None
    precision = sum(source in expected for source in retrieved[:k]) / k

    rr = None
    if expected:
        rr = next((1.0 / rank for rank, source in enumerate(retrieved, 1)
                   if source in expected), 0.0)

    if not expected:
        ndcg = None
    else:
        rel = [int(source in expected) for source in retrieved[:k]]
        rel += [0] * (k - len(rel))
        dcg = sum(score / math.log2(rank + 1)
                  for rank, score in enumerate(rel, 1))
        ideal = sorted(rel, reverse=True)
        idcg = sum(score / math.log2(rank + 1)
                   for rank, score in enumerate(ideal, 1))
        ndcg = dcg / idcg if idcg else 0.0

    return {
        f"recall@{k}": recall,
        f"precision@{k}": precision,
        "mrr": rr,
        f"ndcg@{k}": ndcg,
        "retrieved_sources": retrieved[:k],
        "expected_sources": sorted(expected),
        "retrieval_metrics_applicable": bool(expected),
    }


def aggregate_metrics(rows, metric_names=("recall@3", "precision@3", "mrr", "ndcg@3")):
    summary = {}
    for name in metric_names:
        values = [row[name] for row in rows
                  if isinstance(row.get(name), (int, float))]
        summary[name] = sum(values) / len(values) if values else None
        summary[name + "_n"] = len(values)
    return summary
