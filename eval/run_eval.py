import sys
import os
import json
import time
import importlib.util
import contextlib
import io
import statistics

from retrieval_metrics import evaluate_retrieval, aggregate_metrics


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(PROJECT_ROOT, "eval")
RAG_DIR = os.path.join(PROJECT_ROOT, "rag_pipeline")
OKF_DIR = os.path.join(PROJECT_ROOT, "okf_pipeline")

QUESTIONS_PATH = os.path.join(
    EVAL_DIR, "golden_dataset", "questions.json"
)
RESULTS_PATH = os.path.join(EVAL_DIR, "results.json")
SUMMARY_PATH = os.path.join(EVAL_DIR, "summary.json")

TOP_K = 3


def load_module(module_name, file_path, extra_sys_path):
    """Load a Python file as a module and support its local imports."""
    if extra_sys_path not in sys.path:
        sys.path.insert(0, extra_sys_path)

    spec = importlib.util.spec_from_file_location(
        module_name, file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_quietly(func, *args, **kwargs):
    """Run a function while suppressing its printed debug output."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return func(*args, **kwargs)


def get_rag_sources(rag_search, question):
    """Return retrieved RAG source paths in retrieval-rank order."""
    hits = run_quietly(
        rag_search.search, question, top_k=TOP_K
    )

    # RAG search returns (score, match) pairs.
    return [
        match["source"]
        for score, match in hits
        if isinstance(match, dict) and match.get("source")
    ]


def get_okf_sources(okf_search, question):
    """Return OKF concept IDs in the order returned by its search."""
    hits = run_quietly(
        okf_search.search_with_hops,
        question,
        max_hops=1
    )

    # OKF search returns (concept_id, concept_dict) pairs.
    return [
        concept_id
        for concept_id, concept in hits
        if concept_id
    ]


def latency_summary(results, field):
    """Summarize end-to-end generation latency in seconds."""
    values = [
        row[field]
        for row in results
        if isinstance(row.get(field), (int, float))
    ]

    if not values:
        return {
            "count": 0,
            "mean_seconds": None,
            "median_seconds": None,
            "total_seconds": None,
        }

    return {
        "count": len(values),
        "mean_seconds": round(statistics.mean(values), 3),
        "median_seconds": round(statistics.median(values), 3),
        "total_seconds": round(sum(values), 3),
    }


def build_summary(results):
    """Calculate aggregate retrieval metrics and latency statistics."""
    # Retrieval quality is averaged over answerable questions only.
    # Questions with no expected sources are not retrieval-recall cases.
    rag_rows = [
        row["rag_retrieval"]
        for row in results
        if row.get("rag_retrieval", {}).get(
            "retrieval_metrics_applicable", False
        )
    ]

    okf_rows = [
        row["okf_retrieval"]
        for row in results
        if row.get("okf_retrieval", {}).get(
            "retrieval_metrics_applicable", False
        )
    ]

    metric_names = (
        "recall@3",
        "precision@3",
        "mrr",
        "ndcg@3",
    )

    return {
        "questions_evaluated": len(results),
        "retrieval_metrics": {
            "k": TOP_K,
            "averaging": "macro average across answerable questions",
            "rag": aggregate_metrics(rag_rows, metric_names),
            "okf": aggregate_metrics(okf_rows, metric_names),
            "answerable_questions_scored": len(rag_rows),
            "unanswerable_questions_excluded_from_retrieval_averages": (
                len(results) - len(rag_rows)
            ),
        },
        "latency": {
            "unit": "seconds",
            "measurement": (
                "End-to-end generate_answer call, including the "
                "pipeline's own retrieval and LLM generation"
            ),
            "rag": latency_summary(results, "rag_time_seconds"),
            "okf": latency_summary(results, "okf_time_seconds"),
        },
    }


def main():
    required_files = [
        QUESTIONS_PATH,
        os.path.join(RAG_DIR, "generate.py"),
        os.path.join(RAG_DIR, "search.py"),
        os.path.join(OKF_DIR, "generate.py"),
        os.path.join(OKF_DIR, "okf_search.py"),
    ]

    missing = [path for path in required_files if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Required project files not found:\n"
            + "\n".join(missing)
        )

    # Load generation and retrieval modules.
    rag_generate = load_module(
        "rag_generate",
        os.path.join(RAG_DIR, "generate.py"),
        RAG_DIR,
    )

    rag_search = load_module(
        "rag_search_module",
        os.path.join(RAG_DIR, "search.py"),
        RAG_DIR,
    )

    okf_generate = load_module(
        "okf_generate",
        os.path.join(OKF_DIR, "generate.py"),
        OKF_DIR,
    )

    okf_search = load_module(
        "okf_search_module",
        os.path.join(OKF_DIR, "okf_search.py"),
        OKF_DIR,
    )

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    results = []
    total = len(questions)

    for i, question_data in enumerate(questions, start=1):
        question = question_data["question"]
        expected_sources = question_data.get("expected_sources", [])

        print(f"[{i}/{total}] {question}")

        # Retrieve source rankings for metric calculations.
        # These calls are separate from the timed generation calls.
        rag_sources = get_rag_sources(rag_search, question)
        okf_sources = get_okf_sources(okf_search, question)

        rag_retrieval_metrics = evaluate_retrieval(
            rag_sources,
            expected_sources,
            k=TOP_K,
        )

        okf_retrieval_metrics = evaluate_retrieval(
            okf_sources,
            expected_sources,
            k=TOP_K,
        )

        # Measure end-to-end generation latency for each pipeline.
        start = time.perf_counter()
        rag_answer = run_quietly(
            rag_generate.generate_answer, question
        )
        rag_time = time.perf_counter() - start

        start = time.perf_counter()
        okf_answer = run_quietly(
            okf_generate.generate_answer, question
        )
        okf_time = time.perf_counter() - start

        result = {
            "id": question_data["id"],
            "type": question_data["type"],
            "question": question,
            "expected_answer": question_data["expected_answer"],
            "expected_sources": expected_sources,

            "rag_answer": rag_answer,
            "rag_time_seconds": round(rag_time, 3),
            "rag_retrieval": rag_retrieval_metrics,

            "okf_answer": okf_answer,
            "okf_time_seconds": round(okf_time, 3),
            "okf_retrieval": okf_retrieval_metrics,
        }

        results.append(result)

        print(
            f"   RAG: {rag_time:.2f}s | "
            f"Recall@3={rag_retrieval_metrics['recall@3']} | "
            f"Precision@3={rag_retrieval_metrics['precision@3']}"
        )
        print(
            f"   OKF: {okf_time:.2f}s | "
            f"Recall@3={okf_retrieval_metrics['recall@3']} | "
            f"Precision@3={okf_retrieval_metrics['precision@3']}"
        )

    # Save per-question answers, timings, and retrieval metrics.
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    # Save macro-averaged metrics and latency summaries.
    summary = build_summary(results)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nEvaluation complete.")
    print(f"Per-question results: {RESULTS_PATH}")
    print(f"Aggregate summary:    {SUMMARY_PATH}")

    print("\nAggregate retrieval metrics:")
    print(json.dumps(summary["retrieval_metrics"], indent=2))

    print("\nLatency summary:")
    print(json.dumps(summary["latency"], indent=2))


if __name__ == "__main__":
    main()