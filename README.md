# RAG vs. OKF — Retrieval Benchmark

A hands-on comparison of two approaches for retrieving information from a knowledge base:

- **Retrieval-Augmented Generation (RAG):** retrieve semantically similar text chunks using embeddings and vector search.
- **Open Knowledge Format (OKF):** retrieve structured Markdown concepts using metadata, keyword matching, and links between documents.

The goal is to compare retrieval quality and latency on the same knowledge base and question set, then understand the trade-offs rather than declare a universal winner.

## Benchmark results

The latest recorded evaluation produced these metrics:

| Metric | RAG | OKF | Better result |
|---|---:|---:|---|
| Recall@3 | 84.67% | **86.00%** | OKF |
| Precision@3 | 35.56% | **37.78%** | OKF |
| MRR* | **1.0000** | 0.8222 | RAG* |
| nDCG@3* | **1.0000** | 0.8550 | RAG* |
| Mean latency (18 questions) | **17.94 s** | 42.66 s | RAG |

\* MRR and nDCG are reproduced as reported by the evaluation output. Their interpretation depends on how relevance and ranking are defined in the evaluation code; see `eval/retrieval_metrics.py`.

### What the numbers suggest

- **OKF has slightly higher top-3 coverage.** Recall@3 is 86.00% versus 84.67% for RAG, a difference of 1.33 percentage points.
- **OKF has slightly higher top-3 precision.** Precision@3 is 37.78% versus 35.56% for RAG, a difference of 2.22 percentage points. In this evaluation, the returned top-three results from OKF contain a slightly higher proportion of relevant results.
- **RAG ranks relevant results more strongly under the reported ranking metrics.** RAG has an MRR of 1.00 and nDCG@3 of 1.00, compared with 0.8222 and 0.8550 for OKF. Confirm the metric implementation and relevance labels before interpreting these values as a general ranking advantage.
- **RAG is faster in this run.** Mean latency was 17.94 seconds for RAG and 42.66 seconds for OKF. RAG's measured latency was about 58% lower; OKF took about 2.38 times as long on average.
- **The results are close on recall and precision, but differ more on ranking and latency.** The best choice depends on whether top-k coverage, ranking quality, response time, or explicit relationships between concepts matters most.

These are results from this benchmark run, not a guarantee of performance on other datasets. The size and composition of the question set, hardware, model configuration, caching, and the exact metric definitions can affect the results.

## The setup

### Knowledge base

The benchmark uses a synthetic but realistic internal knowledge base for a fictional AI Platform Team. It contains 20 interlinked Markdown documents covering:

- **Services (5):** auth-service, orders-api, agent-orchestrator, embedding-service, notification-service
- **Metrics (5):** p99 latency, agent success rate, throughput, auth failure rate, daily active agents
- **Runbooks (5):** deployment, rollback, incident response, key rotation, scaling
- **Architecture and process documents (5):** system overview, agent lifecycle, on-call process, and order data flow

The documents cross-reference one another so that questions can require facts from multiple concepts.

### Path A — RAG

1. Split documents into chunks by section header.
2. Embed each chunk with `nomic-embed-text` through Ollama.
3. Store vectors in a FAISS index (`IndexFlatIP` with L2-normalized vectors, so inner product corresponds to cosine similarity).
4. At query time, embed the question, retrieve the top three chunks, and pass the retrieved context to `phi3` for generation.

### Path B — OKF

The same documents are represented in OKF-style Markdown with YAML frontmatter such as `type`, `title`, `description`, and `tags`.

1. Match the question against a concept name or title.
2. If there is no direct match, use keyword overlap across the title, description, and tags.
3. Follow Markdown links from the matched concept to related concepts.
4. Pass the retrieved concept content to `phi3` for generation.

This approach relies on metadata, text matching, and explicit links rather than a vector index.

## Interpreting retrieval metrics

- **Recall@3** measures how much of the relevant material is retrieved within the top three results. Higher is generally better.
- **Precision@3** measures the proportion of the top-three results that are relevant. Higher is generally better.
- **MRR (Mean Reciprocal Rank)** rewards retrieving a relevant result near the top of the ranking. Higher is better.
- **nDCG@3 (normalized Discounted Cumulative Gain)** evaluates the quality of the ordering of the top three results, accounting for graded relevance if the evaluator supplies it. Higher is better.
- **Mean latency** is the average time recorded per question in this run. Lower is better.

Retrieval metrics are not the same as end-to-end answer correctness. A system can retrieve relevant material yet generate an incorrect answer, or generate a plausible answer despite retrieving weak evidence. Evaluate answer quality separately if the goal is to compare complete question-answering systems.

## Engineering observations

### Numeric facts need context during embedding

A chunk containing only a number, such as `94.2%`, carries little semantic information. It may not rank well for a natural-language question even if it contains the exact answer.

A useful mitigation is to prepend the document title and section heading to the text used for embedding. For example, embed `agent-success-rate - Current Value: 94.2%` rather than the bare value, while retaining the clean content for display and generation.

### Structural and semantic retrieval have different trade-offs

Explicit links can help when a question maps clearly to a concept and requires following its dependencies. Semantic embeddings can help when the question uses different wording from the source document. A hybrid design—semantic retrieval to identify candidate concepts, followed by link traversal to gather related context—is worth testing.

### Closed-book behavior needs to be evaluated separately

A retrieval system should not be assumed to refuse unsupported questions just because it uses RAG or OKF. Test unanswerable questions explicitly, provide clear instructions to answer only from retrieved evidence, and implement a confidence or abstention policy where appropriate.

## Limitations

- **Small benchmark:** 20 documents and 18 questions are useful for a demonstration, but are not enough to establish broad general performance.
- **Local model:** `phi3` is used for generation to keep the experiment local. Model size and configuration can affect answer quality and latency.
- **Metric definitions matter:** inspect `eval/retrieval_metrics.py` and the relevance labels before drawing conclusions, especially for MRR and nDCG.
- **Latency is environment-dependent:** hardware, warm-up, model loading, caching, and prompt length can change timings. Repeat runs under controlled conditions for a stronger comparison.
- **Retrieval and generation are separate stages:** report retrieval metrics and answer-quality metrics separately to identify where errors originate.

## Takeaway

In this run, **OKF edges ahead on Recall@3 and Precision@3**, while **RAG scores higher on the reported MRR and nDCG@3 values and has substantially lower mean latency**. The differences in recall and precision are small, so further runs and a larger question set would help establish whether they are robust.

Rather than treating either method as universally better, choose based on the workload and validate with representative questions. A hybrid approach combining semantic concept selection with structural link traversal may be a useful next experiment.

## Running it yourself

Requirements: Python 3.10+ and [Ollama](https://ollama.com/) installed and running.

```bash
# Pull the models
ollama pull nomic-embed-text
ollama pull phi3

# Install dependencies
pip install -r requirements.txt

# Build the RAG index
python rag_pipeline/build_index.py

# Ask a single question through either pipeline
python rag_pipeline/generate.py
python okf_pipeline/generate.py

# Run the evaluation
python eval/run_eval.py
```

## Project structure

```text
.
├── data/
│   ├── source_docs/       # Source Markdown documents for RAG
│   └── OKF_bundle/        # OKF-formatted documents
├── rag_pipeline/          # Chunking, embeddings, FAISS, retrieval, generation
├── okf_pipeline/          # Concept parsing, matching, link traversal, generation
├── eval/
│   ├── golden_dataset/    # Evaluation questions and expected sources
│   ├── retrieval_metrics.py
│   ├── run_eval.py
│   ├── results.json
│   └── summary.json
└── README.md
```
