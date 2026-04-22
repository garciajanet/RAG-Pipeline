"""
Evaluation harness for the RAG pipeline.

Usage:
    # Score retrieval only (fast — no LLM calls):
    python eval.py --retrieval-only

    # Full eval including LLM answer generation:
    python eval.py

    # Run a specific experiment config:
    python eval.py --embedding mxbai-embed-large --llm llama3.1:8b --chunk-size 128 --k 10

    # Force rebuild the vector index:
    python eval.py --rebuild
"""

import argparse
import csv
import json
import os
from datetime import datetime

from pipeline import PipelineConfig, build_pipeline

GOLD_SET_PATH = "gold_set.csv"
RESULTS_DIR = "eval_results"
FALLBACK_PHRASES = [
    "cannot find evidence",
    "no information",
    "not found",
    "no clients",
    "no evidence",
    "i don't have",
    "not mentioned",
    "not available",
]


def load_gold_set(path: str = GOLD_SET_PATH):
    questions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_ids = row["client_ids"].strip()
            if raw_ids == "NONE" or raw_ids == "":
                expected_ids = set()
                is_no_evidence = True
            else:
                expected_ids = set(raw_ids.replace(" ", "").split(","))
                is_no_evidence = False
            questions.append({
                "id": row["id"],
                "question": row["question"],
                "type": row["type"],
                "difficulty": row["difficulty"],
                "expected_answer": row["expected_answer"],
                "expected_ids": expected_ids,
                "is_no_evidence": is_no_evidence,
                "notes": row["notes"],
            })
    return questions


def score_retrieval(nodes, expected_ids: set, k: int):
    """Compute Recall@k and Precision@k from retrieved nodes."""
    retrieved_ids = {str(n.metadata.get("client_id", "")) for n in nodes}
    if not expected_ids:
        return None, None, retrieved_ids
    correct = retrieved_ids & expected_ids
    recall = len(correct) / len(expected_ids)
    precision = len(correct) / k if k > 0 else 0.0
    return round(recall, 4), round(precision, 4), retrieved_ids


def check_attribution(response_text: str, retrieved_ids: set, nodes):
    """
    Flag if the response mentions a client name that isn't in the retrieved nodes.
    Returns list of potentially unsupported client names (heuristic, not perfect).
    """
    retrieved_names = {
        n.metadata.get("client_name", "").lower()
        for n in nodes
        if n.metadata.get("client_name")
    }
    unsupported = []
    for name in retrieved_names:
        # If the name appears in response but wasn't in retrieved set — skip (it was retrieved)
        pass
    # Simpler check: extract any "Client: X" mentions from retrieved text
    # and verify they appear in retrieved nodes
    return unsupported  # placeholder for manual review


def is_fallback_response(response_text: str) -> bool:
    text = response_text.lower()
    return any(phrase in text for phrase in FALLBACK_PHRASES)


def run_eval(
    config: PipelineConfig,
    gold_questions: list,
    retrieval_only: bool = False,
    force_rebuild: bool = False,
    output_path: str = None,
):
    query_engine, retriever = build_pipeline(config, force_rebuild=force_rebuild)

    results = []
    for q in gold_questions:
        print(f"  [{q['id']}] {q['question'][:70]}...")

        nodes = retriever.retrieve(q["question"])
        recall, precision, retrieved_ids = score_retrieval(
            nodes, q["expected_ids"], config.similarity_top_k
        )

        row = {
            "config": config.label(),
            "question_id": q["id"],
            "question_type": q["type"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "n_expected": len(q["expected_ids"]) if not q["is_no_evidence"] else 0,
            "n_retrieved": len(retrieved_ids),
            "recall_at_k": recall,
            "precision_at_k": precision,
            "is_no_evidence": q["is_no_evidence"],
            "fallback_detected": None,
            "response": None,
        }

        if not retrieval_only:
            response = query_engine.query(q["question"])
            response_text = str(response)
            row["response"] = response_text
            if q["is_no_evidence"]:
                row["fallback_detected"] = is_fallback_response(response_text)

        results.append(row)
        _print_row(row)

    _print_summary(results, config)

    if output_path:
        _save_results(results, output_path)
    else:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = config.label().replace("|", "_").replace(":", "_")
        path = os.path.join(RESULTS_DIR, f"{ts}_{safe_label}.csv")
        _save_results(results, path)
        print(f"\nResults saved to {path}")

    return results


def _print_row(row):
    if row["is_no_evidence"]:
        fallback = "YES" if row["fallback_detected"] else ("NO" if row["fallback_detected"] is False else "—")
        print(f"    [no-evidence] fallback_detected={fallback}")
    elif row["recall_at_k"] is not None:
        print(
            f"    Recall@k={row['recall_at_k']:.3f}  "
            f"Precision@k={row['precision_at_k']:.3f}  "
            f"({row['n_retrieved']} retrieved / {row['n_expected']} expected)"
        )


def _print_summary(results, config):
    scored = [r for r in results if r["recall_at_k"] is not None]
    fallback_tests = [r for r in results if r["is_no_evidence"] and r["fallback_detected"] is not None]

    print(f"\n{'='*60}")
    print(f"Config: {config.label()}")
    if scored:
        avg_recall = sum(r["recall_at_k"] for r in scored) / len(scored)
        avg_precision = sum(r["precision_at_k"] for r in scored) / len(scored)
        print(f"Avg Recall@{config.similarity_top_k}:    {avg_recall:.3f}  (over {len(scored)} questions)")
        print(f"Avg Precision@{config.similarity_top_k}: {avg_precision:.3f}")
    if fallback_tests:
        n_correct = sum(1 for r in fallback_tests if r["fallback_detected"])
        print(f"Fallback accuracy: {n_correct}/{len(fallback_tests)}")
    print("="*60)


def _save_results(results, path):
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline against the gold set.")
    parser.add_argument("--embedding", default="mxbai-embed-large", help="Ollama embedding model name")
    parser.add_argument("--llm", default="llama3.1:8b", help="Ollama LLM model name")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=32)
    parser.add_argument("--k", type=int, default=20, help="similarity_top_k")
    parser.add_argument("--splitter", choices=["sentence", "simple"], default="sentence")
    parser.add_argument("--prompt-style", choices=["direct", "evidence", "fallback"], default="direct")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM generation; score retrieval only")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the vector index")
    parser.add_argument("--gold-set", default=GOLD_SET_PATH)
    parser.add_argument("--output", default=None, help="Path for results CSV (auto-named if omitted)")
    args = parser.parse_args()

    config = PipelineConfig(
        embedding_model=args.embedding,
        llm_model=args.llm,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        similarity_top_k=args.k,
        splitter=args.splitter,
        prompt_style=args.prompt_style,
    )

    gold_questions = load_gold_set(args.gold_set)
    print(f"Loaded {len(gold_questions)} gold questions from {args.gold_set}\n")

    run_eval(
        config=config,
        gold_questions=gold_questions,
        retrieval_only=args.retrieval_only,
        force_rebuild=args.rebuild,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
