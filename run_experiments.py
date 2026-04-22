"""
Run all comparative experiments and collect results.

Experiments cover:
  - Embedding models: mxbai-embed-large, nomic-embed-text, all-minilm
  - Chunk sizes: 128, 256, 512
  - Splitters: sentence, simple
  - k values: 5, 10, 20
  - Prompt styles: direct, evidence, fallback
  - LLMs: llama3.1:8b, mistral (run retrieval-only for speed unless LLM comparison needed)

Usage:
    # Retrieval-only sweep (fast — no LLM calls):
    python run_experiments.py --retrieval-only

    # Full sweep including LLM generation (slow):
    python run_experiments.py
"""

import argparse
import csv
import os
from datetime import datetime

from pipeline import PipelineConfig
from eval import load_gold_set, run_eval

RESULTS_DIR = "eval_results"

# --- Define experiment matrix ---

EMBEDDING_EXPERIMENTS = [
    PipelineConfig(embedding_model="mxbai-embed-large", chunk_size=256, splitter="sentence", similarity_top_k=20),
    PipelineConfig(embedding_model="nomic-embed-text",  chunk_size=256, splitter="sentence", similarity_top_k=20),
    PipelineConfig(embedding_model="all-minilm",        chunk_size=256, splitter="sentence", similarity_top_k=20),
]

CHUNK_EXPERIMENTS = [
    PipelineConfig(embedding_model="mxbai-embed-large", chunk_size=128, chunk_overlap=16,  splitter="sentence", similarity_top_k=20),
    PipelineConfig(embedding_model="mxbai-embed-large", chunk_size=256, chunk_overlap=32,  splitter="sentence", similarity_top_k=20),
    PipelineConfig(embedding_model="mxbai-embed-large", chunk_size=512, chunk_overlap=64,  splitter="sentence", similarity_top_k=20),
]

SPLITTER_EXPERIMENTS = [
    PipelineConfig(embedding_model="mxbai-embed-large", chunk_size=256, splitter="sentence", similarity_top_k=20),
    PipelineConfig(embedding_model="mxbai-embed-large", chunk_size=256, splitter="simple",   similarity_top_k=20),
]

K_EXPERIMENTS = [
    PipelineConfig(embedding_model="mxbai-embed-large", chunk_size=256, splitter="sentence", similarity_top_k=5),
    PipelineConfig(embedding_model="mxbai-embed-large", chunk_size=256, splitter="sentence", similarity_top_k=10),
    PipelineConfig(embedding_model="mxbai-embed-large", chunk_size=256, splitter="sentence", similarity_top_k=20),
]

PROMPT_EXPERIMENTS = [
    PipelineConfig(embedding_model="mxbai-embed-large", chunk_size=256, splitter="sentence", similarity_top_k=20, prompt_style="direct"),
    PipelineConfig(embedding_model="mxbai-embed-large", chunk_size=256, splitter="sentence", similarity_top_k=20, prompt_style="evidence"),
    PipelineConfig(embedding_model="mxbai-embed-large", chunk_size=256, splitter="sentence", similarity_top_k=20, prompt_style="fallback"),
]

ALL_EXPERIMENTS = (
    EMBEDDING_EXPERIMENTS
    + CHUNK_EXPERIMENTS[0:1]   # skip 256 — already in embedding experiments
    + CHUNK_EXPERIMENTS[2:3]
    + SPLITTER_EXPERIMENTS[1:2]
    + K_EXPERIMENTS[0:2]
    + PROMPT_EXPERIMENTS[1:]
)


def merge_results(all_results: list[list[dict]], output_path: str):
    if not all_results:
        return
    rows = [r for results in all_results for r in results]
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAll results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild all indexes")
    parser.add_argument(
        "--suite",
        choices=["embedding", "chunk", "splitter", "k", "prompt", "all"],
        default="all",
        help="Which experiment suite to run",
    )
    args = parser.parse_args()

    suite_map = {
        "embedding": EMBEDDING_EXPERIMENTS,
        "chunk": CHUNK_EXPERIMENTS,
        "splitter": SPLITTER_EXPERIMENTS,
        "k": K_EXPERIMENTS,
        "prompt": PROMPT_EXPERIMENTS,
        "all": ALL_EXPERIMENTS,
    }
    experiments = suite_map[args.suite]

    gold_questions = load_gold_set()
    print(f"Loaded {len(gold_questions)} gold questions\n")
    print(f"Running {len(experiments)} experiments (suite={args.suite})\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_path = os.path.join(RESULTS_DIR, f"{ts}_all_{args.suite}.csv")

    all_results = []
    for i, config in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {config.label()}")
        results = run_eval(
            config=config,
            gold_questions=gold_questions,
            retrieval_only=args.retrieval_only,
            force_rebuild=args.rebuild,
            output_path=os.devnull,
        )
        all_results.append(results)

    merge_results(all_results, merged_path)


if __name__ == "__main__":
    main()
