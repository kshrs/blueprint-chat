#!/usr/bin/env python3
"""CLI interface for Blueprint-Chat Multimodal RAG."""
import argparse
import sys
from pathlib import Path
from src.rag_pipeline import MultimodalRAGPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Blueprint-Chat: Multimodal RAG with Late-Fusion Reranking"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Fire exit evacuation route for Lab B-201",
        help="Query to search across technical manuals and blueprint schematics"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Late-fusion weight for text (1.0 = text only, 0.0 = blueprint only)"
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent / "data"
    pipeline = MultimodalRAGPipeline(str(data_dir), alpha=args.alpha)

    print("==================================================")
    print("   Blueprint-Chat: Multimodal Technical RAG       ")
    print("==================================================")
    print(f"Query: {args.query}")
    print(f"Late-Fusion Alpha: {args.alpha} (Text weight: {args.alpha}, Visual weight: {round(1 - args.alpha, 2)})\n")

    result = pipeline.query(args.query)

    print("--- Retrievable Multimodal Context (After Late-Fusion) ---")
    for idx, item in enumerate(result["reranked_context"], 1):
        if item["type"] == "text_manual":
            print(f"[{idx}] (TEXT) [{item['section']}] Fused Score: {item['fused_score']} (Raw: {round(item['raw_score'], 3)})")
            print(f"    Excerpt: {item['content']}")
        else:
            print(f"[{idx}] (VISUAL) [{item['title']}] Fused Score: {item['fused_score']} (Raw: {round(item['raw_score'], 3)})")
            print(f"    Diagram Detail: {item['description']}")
        print()

    print("--- Grounded Answer Synthesis ---")
    print(result["answer"])
    print("==================================================")

if __name__ == "__main__":
    main()
