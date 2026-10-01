#!/usr/bin/env python3
"""
Script to execute evaluation queries against the retrieval endpoint
and reproduce results.json conforming strictly to the assessment requirements.
"""

import json
from pathlib import Path
from fastapi.testclient import TestClient
from server import app

SAMPLE_QUERIES = [
    {
        "query": "What are the four classes detected by the YOLOv8l model?",
        "top_k": 3
    },
    {
        "query": "What is the dataset size and how is it divided into training, validation, and testing sets?",
        "top_k": 3
    },
    {
        "query": "What technologies are used in the system?",
        "top_k": 3
    },
    {
        "query": "What are the main objectives of the proposed traffic violation detection system?",
        "top_k": 3
    }
]


def run_and_save_queries(output_file: str = "results.json"):
    client = TestClient(app)
    results_data = []

    print("=" * 64)
    print("  EXECUTING ASSESSMENT EVALUATION QUERIES VIA POST /query")
    print("=" * 64)

    for item in SAMPLE_QUERIES:
        q = item["query"]
        k = item["top_k"]
        print(f"\nQuery : '{q}' (top_k={k})")

        response = client.post("/query", json={"query": q, "top_k": k})
        if response.status_code != 200:
            print(f"  [ERROR] {response.status_code}: {response.text}")
            continue

        chunks = response.json()
        results_data.append({
            "query": q,
            "top_k": k,
            "results": chunks
        })

        for i, chunk in enumerate(chunks):
            page = chunk.get("page_number")
            score = chunk.get("score")
            preview = chunk.get("chunk_text", "").replace("\n", " ")[:100]
            print(f"  [{i+1}] Page {page} | Score: {score}")
            print(f"      Text: {preview}...")

    out_path = Path(output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print("-" * 64)
    print(f"[OK] Successfully saved {len(results_data)} query results to: {out_path.resolve()}")
    print("=" * 64)


if __name__ == "__main__":
    run_and_save_queries()
