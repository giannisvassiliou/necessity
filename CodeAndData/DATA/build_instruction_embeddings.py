#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the example-retrieval index used by exp1b_wired.EmbeddingSearcher.

Embeds every natural-language paraphrase in instructions.json with
BAAI/bge-large-en-v1.5 (fastembed) and writes

    embeddings_output/instructions_embeddings.pkl
        {query_id: {"instructions": [str, ...],
                    "embeddings":   np.ndarray (n_paraphrases, 1024)}}

which is exactly the structure EmbeddingSearcher reads.

    python build_instruction_embeddings.py
    python build_instruction_embeddings.py --check old_instructions_embeddings.pkl

--check compares a freshly built index against an existing one (same ids,
same paraphrases, cosine similarity of each vector) so you can confirm the
rebuilt index matches the one used for the reported runs.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

MODEL = "BAAI/bge-large-en-v1.5"


def build(instructions_path: Path, batch_size: int = 64) -> dict:
    from fastembed import TextEmbedding

    data = json.loads(instructions_path.read_text(encoding="utf-8"))
    ids = sorted(data, key=int)
    texts, owners = [], []
    for qid in ids:
        paras = data[qid]
        if isinstance(paras, str):
            paras = [paras]
        for t in paras:
            texts.append(t)
            owners.append(qid)

    print(f"embedding {len(texts)} paraphrases of {len(ids)} queries with {MODEL} ...")
    model = TextEmbedding(model_name=MODEL)
    vecs = np.array(list(model.embed(texts, batch_size=batch_size)), dtype=np.float32)

    out: dict = {qid: {"instructions": [], "embeddings": []} for qid in ids}
    for qid, t, v in zip(owners, texts, vecs):
        out[qid]["instructions"].append(t)
        out[qid]["embeddings"].append(v)
    for qid in ids:
        out[qid]["embeddings"] = np.vstack(out[qid]["embeddings"])
    return out


def check(new: dict, old_path: Path) -> int:
    with old_path.open("rb") as f:
        old = pickle.load(f)
    if set(map(str, old)) != set(map(str, new)):
        print(f"MISMATCH: query ids differ ({len(old)} vs {len(new)})")
        return 1
    worst, n_text_diff = 1.0, 0
    for qid in new:
        o = old.get(qid, old.get(int(qid)) if qid.isdigit() else None)
        if list(o["instructions"]) != list(new[qid]["instructions"]):
            n_text_diff += 1
            continue
        a = np.asarray(o["embeddings"], dtype=np.float32)
        b = new[qid]["embeddings"]
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        worst = min(worst, float(cos.min()))
    print(f"paraphrase lists differing: {n_text_diff}")
    print(f"lowest cosine between old and new vectors: {worst:.6f}")
    ok = n_text_diff == 0 and worst > 0.9999
    print("MATCH" if ok else "DIFFERENT -- retrieval results may not reproduce exactly")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instructions", type=Path, default=Path("instructions.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("embeddings_output/instructions_embeddings.pkl"))
    ap.add_argument("--check", type=Path, default=None,
                    help="existing index to compare against (nothing is written)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    index = build(args.instructions, args.batch_size)
    if args.check:
        return check(index, args.check)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(index, f)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
