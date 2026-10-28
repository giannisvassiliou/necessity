#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle and placebo control conditions (Wikidata).

ORACLE. Supply exactly the reference identifiers on every item. Every item then
has perfect candidate recall and no irrelevant candidates, so necessity under
the oracle shows how much of the reference the model still reproduces unaided.

PLACEBO. Supply the same NUMBER of identifiers as the deployed pipeline, drawn
from the reference queries of OTHER items (real Wikidata ids with real labels)
and excluding this item's reference and bare-arm identifiers. Necessity under
placebo estimates how much attribution arises from copying identifiers that are
unrelated to the question.

The public_bare arm is read from an existing exp1b_wired.py run (r_keep.csv)
and is not regenerated. Two generations per item.

    export ANTHROPIC_API_KEY=...
    python oracle_placebo.py --results r_keep.csv --out r_oracle_placebo.csv --resume

Must sit in the same directory as exp1b_wired.py, query.json and the caches.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

if sys.platform.startswith("win"):
    os.environ["PYTHONUTF8"] = "1"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import exp1b_wired as X   # shared helpers: execute, score, labels, throttle

RE_ID = re.compile(r"\b[QP]\d+\b")


# =============================================================================
# Divergence
# =============================================================================

def divergence(c_method: Set[str], c_bare: Set[str]) -> float:
    """delta: how far the method-arm query moved from the bare-arm query.

    Jaccard distance over identifier sets. 0 = identical, 1 = disjoint.
    """
    if not c_method and not c_bare:
        return 0.0
    u = c_method | c_bare
    if not u:
        return 0.0
    return 1.0 - len(c_method & c_bare) / len(u)


# =============================================================================
# Label fetching for both entities and properties
# =============================================================================

def fetch_labels(ids: Set[str], ctx) -> Dict[str, Dict[str, str]]:
    """{id: {'id','label','description'}} for Q- and P-ids. Batched and cached.

    X.fetch_property_labels only handles P-ids; entity labels are needed for the
    oracle context, so this generalises it over the same cache.
    """
    import requests

    want = {i for i in ids if RE_ID.fullmatch(i)}
    missing = sorted(want - set(ctx.prop_labels.keys()))
    for i in range(0, len(missing), 50):
        batch = missing[i:i + 50]
        params = {"action": "wbgetentities", "ids": "|".join(batch),
                  "format": "json", "props": "labels|descriptions", "languages": "en"}
        got = None
        for attempt in range(4):
            try:
                r = requests.get(X._WD_API, params=params,
                                 headers={"User-Agent": X.USER_AGENT}, timeout=30)
                if r.status_code == 429:
                    time.sleep(5 * (attempt + 1)); continue
                r.raise_for_status()
                got = r.json().get("entities", {})
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        if got is None:
            for p in batch:
                ctx.prop_labels.setdefault(p, None)
            continue
        for p in batch:
            e = got.get(p) or {}
            lbl = (e.get("labels", {}).get("en") or {}).get("value")
            dsc = (e.get("descriptions", {}).get("en") or {}).get("value")
            ctx.prop_labels[p] = ({"id": p, "label": lbl or "", "description": dsc or ""}
                                  if (lbl or dsc) else None)
    return {p: ctx.prop_labels[p] for p in want if ctx.prop_labels.get(p)}


# =============================================================================
# Context construction
# =============================================================================

def build_context(ids: Set[str], ctx, header: str) -> str:
    """Render a candidate block in the same shape as the real method context,
    so the only thing that differs between conditions is WHICH ids are listed."""
    info = fetch_labels(ids, ctx)
    lines = [header]
    for i in sorted(ids, key=lambda x: (x[0], int(x[1:]))):
        d = info.get(i)
        if d:
            lines.append(f"Found: {d['label']} ({i}) - {d['description']}")
        else:
            lines.append(f"Found: ({i})")
    return "\n".join(lines) + "\n"


def sample_placebo(n: int, forbidden: Set[str], pool: List[str],
                   rng: random.Random) -> Set[str]:
    """n real, well-formed, labelled ids that are unrelated to this question.

    Drawn from reference queries of OTHER items, so they are exactly as plausible
    as genuine candidates -- same namespace, same distribution, real labels --
    while carrying no information about this question.
    """
    out: Set[str] = set()
    tries = 0
    while len(out) < n and tries < n * 200:
        tries += 1
        c = rng.choice(pool)
        if c not in forbidden:
            out.add(c)
    return out


# =============================================================================
# Records
# =============================================================================

CSV_COLS = ["query_index", "condition", "n_candidates", "n_elements",
            "provision", "necessity", "redundancy", "memory",
            "gold_recall", "divergence", "correct", "f1", "executed",
            "necessary_ids", "redundant_ids", "memory_ids",
            "question", "sparql", "error"]


@dataclass
class Rec:
    query_index: str
    condition: str
    question: str = ""
    sparql: str = ""
    n_candidates: int = 0
    n_elements: int = 0
    provision: float = 0.0
    necessity: float = 0.0
    redundancy: float = 0.0
    memory: float = 0.0
    gold_recall: float = 0.0
    divergence: float = 0.0
    correct: bool = False
    f1: float = 0.0
    executed: bool = False
    necessary_ids: str = ""
    redundant_ids: str = ""
    memory_ids: str = ""
    error: str = ""


def to_row(r: Rec) -> Dict:
    d = r.__dict__.copy()
    for k in ("provision", "necessity", "redundancy", "memory",
              "gold_recall", "divergence", "f1"):
        d[k] = f"{d[k]:.4f}"
    d["correct"] = int(r.correct); d["executed"] = int(r.executed)
    return {c: d.get(c, "") for c in CSV_COLS}


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("r_keep.csv"),
                    help="existing run; public_bare arm is read from here")
    ap.add_argument("--queries", type=Path, default=Path("query.json"))
    ap.add_argument("--instructions", type=Path, default=Path("instructions.json"))
    ap.add_argument("--out", type=Path, default=Path("r_oracle_placebo.csv"))
    ap.add_argument("--cache", type=Path, default=Path("exp1b_sparql_cache.pkl"))
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--api-key-file", type=Path, default=None)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    key = args.api_key
    if not key and args.api_key_file:
        key = args.api_key_file.read_text(encoding="utf-8").strip()
    key = key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ERROR: no API key (env ANTHROPIC_API_KEY or --api-key-file)"); return 2
    os.environ["ANTHROPIC_API_KEY"] = key

    gold = json.loads(args.queries.read_text(encoding="utf-8"))
    prev = list(csv.DictReader(args.results.open(encoding="utf-8")))
    bare = {r["query_index"]: r for r in prev if r["condition"] == "public_bare"}
    meth = {r["query_index"]: r for r in prev if r["condition"] == "public_method"}
    items = [q for q in sorted(bare, key=int) if q in gold and q in meth]
    if args.limit:
        items = items[:args.limit]

    done = set()
    if args.resume and args.out.exists():
        done = {r["query_index"] for r in csv.DictReader(args.out.open(encoding="utf-8"))}
        items = [q for q in items if q not in done]
    print(f"{len(bare)} bare arms available; running {len(items)} items "
          f"(skipping {len(done)} done)")
    print(f"est. {len(items)*2} LLM calls, ~{len(items)*2} SPARQL executions")

    # placebo pool: every identifier appearing in any reference query
    pool = sorted({i for v in gold.values() for i in RE_ID.findall(v)})
    print(f"placebo pool: {len(pool)} distinct identifiers from reference queries")

    import anthropic, pickle
    X.install_global_api_throttle()
    ctx = X.PipelineHandles(model=args.model, label_mode="keep", cache_path=args.cache)
    ctx.client = anthropic.Anthropic()
    ctx.label_cache_path = args.cache.with_name(args.cache.stem + "_labels.pkl")
    for p, attr in ((args.cache, "exec_cache"), (ctx.label_cache_path, "prop_labels")):
        if p.exists():
            try: setattr(ctx, attr, pickle.load(p.open("rb")))
            except Exception: pass
    print(f"  loaded {len(ctx.exec_cache)} cached results, "
          f"{len(ctx.prop_labels)} cached labels")

    rng = random.Random(args.seed)
    new = not args.out.exists() or not args.resume
    fh = args.out.open("w" if new else "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=CSV_COLS)
    if new: w.writeheader()

    recs: List[Rec] = []
    try:
        for n, qi in enumerate(items, 1):
            q_text = bare[qi]["question"]
            gold_ids = X.only_ids(RE_ID.findall(gold[qi]))
            c_bare = X.extract_elements(bare[qi]["sparql"].replace(" / ", "\n"), False)
            gold_ans = X.execute_sparql(gold[qi], ctx, timeout=120)

            # ORACLE  -- exactly the reference identifiers, no noise
            # PLACEBO -- same count as the real run, unrelated ids
            n_real = int(meth[qi]["n_candidates"] or 0) or len(gold_ids)
            placebo_ids = sample_placebo(n_real, gold_ids | c_bare, pool, rng)

            conds = [("oracle", gold_ids,
                      "USE THESE IDENTIFIERS. They are the ones the query needs:"),
                     ("placebo", placebo_ids,
                      "GET THESE PREDICATES and their CODES AS GUIDANCE TO USE WITH SPARQL:")]

            print(f"\n[{n}/{len(items)}] Q{qi}: {q_text[:70]}")
            for cond, ids, header in conds:
                rec = Rec(query_index=qi, condition=cond, question=q_text,
                          n_candidates=len(ids))
                try:
                    block = build_context(ids, ctx, header)
                    sp = X.generate_sparql(q_text, block, ctx)
                    rec.sparql = sp
                    els = X.extract_elements(sp, False)
                    rec.n_elements = len(els)

                    a = X.compute_attribution(c_method=els, c_bare=c_bare,
                                              candidates=ids, c_gold=gold_ids)
                    rec.provision, rec.necessity = a.provision_rate, a.necessity_rate
                    rec.redundancy, rec.memory = a.redundancy_rate, a.memory_rate
                    rec.gold_recall = a.gold_recall
                    rec.necessary_ids = "|".join(a.necessary)
                    rec.redundant_ids = "|".join(a.redundant)
                    rec.memory_ids = "|".join(a.memory)
                    rec.divergence = divergence(els, c_bare)

                    ans = X.execute_sparql(sp, ctx, timeout=X.GEN_TIMEOUT,
                                           retries=1, cap_rows=True)
                    rec.executed = ans is not None
                    if ans is None and ctx.last_exec_error:
                        rec.error = f"exec: {ctx.last_exec_error[:120]}"
                    rec.correct, rec.f1, _, _ = X.score_answers(ans, gold_ans)
                except Exception as e:
                    rec.error = (rec.error + " | " if rec.error else "") + f"{type(e).__name__}: {e}"
                    print(f"  [!] {cond}: {e}")
                print(f"  [{cond:<8}] prov={rec.provision:5.3f} nec={rec.necessity:5.3f} "
                      f"red={rec.redundancy:5.3f} mem={rec.memory:5.3f} "
                      f"div={rec.divergence:5.3f} (|C|={rec.n_elements}, "
                      f"|Cand|={rec.n_candidates})")
                recs.append(rec); w.writerow(to_row(rec))
            fh.flush()
            if n % 25 == 0:
                X.save_cache(ctx)
                cost = (ctx.tok_in/1e6)*3 + (ctx.tok_out/1e6)*15
                print(f"  ... {ctx.n_llm_calls} calls, ${cost:.2f}")
            if args.sleep: time.sleep(args.sleep)
    finally:
        fh.close(); X.save_cache(ctx)

    # ---------------- summary ----------------
    import statistics as st, math
    def mean(v): return st.mean(v) if v else 0.0
    def ci(v):
        if len(v) < 2: return (0, 0)
        m = st.mean(v); s = st.stdev(v)/math.sqrt(len(v)); return (m-1.96*s, m+1.96*s)

    print("\n" + "=" * 96)
    print("ORACLE / PLACEBO SUMMARY")
    print("=" * 96)
    print(f"{'condition':<12}{'n':>5}{'|Cand|':>8}{'prov':>8}{'nec':>8}"
          f"{'[95% CI]':>18}{'red':>8}{'mem':>8}{'div':>8}{'F1':>8}")
    print("-" * 96)
    for cond in ("oracle", "placebo"):
        g = [r for r in recs if r.condition == cond and r.n_elements > 0]
        if not g: continue
        nv = [r.necessity for r in g]; lo, hi = ci(nv)
        print(f"{cond:<12}{len(g):>5}{mean([r.n_candidates for r in g]):>8.1f}"
              f"{mean([r.provision for r in g]):>8.3f}{mean(nv):>8.3f}"
              f"{'['+format(lo,'.3f')+','+format(hi,'.3f')+']':>18}"
              f"{mean([r.redundancy for r in g]):>8.3f}"
              f"{mean([r.memory for r in g]):>8.3f}"
              f"{mean([r.divergence for r in g]):>8.3f}"
              f"{mean([r.f1 for r in g]):>8.3f}")
    print("=" * 96)

    orc = [r for r in recs if r.condition == "oracle" and r.n_elements > 0]
    plc = [r for r in recs if r.condition == "placebo" and r.n_elements > 0]
    print("\nHOW TO READ THIS")
    if orc:
        print(f"  ORACLE necessity = {mean([r.necessity for r in orc]):.3f}")
        print("    Every item has perfect recall and no irrelevant candidates, so")
        print("    reference identifiers that are not necessary here were already")
        print("    produced by the bare arm, not missed by retrieval.")
    if plc and orc:
        dp, do = mean([r.divergence for r in plc]), mean([r.divergence for r in orc])
        print(f"\n  PLACEBO divergence = {dp:.3f}  vs  ORACLE divergence = {do:.3f}")
        if dp < do * 0.6:
            print("    Placebo divergence is clearly lower. Adding context is NOT by")
            print("    itself enough to move the output; divergence tracks genuine")
            print("    supply. This licenses the corr(nu, delta) argument.")
        else:
            print("    Placebo divergence is close to oracle. Adding ANY context moves")
            print("    the output, so divergence is not evidence of genuine supply and")
            print("    corr(nu, delta) is confounded.")
        print(f"\n  PLACEBO necessity = {mean([r.necessity for r in plc]):.3f}")
        print("    Near zero means the attribution rule rarely credits identifiers")
        print("    unrelated to the question.")

    print(f"\nWrote {len(recs)} rows -> {args.out}")
    cost = (ctx.tok_in/1e6)*3 + (ctx.tok_out/1e6)*15
    print(f"API: {ctx.n_llm_calls} calls, {ctx.tok_in:,} in / {ctx.tok_out:,} out, ${cost:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
