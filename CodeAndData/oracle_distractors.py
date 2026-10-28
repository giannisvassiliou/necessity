#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle + distractors and size-matched small placebo: a 2x2 over candidate sets.

The oracle condition differs from the deployed pipeline in two ways: every
candidate is a reference identifier, and the candidate set is much smaller.
This script adds the two missing cells so the factors can be separated.

Factors: whether the reference identifiers are PRESENT, and whether the
candidate set is SMALL (reference-sized) or LARGE (deployment-sized).

                      | small set             | large set
    ------------------+-----------------------+--------------------------
    gold present      | oracle                | oracle+distractors  (here)
    gold absent       | placebo_small (here)  | placebo

    The deployed pipeline sits off-grid: large set, gold partially present.

Comparisons:
    oracle             vs oracle+distractors   -> set size, recall held perfect
    oracle+distractors vs placebo              -> presence of reference ids, size matched
    placebo            vs placebo_small        -> set size, no reference ids

    python oracle_distractors.py --results r_keep.csv \
        --oracle r_oracle_placebo.csv --out r_distractors.csv --resume

Requires exp1b_wired.py, oracle_placebo.py, query.json and the caches in the
same directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import re
import statistics as st
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

if sys.platform.startswith("win"):
    os.environ["PYTHONUTF8"] = "1"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import exp1b_wired as X
import oracle_placebo as OP     # reuse fetch_labels, build_context, divergence

RE_ID = re.compile(r"\b[QP]\d+\b")

CSV_COLS = ["query_index", "condition", "n_candidates", "n_gold", "n_distractors",
            "n_elements", "provision", "necessity", "redundancy", "memory",
            "gold_recall", "divergence", "correct", "f1", "executed",
            "necessary_ids", "redundant_ids", "memory_ids",
            "question", "sparql", "error"]


@dataclass
class Rec:
    query_index: str = ""
    condition: str = ""
    question: str = ""
    sparql: str = ""
    n_candidates: int = 0
    n_gold: int = 0
    n_distractors: int = 0
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("r_keep.csv"))
    ap.add_argument("--oracle", type=Path, default=Path("r_oracle_placebo.csv"),
                    help="existing oracle/placebo run, used for the summary only")
    ap.add_argument("--queries", type=Path, default=Path("query.json"))
    ap.add_argument("--out", type=Path, default=Path("r_distractors.csv"))
    ap.add_argument("--cache", type=Path, default=Path("exp1b_sparql_cache.pkl"))
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--api-key-file", type=Path, default=None)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    # ---- validate inputs BEFORE spending anything ----
    for lab, pth in (("--results", args.results), ("--queries", args.queries),
                     ("--oracle", args.oracle)):
        if not pth.exists():
            print(f"ERROR: {lab} file not found: {pth}")
            print("  CSV files in this directory:")
            for f in sorted(Path(".").glob("*.csv")):
                print(f"    {f}")
            return 2
    _have = {r["condition"] for r in csv.DictReader(args.oracle.open(encoding="utf-8"))}
    _missing = {"oracle", "placebo"} - _have
    if _missing:
        print(f"ERROR: {args.oracle} has no rows for {sorted(_missing)}; "
              f"found {sorted(_have)}")
        return 2

    key = args.api_key
    if not key and args.api_key_file:
        key = args.api_key_file.read_text(encoding="utf-8").strip()
    key = key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ERROR: no API key"); return 2
    os.environ["ANTHROPIC_API_KEY"] = key

    gold = json.loads(args.queries.read_text(encoding="utf-8"))
    prev = list(csv.DictReader(args.results.open(encoding="utf-8")))
    bare = {r["query_index"]: r for r in prev if r["condition"] == "public_bare"}
    meth = {r["query_index"]: r for r in prev if r["condition"] == "public_method"}
    items = [q for q in sorted(bare, key=int) if q in gold and q in meth]
    if args.limit: items = items[:args.limit]

    done = set()
    if args.resume and args.out.exists():
        done = {r["query_index"] for r in csv.DictReader(args.out.open(encoding="utf-8"))}
        items = [q for q in items if q not in done]
    print(f"running {len(items)} items (skipping {len(done)} done)")
    print(f"est. {len(items)*2} LLM calls")

    pool = sorted({i for v in gold.values() for i in RE_ID.findall(v)})
    print(f"distractor pool: {len(pool)} identifiers from reference queries")

    import anthropic
    X.install_global_api_throttle()
    ctx = X.PipelineHandles(model=args.model, label_mode="keep", cache_path=args.cache)
    ctx.client = anthropic.Anthropic()
    ctx.label_cache_path = args.cache.with_name(args.cache.stem + "_labels.pkl")
    for p, attr in ((args.cache, "exec_cache"), (ctx.label_cache_path, "prop_labels")):
        if p.exists():
            try: setattr(ctx, attr, pickle.load(p.open("rb")))
            except Exception: pass
    print(f"  loaded {len(ctx.exec_cache)} results, {len(ctx.prop_labels)} labels")

    rng = random.Random(args.seed)
    new = not args.out.exists() or not args.resume
    fh = args.out.open("w" if new else "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=CSV_COLS); 
    if new: w.writeheader()

    HDR = "GET THESE PREDICATES and their CODES AS GUIDANCE TO USE WITH SPARQL:"
    recs: List[Rec] = []
    try:
        for n, qi in enumerate(items, 1):
            q_text = bare[qi]["question"]
            g_ids = X.only_ids(RE_ID.findall(gold[qi]))
            c_bare = X.extract_elements(bare[qi]["sparql"].replace(" / ", "\n"), False)
            gold_ans = X.execute_sparql(gold[qi], ctx, timeout=120)
            n_deployed = int(meth[qi]["n_candidates"] or 0) or 40

            # ORACLE + DISTRACTORS: all gold, padded to deployment size
            n_pad = max(0, n_deployed - len(g_ids))
            distr = OP.sample_placebo(n_pad, g_ids | c_bare, pool, rng)
            od_ids = g_ids | distr

            # PLACEBO SMALL: no gold, sized like the oracle
            ps_ids = OP.sample_placebo(max(1, len(g_ids)), g_ids | c_bare, pool, rng)

            print(f"\n[{n}/{len(items)}] Q{qi}: {q_text[:66]}")
            for cond, ids, ng, nd in (("oracle_distractors", od_ids, len(g_ids), len(distr)),
                                      ("placebo_small",      ps_ids, 0,          len(ps_ids))):
                rec = Rec(query_index=qi, condition=cond, question=q_text,
                          n_candidates=len(ids), n_gold=ng, n_distractors=nd)
                try:
                    block = OP.build_context(ids, ctx, HDR)
                    sp = X.generate_sparql(q_text, block, ctx)
                    rec.sparql = sp
                    els = X.extract_elements(sp, False)
                    rec.n_elements = len(els)
                    a = X.compute_attribution(c_method=els, c_bare=c_bare,
                                              candidates=ids, c_gold=g_ids)
                    rec.provision, rec.necessity = a.provision_rate, a.necessity_rate
                    rec.redundancy, rec.memory = a.redundancy_rate, a.memory_rate
                    rec.gold_recall = a.gold_recall
                    rec.necessary_ids = "|".join(a.necessary)
                    rec.redundant_ids = "|".join(a.redundant)
                    rec.memory_ids = "|".join(a.memory)
                    rec.divergence = OP.divergence(els, c_bare)
                    ans = X.execute_sparql(sp, ctx, timeout=X.GEN_TIMEOUT,
                                           retries=1, cap_rows=True)
                    rec.executed = ans is not None
                    if ans is None and ctx.last_exec_error:
                        rec.error = f"exec: {ctx.last_exec_error[:120]}"
                    rec.correct, rec.f1, _, _ = X.score_answers(ans, gold_ans)
                except Exception as e:
                    rec.error = f"{type(e).__name__}: {e}"
                    print(f"  [!] {cond}: {e}")
                print(f"  [{cond:<18}] prov={rec.provision:5.3f} nec={rec.necessity:5.3f} "
                      f"red={rec.redundancy:5.3f} F1={rec.f1:5.3f} "
                      f"(|Cand|={rec.n_candidates}: {rec.n_gold} gold + {rec.n_distractors} distr)")
                recs.append(rec); w.writerow(to_row(rec))
            fh.flush()
            if n % 25 == 0:
                X.save_cache(ctx)
                print(f"  ... {ctx.n_llm_calls} calls, "
                      f"${(ctx.tok_in/1e6)*3+(ctx.tok_out/1e6)*15:.2f}")
            if args.sleep: time.sleep(args.sleep)
    finally:
        fh.close(); X.save_cache(ctx)

    # ---------------- 2x2 summary ----------------
    def agg(rows, key):
        v = [getattr(r, key) for r in rows if r.n_elements > 0]
        return st.mean(v) if v else 0.0
    def f1m(rows):
        return st.mean([r.f1 for r in rows]) if rows else 0.0

    old = list(csv.DictReader(args.oracle.open(encoding='utf-8')))
    def old_rows(c):
        out = []
        for r in old:
            if r["condition"] != c: continue
            k = Rec(query_index=r["query_index"], condition=c)
            k.n_candidates = int(r["n_candidates"] or 0)
            k.n_elements = int(r["n_elements"] or 0)
            for f in ("provision","necessity","redundancy","memory","divergence","f1"):
                try: setattr(k, f, float(r[f]))
                except Exception: pass
            out.append(k)
        return out

    orc_all, plc_all = old_rows("oracle"), old_rows("placebo")
    od = [r for r in recs if r.condition == "oracle_distractors"]
    ps = [r for r in recs if r.condition == "placebo_small"]

    # CRITICAL: restrict every condition to the SAME items. Comparing new
    # conditions on a partial run against old conditions on the full run
    # confounds the effect with item difficulty -- e.g. a 5-item pilot whose
    # items happen to be easy will show a large spurious "set size" effect.
    common = ({r.query_index for r in od} & {r.query_index for r in ps}
              & {r.query_index for r in orc_all} & {r.query_index for r in plc_all})
    keep = lambda rows: [r for r in rows if r.query_index in common]
    orc, plc, od, ps = keep(orc_all), keep(plc_all), keep(od), keep(ps)
    print(f"\n[summary restricted to {len(common)} items present in all four conditions]")
    if len(common) < 30:
        print("[WARNING: fewer than 30 common items -- the decomposition below is")
        print(" NOT interpretable yet. Finish the run before reading it.]")

    print("\n" + "=" * 100)
    print("2x2: presence of reference identifiers  x  candidate-set size")
    print("=" * 100)
    print(f"{'condition':<22}{'gold?':>7}{'|Cand|':>9}{'prov':>8}{'nec':>8}{'red':>8}{'F1':>8}")
    print("-" * 100)
    for lab, rows, g in (("oracle", orc, "yes"), ("oracle+distractors", od, "yes"),
                         ("placebo_small", ps, "no"), ("placebo", plc, "no")):
        if not rows: continue
        print(f"{lab:<22}{g:>7}{agg(rows,'n_candidates'):>9.1f}{agg(rows,'provision'):>8.3f}"
              f"{agg(rows,'necessity'):>8.3f}{agg(rows,'redundancy'):>8.3f}{f1m(rows):>8.3f}")
    print("=" * 100)

    print("\nDECOMPOSITION")
    if orc and od:
        print(f"  SET SIZE   (oracle -> oracle+distractors, recall held perfect)")
        print(f"     F1 {f1m(orc):.3f} -> {f1m(od):.3f}   ({f1m(od)-f1m(orc):+.3f})")
        print(f"     nec {agg(orc,'necessity'):.3f} -> {agg(od,'necessity'):.3f}")
    if od and plc:
        print(f"  CORRECTNESS (placebo -> oracle+distractors, size held ~equal)")
        print(f"     F1 {f1m(plc):.3f} -> {f1m(od):.3f}   ({f1m(od)-f1m(plc):+.3f})")
    if od:
        print(f"  RECALL     (deployed -> oracle+distractors, size held ~equal)")
        print(f"     compare with the deployed pipeline's F1 and necessity from")
        print(f"     --results; both conditions present "
              f"~{agg(od,'n_candidates'):.0f} candidates.")

    print(f"\nWrote {len(recs)} rows -> {args.out}")
    print(f"API: {ctx.n_llm_calls} calls, "
          f"${(ctx.tok_in/1e6)*3+(ctx.tok_out/1e6)*15:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
