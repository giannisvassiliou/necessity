#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Necessity / provision attribution for retrieval-augmented SPARQL generation
(Wikidata).

For each question the same model generates two queries: a bare arm (question
only) and a method arm (question plus retrieved context). The identifiers in
the method-arm query are partitioned against the retrieved candidates and the
bare-arm query:

    N necessary  = in candidates, NOT in the bare-arm query
    R redundant  = in candidates, ALSO in the bare-arm query
    L memory     = NOT in candidates
    provision = (N+R)/|C|   necessity = N/|C|   redundancy = R/|C|

Each question is also run with per-question anonymised identifiers
(public / anonymised x bare / method).

The question is rewritten once into Wikidata-oriented form before retrieval.
By default (--rephrase both) both arms receive the same rewrite, so they differ
only in whether retrieved context is present.

    export ANTHROPIC_API_KEY=...
    python exp1b_wired.py --split-filter test_indices.json --use-gold-filter \
        --limit 0 --label-mode keep --out r_keep.csv

Requires utilLookup.py, utilSem2.py, utilsPre.py and utilExpre.py in the same
directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import re
import string
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

if sys.platform.startswith("win"):
    os.environ["PYTHONUTF8"] = "1"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "necessity-eval/1.0 (anonymous@example.org)"  # set a real contact before running

ANON_SCOPE_CHOICES = ("per_query", "per_run")
LABEL_MODE_CHOICES = ("keep", "nolabels", "strip")
REPHRASE_CHOICES = ("none", "both", "method")

RE_REAL_ENTITY = re.compile(r"\bQ\d+\b")
RE_REAL_PROPERTY = re.compile(r"\bP\d+\b")
RE_ANON_ENTITY = re.compile(r"\bE[A-Za-z0-9]{7}\b")
RE_ANON_PROPERTY = re.compile(r"\bR[A-Za-z0-9]{7}\b")
RE_ANY_REAL_ID = re.compile(r"\b[QP]\d+\b")

_TOKEN_ALPHABET = string.ascii_lowercase + string.digits


# =============================================================================
# Anonymizer
# =============================================================================

class Anonymizer:
    def __init__(self, seed: int = 0, scope: str = "per_query"):
        if scope not in ANON_SCOPE_CHOICES:
            raise ValueError(f"scope must be one of {ANON_SCOPE_CHOICES}")
        self.scope = scope
        self.base_seed = seed
        self._rng = random.Random(seed)
        self.fwd: Dict[str, str] = {}
        self.rev: Dict[str, str] = {}

    def reset_for_query(self, query_index: int) -> None:
        if self.scope == "per_query":
            self.fwd.clear()
            self.rev.clear()
            self._rng = random.Random(self.base_seed * 100003 + query_index)

    def _mint(self, prefix: str) -> str:
        while True:
            tok = prefix + "".join(self._rng.choice(_TOKEN_ALPHABET) for _ in range(7))
            if tok not in self.rev:
                return tok

    def token_for(self, real_id: str) -> str:
        if real_id in self.fwd:
            return self.fwd[real_id]
        if RE_REAL_ENTITY.fullmatch(real_id):
            tok = self._mint("E")
        elif RE_REAL_PROPERTY.fullmatch(real_id):
            tok = self._mint("R")
        else:
            return real_id
        self.fwd[real_id] = tok
        self.rev[tok] = real_id
        return tok

    def anonymize_text(self, text: str) -> str:
        if not text:
            return text
        return RE_ANY_REAL_ID.sub(lambda m: self.token_for(m.group(0)), text)

    def deanonymize_text(self, text: str) -> str:
        if not text:
            return text
        text = RE_ANON_ENTITY.sub(lambda m: self.rev.get(m.group(0), m.group(0)), text)
        text = RE_ANON_PROPERTY.sub(lambda m: self.rev.get(m.group(0), m.group(0)), text)
        return text

    def anonymize_set(self, ids: Set[str]) -> Set[str]:
        return {self.token_for(i) for i in ids}


def extract_elements(sparql: str, anonymized: bool) -> Set[str]:
    if not sparql:
        return set()
    if anonymized:
        return set(RE_ANON_ENTITY.findall(sparql)) | set(RE_ANON_PROPERTY.findall(sparql))
    return set(RE_REAL_ENTITY.findall(sparql)) | set(RE_REAL_PROPERTY.findall(sparql))


def only_ids(items) -> Set[str]:
    """Keep only well-formed Q<n>/P<n>; drops placeholder values such as 'none'."""
    out = set()
    for x in items or []:
        if isinstance(x, str) and RE_ANY_REAL_ID.fullmatch(x.strip()):
            out.add(x.strip())
    return out


# =============================================================================
# Attribution
# =============================================================================

@dataclass
class Attribution:
    n_elements: int = 0
    necessary: List[str] = field(default_factory=list)
    redundant: List[str] = field(default_factory=list)
    memory: List[str] = field(default_factory=list)
    provision_rate: float = 0.0
    necessity_rate: float = 0.0
    redundancy_rate: float = 0.0
    memory_rate: float = 0.0
    gold_recall: float = 0.0

    def summary_line(self) -> str:
        return (f"prov={self.provision_rate:6.3f} nec={self.necessity_rate:6.3f} "
                f"red={self.redundancy_rate:6.3f} mem={self.memory_rate:6.3f} "
                f"goldrec={self.gold_recall:6.3f} (|C|={self.n_elements})")


def compute_attribution(c_method: Set[str], c_bare: Set[str],
                        candidates: Set[str], c_gold: Set[str]) -> Attribution:
    a = Attribution()
    a.n_elements = len(c_method)
    for c in sorted(c_method):
        if c in candidates:
            (a.redundant if c in c_bare else a.necessary).append(c)
        else:
            a.memory.append(c)
    if a.n_elements:
        d = float(a.n_elements)
        a.necessity_rate = len(a.necessary) / d
        a.redundancy_rate = len(a.redundant) / d
        a.memory_rate = len(a.memory) / d
        a.provision_rate = a.necessity_rate + a.redundancy_rate
    if c_gold:
        a.gold_recall = len(candidates & c_gold) / float(len(c_gold))
    assert abs(a.provision_rate - (a.necessity_rate + a.redundancy_rate)) < 1e-9
    return a


# =============================================================================
# EmbeddingSearcher  (example retrieval over the paraphrase index)
# =============================================================================

class EmbeddingSearcher:
    def __init__(self, embeddings_path: str, sparql_json_path: str = None,
                 model_name: str = "BAAI/bge-large-en-v1.5"):
        from fastembed import TextEmbedding
        print(f"  loading embedding model {model_name} ...")
        self.model = TextEmbedding(model_name=model_name)
        with open(embeddings_path, "rb") as f:
            self.data = pickle.load(f)
        self.sparql_queries = {}
        if sparql_json_path:
            with open(sparql_json_path, "r", encoding="utf-8") as f:
                self.sparql_queries = json.load(f)
        self.all_embeddings, self.all_metadata = [], []
        for query_id, d in self.data.items():
            for i, (instruction, emb) in enumerate(zip(d["instructions"], d["embeddings"])):
                self.all_embeddings.append(emb)
                self.all_metadata.append({"query_id": query_id, "instruction": instruction, "index": i})
        self.all_embeddings = np.array(self.all_embeddings)
        print(f"  index {self.all_embeddings.shape}, {len(self.sparql_queries)} sparql")

    def get_sparql_by_id(self, query_id: str):
        return self.sparql_queries.get(query_id, None)

    def search(self, query_text: str, top_k: int = 5) -> List[Dict]:
        qe = list(self.model.embed([query_text]))[0]
        sims = np.dot(self.all_embeddings, qe) / (
            np.linalg.norm(self.all_embeddings, axis=1) * np.linalg.norm(qe))
        idx = np.argsort(sims)[-top_k:][::-1]
        return [{"instruction": self.all_metadata[i]["instruction"],
                 "query_id": self.all_metadata[i]["query_id"],
                 "similarity": float(sims[i]),
                 "index": self.all_metadata[i]["index"]} for i in idx]

    def search_with_sparql(self, query_text: str, top_k: int = 5) -> List[Dict]:
        res = self.search(query_text, top_k)
        for r in res:
            r["sparql"] = self.get_sparql_by_id(r["query_id"])
        return res


# =============================================================================
# Pipeline handles
# =============================================================================

@dataclass
class PipelineHandles:
    nlp: Any = None
    lookup: Any = None
    searcher: Any = None
    client: Any = None
    args: Any = None
    model: str = "claude-sonnet-4-5-20250929"
    label_mode: str = "keep"
    prop_df: Any = None
    prop_vecs: Any = None
    exec_cache: Dict[str, Any] = field(default_factory=dict)
    cache_path: Optional[Path] = None
    n_llm_calls: int = 0
    tok_in: int = 0
    tok_out: int = 0
    n_sparql_calls: int = 0
    n_parse_failures: int = 0
    n_label_failures: int = 0
    n_self_retrievals: int = 0
    n_dup_retrievals: int = 0
    n_heldout_retrievals: int = 0
    retrieval_exclude: Any = field(default_factory=set)
    n_gold_failures: int = 0
    gold_fail_reasons: Any = field(default_factory=dict)
    last_exec_error: str = ""
    exec_errors: Dict[str, str] = field(default_factory=dict)
    prop_labels: Dict[str, Any] = field(default_factory=dict)
    label_cache_path: Optional[Path] = None


def install_global_api_throttle(min_interval: float = 0.4) -> None:
    """utilLookup/utilSem2 call wbsearchentities with no throttle and no cache,
    which triggers HTTP 429. We cannot edit those modules, so we wrap
    requests.get once, globally, and pace every wikidata.org call."""
    import requests
    if getattr(requests, "_exp1b_throttled", False):
        return
    orig = requests.get
    state = {"last": 0.0}

    def throttled_get(url, **kw):
        if "wikidata.org" in str(url):
            dt = time.time() - state["last"]
            if dt < min_interval:
                time.sleep(min_interval - dt)
            state["last"] = time.time()
            kw.setdefault("headers", {})
            kw["headers"].setdefault("User-Agent", USER_AGENT)
            for attempt in range(4):
                r = orig(url, **kw)
                if r.status_code != 429:
                    return r
                time.sleep(4 * (attempt + 1))
            return r
        return orig(url, **kw)

    requests.get = throttled_get
    requests._exp1b_throttled = True


def init_pipeline(model: str, label_mode: str, cache_path: Path) -> PipelineHandles:
    import spacy
    import anthropic
    import utilLookup
    from types import SimpleNamespace

    install_global_api_throttle()
    ctx = PipelineHandles(model=model, label_mode=label_mode, cache_path=cache_path)

    print("  loading spaCy en_core_web_lg ...")
    ctx.nlp = spacy.load("en_core_web_lg")
    ctx.lookup = utilLookup.WikidataPropertyLookup()
    ctx.searcher = EmbeddingSearcher(
        embeddings_path="embeddings_output/instructions_embeddings.pkl",
        sparql_json_path="query.json")
    ctx.client = anthropic.Anthropic()

    a = SimpleNamespace()
    a.topk, a.lang, a.no_progress = 5, "en", True
    a.device, a.model = "auto", "sentence-transformers/all-MiniLM-L6-v2"
    a.refresh_cache = False
    ctx.args = a

    # Property table + embeddings: built once per run, not per question.
    import utilsPre
    print("  fetching + embedding Wikidata properties (once) ...")
    ctx.prop_df = utilsPre.fetch_properties(language=a.lang)
    ctx.prop_vecs = utilsPre.embed_properties(
        df=ctx.prop_df, model_name=a.model, language=a.lang, device=a.device,
        batch_size=256, show_progress=False, force_rebuild=False)

    ctx.label_cache_path = cache_path.with_name(cache_path.stem + "_labels.pkl")
    if ctx.label_cache_path.exists():
        try:
            with ctx.label_cache_path.open("rb") as f:
                ctx.prop_labels = pickle.load(f)
            print(f"  loaded {len(ctx.prop_labels)} cached property labels")
        except Exception:
            ctx.prop_labels = {}

    if cache_path and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                ctx.exec_cache = pickle.load(f)
            print(f"  loaded {len(ctx.exec_cache)} cached SPARQL results")
        except Exception:
            ctx.exec_cache = {}
    return ctx


def save_cache(ctx: PipelineHandles) -> None:
    if ctx.cache_path:
        try:
            with ctx.cache_path.open("wb") as f:
                pickle.dump(ctx.exec_cache, f)
        except Exception as e:
            print(f"  [!] sparql cache save failed: {e}")
    if ctx.label_cache_path:
        try:
            with ctx.label_cache_path.open("wb") as f:
                pickle.dump(ctx.prop_labels, f)
        except Exception as e:
            print(f"  [!] label cache save failed: {e}")


# =============================================================================
# 1. Rephrase
# =============================================================================

def rephrase_question(question: str, ctx: PipelineHandles) -> str:
    try:
        msg = ctx.client.messages.create(
            model=ctx.model, max_tokens=1000, temperature=0,
            system="You are a SPARQL EXPERT",
            messages=[{"role": "user", "content": [{"type": "text", "text":
                "Rephrase this question :" + str(question) +
                " so that is wikidata friendly for  sparql to be created afterwards, "
                "targeting the wikidata Knowledge Graph give bare question without "
                "titles/comments or prologue and keep the meaning intact"}]}])
        ctx.n_llm_calls += 1
        u = getattr(msg, 'usage', None)
        if u:
            ctx.tok_in += getattr(u, 'input_tokens', 0)
            ctx.tok_out += getattr(u, 'output_tokens', 0)
        return msg.content[0].text.strip()
    except Exception as e:
        print(f"  [!] rephrase failed ({e}); using original")
        return question


# =============================================================================
# 2. build_method_context
# =============================================================================

def normalize_query(q: str) -> str:
    """Comment-stripped, whitespace-collapsed, lowercased form -- used to detect
    duplicate gold queries that share content under different ids."""
    q = re.sub(r"#[^\n]*", "", q or "")
    return re.sub(r"\s+", " ", q).strip().lower()


def build_method_context(question: str, ctx: PipelineHandles,
                        exclude_qid: Optional[str] = None,
                        exclude_norm: Optional[str] = None,
                        exclude_ids: Optional[Set[str]] = None) -> Tuple[Set[str], str]:
    """Returns (candidates, context). Candidates = entity IDs + predicate IDs
    surfaced by ANY method component: property lookup, entity linking,
    embedding property search, and predicates/entities inside retrieved examples."""
    import utilLookup
    import utilSem2
    import utilsPre
    import utilExpre

    predset: Set[str] = set()
    entset: Set[str] = set()
    accol = "GET THESE PREDICATES and their CODES AS GUIDANCE TO USE WITH SPARQL: "
    listamen: List[str] = []
    pc1 = question
    pc2 = question

    # --- (a) NOUN/VERB phrase -> property lookup --------------------------
    doc = ctx.nlp(pc1)
    for token in doc:
        if token.pos_ in ("NOUN", "VERB"):
            phrase = [token.text] + [c.text for c in token.children if c.dep_ == "prep"]
            term = " ".join(phrase)
            if not term:
                continue
            try:
                results = ctx.lookup.search_property_id(term, limit=5)
                mata, sata = utilLookup.print2_search_results(results)
                accol += mata
                predset.update(only_ids(sata))
                listamen.append(term)
            except Exception:
                pass

    # --- (b) NER -> entity linking ----------------------------------------
    common_entities = {"first", "second"}
    lista = [e.text for e in ctx.nlp(pc1).ents if e.text not in common_entities]
    for l in lista:
        try:
            results = utilSem2.get_wikidata_id(l, limit=10)
            if not results:
                continue
            ranked = utilSem2.calculate_combined_similarity(
                l, pc1, results,
                label_weight=0.2, semantic_weight=0.5, label_phrase_weight=0.3)
            for item in ranked:
                pc2 = pc2.replace(l, item["id"])
                entset.update(only_ids([item["id"]]))
                desc = item.get("description")
                accol += (f"Found: {item['label']} ({item['id']}) - {desc}"
                          if desc else f"Found: {item['label']} ({item['id']}) ")
                break
        except Exception:
            pass

    # --- (c) embedding property search ------------------------------------
    if listamen:
        try:
            q_vecs = utilsPre.embed_queries(
                terms=listamen, model_name=ctx.args.model, device=ctx.args.device,
                batch_size=256, show_progress=False)
            results = utilsPre.search(listamen, ctx.prop_df, ctx.prop_vecs,
                                      q_vecs, topk=ctx.args.topk)
            for term in listamen:
                for r in results.get(term, []):
                    predset.update(only_ids([r["pid"]]))
                    accol += (f"\n{r['score']:.4f}  {r['pid']:>5}  {r['label']}"
                              f"  -  {r['description']}  ({r['url']})")
        except Exception as e:
            print(f"  [!] embedding property search failed: {e}")

    # --- (d) retrieved examples -------------------------------------------
    accumula = (" This is the user original query [" + pc1 + "] This is with ids "
                "replaced. which you must use the ids from this:[" + pc2 + "]  "
                "Consult the following similar queries and its sparql")
    labelplusdescr: List[str] = []
    # LEAVE-ONE-OUT. The index contains all 5 NL paraphrases of every query in
    # the dataset, including the one being asked. Without this filter the nearest
    # neighbour is another paraphrase of the SAME question and its gold SPARQL
    # goes straight into the prompt -- the method is handed the answer, and
    # provision/gold_recall saturate at 1.0 for reasons that have nothing to do
    # with retrieval quality.
    try:
        raw = ctx.searcher.search_with_sparql(pc1, top_k=3 + 20)
        if exclude_ids:
            # Retrieve from TRAIN only. The embedding index covers all 2838 rows,
            # so without this a test question can retrieve another test question.
            before = len(raw)
            raw = [r for r in raw if str(r.get("query_id")) not in exclude_ids]
            if len(raw) < before:
                ctx.n_heldout_retrievals += (before - len(raw))
        if exclude_qid is not None:
            if any(str(r.get("query_id")) == str(exclude_qid) for r in raw):
                ctx.n_self_retrievals += 1
            raw = [r for r in raw if str(r.get("query_id")) != str(exclude_qid)]
        if exclude_norm:
            # 78 gold queries in this dataset appear under 2-4 different ids.
            # Id-based leave-one-out does NOT catch those: a different id can
            # carry a byte-identical gold query. Filter on content as well.
            before = len(raw)
            raw = [r for r in raw
                   if normalize_query(r.get("sparql") or "") != exclude_norm]
            if len(raw) < before:
                ctx.n_dup_retrievals += 1
        results6 = raw[:3]
    except Exception as e:
        print(f"  [!] example retrieval failed: {e}")
        results6 = []

    for result in results6:
        spq = result.get("sparql") or ""
        accumula += (f"\n qid:{result['query_id']}  DESCRIPTION:{result['instruction']}"
                     f" sparql:{spq}\n the predicates for this query are described here: ")

        # Parser-independent harvest. utilExpre uses rdflib and dies on unknown
        # prefixes (geof:, bd:, wikibase: ...). Anything appearing in the example
        # is in the prompt, so it is method-provided by definition -- harvest it
        # by regex regardless of whether the parser succeeds.
        raw_ids = only_ids(RE_ANY_REAL_ID.findall(spq))
        for rid in raw_ids:
            (entset if rid.startswith("Q") else predset).add(rid)

        parsed_ok = True
        try:
            ents = utilExpre.extract_entities_from_query(spq)
            entlist = (list(ents.get("subject_uris", [])) +
                       list(ents.get("object_uris", [])) +
                       list(ents.get("object_literals", [])))
            for pred in entlist:
                fp = str(pred).rsplit("/", 1)[-1]
                if "#" not in fp:
                    # Entities in retrieved examples are method-provided candidates.
                    entset.update(only_ids([fp]))
        except Exception:
            parsed_ok = False

        try:
            preds = utilExpre.extract_predicates_from_query(spq) or []
        except Exception:
            parsed_ok = False
            preds = []
        for pred in preds:
            fp = str(pred).rsplit("/", 1)[-1]
            if "#" in fp:
                continue
            predset.update(only_ids([fp]))

        if not parsed_ok:
            ctx.n_parse_failures += 1

        # Property labels for the prompt: ONE batched, cached call instead of
        # one uncached wbgetentities request per property (the 429 source).
        wanted = sorted(p for p in raw_ids if p.startswith("P"))
        for fp, stan in sorted(fetch_property_labels(set(wanted), ctx).items()):
            s1 = f"{stan['id']}:label: {stan['label']}description: {stan['description']}"
            if s1 not in labelplusdescr:
                labelplusdescr.append(s1)
                accumula += s1 + "\n"

    accumula += accol + " "
    candidates = only_ids(predset | entset)
    return candidates, accumula


# =============================================================================
# Property label fetching -- batched + cached + throttled
# =============================================================================
# utilLookup.get_property_name() issues one wbgetentities call per property with
# no cache and no User-Agent, which is what produces the 429 storm. We fetch in
# batches of 50, cache across the whole run, and back off on throttling.

_WD_API = "https://www.wikidata.org/w/api.php"
_last_api_call = [0.0]
_MIN_API_INTERVAL = 0.35   # seconds between calls to the Wikidata Action API


def _throttle() -> None:
    dt = time.time() - _last_api_call[0]
    if dt < _MIN_API_INTERVAL:
        time.sleep(_MIN_API_INTERVAL - dt)
    _last_api_call[0] = time.time()


def fetch_property_labels(pids: Set[str], ctx: PipelineHandles) -> Dict[str, Dict[str, str]]:
    """Return {pid: {'id','label','description'}} for the given P-ids.

    Missing/failed lookups are cached as None so we never retry them in a loop.
    """
    import requests

    want = {p for p in pids if p.startswith("P")}
    missing = sorted(want - set(ctx.prop_labels.keys()))

    for i in range(0, len(missing), 50):
        batch = missing[i:i + 50]
        params = {"action": "wbgetentities", "ids": "|".join(batch),
                  "format": "json", "props": "labels|descriptions",
                  "languages": "en"}
        got = None
        for attempt in range(4):
            try:
                _throttle()
                r = requests.get(_WD_API, params=params,
                                 headers={"User-Agent": USER_AGENT}, timeout=30)
                if r.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                r.raise_for_status()
                got = r.json().get("entities", {})
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        if got is None:
            for p in batch:
                ctx.prop_labels.setdefault(p, None)
            ctx.n_label_failures += len(batch)
            continue
        for p in batch:
            e = got.get(p) or {}
            lbl = (e.get("labels", {}).get("en") or {}).get("value")
            dsc = (e.get("descriptions", {}).get("en") or {}).get("value")
            ctx.prop_labels[p] = ({"id": p, "label": lbl or "", "description": dsc or ""}
                                  if lbl or dsc else None)

    return {p: ctx.prop_labels[p] for p in want if ctx.prop_labels.get(p)}




SYSTEM_PROMPT = "You are a SPARQL EXPERT"

BASE_INSTRUCTION = (
    "Produce a bare SPARQL query for the Wikidata Query Service that answers the "
    "question. Use LIMIT / ORDER BY / GROUP BY where the question requires them. "
    "Output ONLY the query. No prose, no markdown fences, no prologue, no headers."
)


def build_prompt(question: str, context: Optional[str], label_mode: str) -> str:
    if context is None:
        return f"{BASE_INSTRUCTION}\n\nQuestion: {question}\n"
    hint = ""
    if label_mode == "strip":
        hint = ("\nNOTE: schema identifiers here are opaque. Do not recall identifiers "
                "from prior knowledge; use only those provided in the context.\n")
    return (f"{BASE_INSTRUCTION}{hint}\n\nQuestion: {question}\n\n"
            f"--- Retrieved context ---\n{context}\n--- end context ---\n")


def clean_sparql(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"^\s*```(?:sparql)?\s*", "", text.strip(), flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def generate_sparql(question: str, context: Optional[str], ctx: PipelineHandles) -> str:
    prompt = build_prompt(question, context, ctx.label_mode)
    last = None
    for attempt in range(4):
        try:
            msg = ctx.client.messages.create(
                model=ctx.model, max_tokens=1200, temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}])
            ctx.n_llm_calls += 1
            u = getattr(msg, 'usage', None)
            if u:
                ctx.tok_in += getattr(u, 'input_tokens', 0)
                ctx.tok_out += getattr(u, 'output_tokens', 0)
            return clean_sparql(msg.content[0].text)
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"generation failed after retries: {last}")


# =============================================================================
# 4. execute_sparql
# =============================================================================

def _canon_key(sparql: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", sparql.strip()).encode("utf-8")).hexdigest()


# =============================================================================
# Gold-set canonicalization
# =============================================================================
# query.json is the Wikidata Query Service example gallery: 89% have no LIMIT
# and 83% project ?xLabel variables filled by SERVICE wikibase:label. Labels are
# language-dependent and unstable, so comparing them produces false negatives on
# semantically correct queries. We therefore compare on the LABEL-FREE
# projection, applying the identical transform to gold and generated queries.

RE_LABEL_VAR = re.compile(r"\?\w*(?:Label|Description|AltLabel)\b")

MAX_COMPARABLE_ROWS = 2000   # beyond this, exclude from accuracy (not from attribution)
GEN_TIMEOUT = 30             # execution budget for GENERATED queries; gold gets longer


def _remove_label_service(q: str) -> str:
    """Brace-depth-aware removal of `SERVICE wikibase:label { ... }`.

    A regex cannot do this reliably: 3 of the 2838 gold queries contain nested
    OPTIONAL/FILTER blocks that defeat a fixed-nesting pattern.
    """
    out = q
    while True:
        mm = re.search(r"SERVICE\s+wikibase:label\s*\{", out, re.IGNORECASE)
        if not mm:
            return out
        i = mm.end() - 1          # at the opening brace
        depth = 0
        end = None
        for j in range(i, len(out)):
            if out[j] == "{":
                depth += 1
            elif out[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end is None:           # unbalanced; drop to end of string
            return out[:mm.start()]
        out = out[:mm.start()] + " " + out[end:]


def strip_label_projection(sparql: str) -> str:
    """Remove the label service block and any ?xLabel/?xDescription projections.

    CRITICAL: label-suffixed variables are only removed from SELECT / GROUP BY /
    ORDER BY -- NEVER from the WHERE body. Some queries bind such a variable
    explicitly (e.g. `?pronoun wikibase:lemma ?pronounLabel .`), and deleting it
    from the body leaves a triple with no object -> QueryBadFormed.
    """
    if not sparql:
        return sparql
    q = _remove_label_service(sparql)

    # 1. SELECT projection
    m = re.search(r"(SELECT\s+(?:DISTINCT\s+|REDUCED\s+)?)(.*?)(\s+WHERE\b|\s*\{)",
                  q, re.IGNORECASE | re.DOTALL)
    if m:
        head, proj, tail = m.group(1), m.group(2), m.group(3)
        if "*" not in proj:
            kept = RE_LABEL_VAR.sub(" ", proj)
            kept = re.sub(r"[ \t]{2,}", " ", kept).strip()
            if not re.search(r"\?\w+", kept):
                kept = "*"
            q = q[:m.start()] + head + kept + tail + q[m.end():]

    # 2. GROUP BY / ORDER BY clauses only (body left untouched)
    def _clean_clause(mm: re.Match) -> str:
        kw, body = mm.group(1), mm.group(2)
        cleaned = re.sub(r"[ \t]{2,}", " ", RE_LABEL_VAR.sub(" ", body)).strip()
        return f"{kw} {cleaned}" if re.search(r"\?\w+", cleaned) else " "

    q = re.sub(r"\b(GROUP\s+BY|ORDER\s+BY)\b((?:\s*(?:ASC|DESC)?\s*\([^()]*\)|\s*\?\w+)+)",
               _clean_clause, q, flags=re.IGNORECASE)
    return re.sub(r"[ \t]{2,}", " ", q)


_COMMON_PREFIXES = {
    "wd": "http://www.wikidata.org/entity/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    "wikibase": "http://wikiba.se/ontology#",
    "p": "http://www.wikidata.org/prop/",
    "ps": "http://www.wikidata.org/prop/statement/",
    "pq": "http://www.wikidata.org/prop/qualifier/",
    "bd": "http://www.bigdata.com/rdf#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "schema": "http://schema.org/",
    "geo": "http://www.opengis.net/ont/geosparql#",
    "geof": "http://www.opengis.net/def/function/geosparql/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


# OFF by default. WDQS (Blazegraph) predefines all of these, and its geof:
# functions have DIFFERENT signatures from standard GeoSPARQL (geof:distance
# takes 2 args in Blazegraph, 3 in the OGC standard). Declaring geof: explicitly
# can shadow the engine's binding and produce HTTP 500. Only enable when
# executing against a non-WDQS endpoint.
INJECT_PREFIXES = False


def inject_prefixes(sparql: str) -> str:
    """No-op unless INJECT_PREFIXES is enabled (non-WDQS endpoints)."""
    if not INJECT_PREFIXES:
        return sparql
    if not sparql:
        return sparql
    head = sparql[:2000].lower()
    lines = []
    for pfx, uri in _COMMON_PREFIXES.items():
        if re.search(rf"\b{pfx}\s*:", sparql) and f"prefix {pfx}:" not in head:
            lines.append(f"PREFIX {pfx}: <{uri}>")
    return ("\n".join(lines) + "\n" + sparql) if lines else sparql


def execute_sparql(sparql: str, ctx: PipelineHandles,
                   timeout: int = 60, retries: int = 3, cap_rows: bool = False) -> Optional[Set[Tuple[str, ...]]]:
    """Answer-set comparison, not string match.

    Canonicalization: per result row, take the VALUES only (not variable names),
    sort them within the row, make a tuple. Invariant to ?person vs ?p and to
    projection order. Returns None on error/timeout; the reason is recorded in
    ctx.last_exec_error so the caller can report it.
    """
    ctx.last_exec_error = ""
    if not sparql or not sparql.strip():
        ctx.last_exec_error = "empty query"
        return None
    key = _canon_key(sparql)
    if key in ctx.exec_cache:
        cached = ctx.exec_cache[key]
        if cached is None:
            ctx.last_exec_error = ctx.exec_errors.get(key, "cached failure")
        return cached

    from SPARQLWrapper import SPARQLWrapper, JSON

    q = inject_prefixes(strip_label_projection(sparql))
    if cap_rows and not re.search(r"\bLIMIT\s+\d+", q, re.IGNORECASE):
        # Screening only: cap the work. If exactly MAX+1 rows come back the
        # query is too large; if fewer, this IS the complete result set.
        q = q.rstrip().rstrip(";") + f"\nLIMIT {MAX_COMPARABLE_ROWS + 1}"
    out: Optional[Set[Tuple[str, ...]]] = None
    err = ""
    for attempt in range(max(1, retries)):
        try:
            sw = SPARQLWrapper(WIKIDATA_ENDPOINT, agent=USER_AGENT)
            sw.setQuery(q)
            sw.setReturnFormat(JSON)
            sw.setTimeout(timeout)
            res = sw.query().convert()
            ctx.n_sparql_calls += 1

            if "boolean" in res:
                out = {("ASK", str(res["boolean"]).lower())}
            else:
                bindings = res.get("results", {}).get("bindings", [])
                if len(bindings) > MAX_COMPARABLE_ROWS:
                    err = f"too many rows ({len(bindings)}>{MAX_COMPARABLE_ROWS})"
                    out = None
                    break
                rows = set()
                for b in bindings:
                    rows.add(tuple(sorted(str(v.get("value", "")) for v in b.values())))
                out = rows
            err = ""
            break
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:180]}"
            low = err.lower()
            if "429" in low or "too many" in low or "timed out" in low or "timeout" in low:
                time.sleep(5 * (attempt + 1))
                continue
            break

    ctx.exec_cache[key] = out
    if out is None:
        ctx.exec_errors[key] = err or "unknown"
        ctx.last_exec_error = err or "unknown"
    return out


def flatten_answers(rows) -> Set[str]:
    """All bound values, ignoring row/column structure.

    Exact row-tuple equality is unusable here: a semantically correct query that
    projects ?river where the gold projects ?river ?coord produces different
    tuples and never matches. Flattening compares the set of returned VALUES,
    which is what QALD-style answer-set scoring does.
    """
    out: Set[str] = set()
    for row in rows or ():
        for v in row:
            if v:
                out.add(v)
    return out


def score_answers(pred, gold) -> Tuple[bool, float, float, float]:
    """-> (exact, f1, precision, recall) over flattened answer sets."""
    if pred is None or gold is None:
        return False, 0.0, 0.0, 0.0
    exact = bool(pred) and pred == gold
    P, G = flatten_answers(pred), flatten_answers(gold)
    if not P and not G:
        return exact, 0.0, 0.0, 0.0          # empty==empty stays a non-answer
    if not P or not G:
        return exact, 0.0, 0.0, 0.0
    tp = len(P & G)
    prec = tp / len(P)
    rec = tp / len(G)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return exact, f1, prec, rec


def answers_match(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    if not pred and not gold:
        return False      # empty==empty is almost always a broken query
    return pred == gold


# =============================================================================
# Label stripping
# =============================================================================

# (pattern, replacement). Identifiers and example SPARQL must SURVIVE -- they are
# method contributions. Only natural-language labels/descriptions are removed.
# Order matters: the most specific patterns run first.
_LABEL_RULES = [
    # " qid:3  DESCRIPTION:<nl text> sparql:..."  -> drop only the NL description
    # NB: no DOTALL -- with it, a lowercase "description:" on an earlier line
    # matches and the non-greedy run swallows the example SPARQL below it.
    (re.compile(r"DESCRIPTION:[^\n]*?(?=\s*sparql:)"), " "),
    # "Rxxxxxxx:label: <label>description: <desc>"  -> keep the id
    (re.compile(r"\b([ER][A-Za-z0-9]{7}):label:[^\n]*", re.IGNORECASE), r"\1"),
    # "Found: Paris (E9ya727j) - capital ..."       -> keep the id
    (re.compile(r"Found:\s*[^(\n]*\(\s*([EQR][A-Za-z0-9]+)\s*\)[^\n]*", re.IGNORECASE), r"\1"),
    # "0.8912   Rcc4h815  place of birth  -  ...  (url)" -> keep the id
    (re.compile(r"^[ \t]*[\d.]+[ \t]+([ER][A-Za-z0-9]{7})[ \t]+[^\n]*$", re.MULTILINE), r"\1"),
    # leftovers
    (re.compile(r"\bdec?ription:[^\n]*?(?=\s*sparql:|\n|$)", re.IGNORECASE), " "),
    (re.compile(r"\b:label:[^\n]*?(?=\s*sparql:|\n|$)", re.IGNORECASE), " "),
]


def strip_labels(context: str, keep_example_nl: bool = False) -> str:
    """Remove schema documentation while preserving identifiers and examples.

    keep_example_nl=True  ("nolabels"): drop property labels/descriptions but
        KEEP each retrieved example's natural-language question. The model can
        then infer token meaning from paired (question, query) examples. This is
        the realistic private-KG condition: opaque schema, but query history.

    keep_example_nl=False ("strip"): also drop the example questions. Nothing
        connects the English question to the tokens, so the task is close to
        unsolvable by construction -- treat this as a FLOOR condition, not as a
        measurement of method quality.
    """
    out = context
    for pat, repl in _LABEL_RULES:
        if keep_example_nl and "DESCRIPTION" in pat.pattern:
            continue
        out = pat.sub(repl, out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return re.sub(r"\n{3,}", "\n\n", out)


def leak_check(context_anon: str) -> List[str]:
    """Any real Wikidata ID surviving in the anonymized context is a leak."""
    return sorted(set(RE_ANY_REAL_ID.findall(context_anon)))


# =============================================================================
# Run
# =============================================================================

@dataclass
class RunRecord:
    query_index: int
    question: str
    condition: str
    anonymized: bool
    arm: str
    sparql: str = ""
    correct: bool = False
    executed: bool = False
    n_candidates: int = 0
    n_elements: int = 0
    attribution: Optional[Attribution] = None
    error: str = ""
    leaks: str = ""
    gold_ok: bool = True
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    n_tokens_used: int = 0
    n_real_leaked: int = 0
    memory_override: float = 0.0


def run_one_question(qi: int, question: str, gold_sparql: str,
                     ctx: PipelineHandles, anon: Anonymizer,
                     rephrase_mode: str, verbose: bool = True) -> List[RunRecord]:
    records: List[RunRecord] = []
    anon.reset_for_query(qi)

    # --- rephrase -----------------------------------------------------------
    if rephrase_mode == "none":
        q_bare = q_method = question
    elif rephrase_mode == "both":
        rp = rephrase_question(question, ctx)
        q_bare = q_method = rp
    else:  # "method"
        q_bare = question
        q_method = rephrase_question(question, ctx)

    # --- method pipeline: once, in public space -----------------------------
    candidates_pub: Set[str] = set()
    context_pub = ""
    ctx_err = ""
    try:
        candidates_pub, context_pub = build_method_context(
            q_method, ctx, exclude_qid=str(qi),
            exclude_norm=normalize_query(gold_sparql),
            exclude_ids=ctx.retrieval_exclude)
    except Exception as e:
        ctx_err = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"  [!] method context failed: {ctx_err}")

    gold_elements_pub = extract_elements(gold_sparql, anonymized=False)
    gold_answers = execute_sparql(gold_sparql, ctx, timeout=120)
    gold_ok = gold_answers is not None
    if not gold_ok:
        ctx.n_gold_failures += 1
        reason = ("timeout" if "time" in ctx.last_exec_error.lower()
                  else "too_many_rows" if "too many rows" in ctx.last_exec_error.lower()
                  else "error")
        ctx.gold_fail_reasons[reason] = ctx.gold_fail_reasons.get(reason, 0) + 1
        if verbose:
            print(f"  [!] GOLD did not execute ({ctx.last_exec_error}) "
                  f"-- question EXCLUDED from accuracy, attribution still valid")

    context_anon = anon.anonymize_text(context_pub)
    if ctx.label_mode in ("strip", "nolabels"):
        context_anon = strip_labels(context_anon, keep_example_nl=(ctx.label_mode == "nolabels"))
    candidates_anon = anon.anonymize_set(candidates_pub)
    gold_elements_anon = anon.anonymize_set(gold_elements_pub)
    leaks = leak_check(context_anon)
    if leaks and verbose:
        print(f"  [!] LEAK: {len(leaks)} real IDs survive in anon context: {leaks[:6]}")

    # Public arm must run first: its bare-arm elements are the counterfactual
    # reference for BOTH spaces.
    public_bare_elements: Set[str] = set()

    for anonymized in (False, True):
        cand = candidates_anon if anonymized else candidates_pub
        cont = context_anon if anonymized else context_pub
        gold_el = gold_elements_anon if anonymized else gold_elements_pub
        tag = "anon" if anonymized else "public"
        elements_by_arm: Dict[str, Set[str]] = {}
        recs_here: Dict[str, RunRecord] = {}

        for arm in ("bare", "method"):
            rec = RunRecord(query_index=qi, question=question,
                            condition=f"{tag}_{arm}", anonymized=anonymized, arm=arm,
                            n_candidates=len(cand) if arm == "method" else 0,
                            error=ctx_err if arm == "method" else "",
                            leaks="|".join(leaks) if (anonymized and arm == "method") else "",
                            gold_ok=gold_ok)
            if verbose:
                print(f"    .. {tag}/{arm}", end="", flush=True)
            t0 = time.time()
            try:
                q_in = q_method if arm == "method" else q_bare
                sparql = generate_sparql(q_in, cont if arm == "method" else None, ctx)
                rec.sparql = sparql
                els = extract_elements(sparql, anonymized=anonymized)
                # The bare arm never sees the anon vocabulary, so in anon space it
                # emits REAL ids. Read it in real space regardless of `anonymized`.
                if arm == "bare" and anonymized:
                    els = extract_elements(sparql, anonymized=False)
                rec.n_elements = len(els)
                elements_by_arm[arm] = els

                exec_q = anon.deanonymize_text(sparql) if anonymized else sparql
                ans = execute_sparql(exec_q, ctx, timeout=GEN_TIMEOUT,
                                    retries=1, cap_rows=True)
                rec.executed = ans is not None
                if ans is None and ctx.last_exec_error:
                    # was silently dropped before: without this the CSV cannot
                    # distinguish a timeout (correct but slow) from malformed
                    # SPARQL (a genuine quality signal)
                    rec.error = ((rec.error + " | ") if rec.error else "") + \
                                f"exec: {ctx.last_exec_error[:150]}"
                rec.correct, rec.f1, rec.precision, rec.recall = \
                    score_answers(ans, gold_answers)
            except Exception as e:
                rec.error = (rec.error + " | " if rec.error else "") + f"{type(e).__name__}: {e}"
                elements_by_arm[arm] = set()
                if verbose:
                    print(f"  [!] {tag}/{arm}: {e}")
            if verbose:
                print(f"  ({time.time() - t0:.1f}s)", flush=True)
            recs_here[arm] = rec
            records.append(rec)

        if not anonymized:
            public_bare_elements = elements_by_arm.get("bare", set())
            attrib = compute_attribution(
                c_method=elements_by_arm.get("method", set()),
                c_bare=public_bare_elements,
                candidates=cand, c_gold=gold_el)
        else:
            # MEMORY OVERRIDE: did the model ignore the opaque token we supplied
            # and emit the REAL identifier it remembers instead? extract_elements
            # with anonymized=True cannot see this -- real ids do not match the
            # E/R token regex, so they were previously dropped entirely.
            mrec = recs_here.get("method")
            if mrec is not None and mrec.sparql:
                toks = extract_elements(mrec.sparql, anonymized=True)
                real = extract_elements(mrec.sparql, anonymized=False)
                mrec.n_tokens_used = len(toks)
                mrec.n_real_leaked = len(real)
                denom = len(toks) + len(real)
                mrec.memory_override = (len(real) / denom) if denom else 0.0

            # CROSS-SPACE attribution. Comparing anon tokens against the anon-arm
            # bare query is degenerate: bare never emits anon tokens, so
            # redundancy would always be 0 and necessity always 1 by construction.
            # Instead map the method query back to real ids and ask the meaningful
            # question: of the elements the method produced under anonymization,
            # which would the bare LLM have produced when given the real schema?
            method_real = {anon.rev.get(e, e) for e in elements_by_arm.get("method", set())}
            attrib = compute_attribution(
                c_method=method_real,
                c_bare=public_bare_elements,
                candidates=candidates_pub, c_gold=gold_elements_pub)
        recs_here["method"].attribution = attrib

        if verbose:
            b, m = recs_here["bare"], recs_here["method"]
            print(f"  [{tag:6s}] bare_ok={int(b.correct)} method_ok={int(m.correct)}  "
                  f"{attrib.summary_line()}")

    return records


# =============================================================================
# Aggregation / IO
# =============================================================================

CSV_COLS = ["query_index", "condition", "anonymized", "arm", "correct", "executed",
            "gold_ok", "f1", "precision", "recall", "n_candidates", "n_elements", "n_tokens_used", "n_real_leaked",
            "memory_override", "provision", "necessity", "redundancy",
            "memory", "gold_recall", "n_necessary", "n_redundant", "n_memory",
            "necessary_ids", "redundant_ids", "memory_ids", "leaks",
            "question", "sparql", "error"]


def row_of(r: RunRecord) -> Dict[str, Any]:
    a = r.attribution
    return {"query_index": r.query_index, "condition": r.condition,
            "anonymized": int(r.anonymized), "arm": r.arm,
            "correct": int(r.correct), "executed": int(r.executed),
            "gold_ok": int(r.gold_ok), "f1": f"{r.f1:.4f}",
            "precision": f"{r.precision:.4f}", "recall": f"{r.recall:.4f}",
            "n_candidates": r.n_candidates,
            "n_tokens_used": r.n_tokens_used, "n_real_leaked": r.n_real_leaked,
            "memory_override": f"{r.memory_override:.4f}", "n_elements": r.n_elements,
            "provision": f"{a.provision_rate:.4f}" if a else "",
            "necessity": f"{a.necessity_rate:.4f}" if a else "",
            "redundancy": f"{a.redundancy_rate:.4f}" if a else "",
            "memory": f"{a.memory_rate:.4f}" if a else "",
            "gold_recall": f"{a.gold_recall:.4f}" if a else "",
            "n_necessary": len(a.necessary) if a else "",
            "n_redundant": len(a.redundant) if a else "",
            "n_memory": len(a.memory) if a else "",
            "necessary_ids": "|".join(a.necessary) if a else "",
            "redundant_ids": "|".join(a.redundant) if a else "",
            "memory_ids": "|".join(a.memory) if a else "",
            "leaks": r.leaks, "question": r.question,
            "sparql": r.sparql.replace("\n", " / "), "error": r.error}


def done_indices(path: Path) -> Set[int]:
    if not path.exists():
        return set()
    seen = set()
    try:
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seen.add(int(row["query_index"]))
    except Exception:
        pass
    return seen


def aggregate(records, correct_only: bool):
    buckets: Dict[str, List[RunRecord]] = {}
    for r in records:
        buckets.setdefault(r.condition, []).append(r)
    out = {}
    for cond, rs in sorted(buckets.items()):
        acc = [r for r in rs if r.gold_ok]      # gold must be usable for accuracy
        attr = [r for r in rs if r.attribution is not None and r.n_elements > 0]
        if correct_only:
            attr = [r for r in attr if r.correct]
        mean = lambda fn: (sum(fn(r) for r in attr) / len(attr)) if attr else 0.0
        out[cond] = {
            "n": len(rs), "n_att": len(attr),
            "n_acc": len(acc),
            "accuracy": sum(1 for r in acc if r.correct) / len(acc) if acc else 0.0,
            "f1": (sum(r.f1 for r in acc) / len(acc)) if acc else 0.0,
            "exec": sum(1 for r in rs if r.executed) / len(rs) if rs else 0.0,
            "mean_C": mean(lambda r: float(r.attribution.n_elements)),
            "mem_override": ((sum(r.memory_override for r in rs if r.arm == "method")
                              / max(1, sum(1 for r in rs if r.arm == "method")))
                             if any(r.arm == "method" for r in rs) else 0.0),
            "provision": mean(lambda r: r.attribution.provision_rate),
            "necessity": mean(lambda r: r.attribution.necessity_rate),
            "redundancy": mean(lambda r: r.attribution.redundancy_rate),
            "memory": mean(lambda r: r.attribution.memory_rate),
            "gold_recall": mean(lambda r: r.attribution.gold_recall)}
    return out


def print_table(agg, title):
    print(f"\n{'=' * 135}\n{title}\n{'=' * 135}")
    print(f"{'condition':<16}{'n':>5}{'n_acc':>7}{'n_att':>7}{'acc':>8}{'exec':>8}{'|C|':>7}"
          f"{'provis':>9}{'necess':>9}{'redund':>9}{'memory':>9}{'goldrec':>9}{'ovrride':>9}")
    print("-" * 135)
    for c in ("public_bare", "public_method", "anon_bare", "anon_method"):
        if c not in agg:
            continue
        m = agg[c]
        print(f"{c:<16}{m['n']:>5}{m['n_acc']:>7}{m['n_att']:>7}"
              f"{m['accuracy']:>8.3f}{m['exec']:>8.3f}{m['mean_C']:>7.2f}"
              f"{m['provision']:>9.3f}{m['necessity']:>9.3f}{m['redundancy']:>9.3f}"
              f"{m['memory']:>9.3f}{m['gold_recall']:>9.3f}{m['mem_override']:>9.3f}")
    print("=" * 135)


def load_data(qp: Path, ip: Path):
    with qp.open(encoding="utf-8") as f:
        qd = json.load(f)
    gold = [qd[k] for k in sorted(qd, key=lambda x: int(x))]
    with ip.open(encoding="utf-8") as f:
        idd = json.load(f)
    qs = [idd[k][0] for k in sorted(idd, key=lambda x: int(x))]
    n = min(len(gold), len(qs))
    return qs[:n], gold[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, default=Path("query.json"))
    ap.add_argument("--instructions", type=Path, default=Path("instructions.json"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--sample", choices=("head", "random"), default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anon-scope", choices=ANON_SCOPE_CHOICES, default="per_query")
    ap.add_argument("--label-mode", choices=LABEL_MODE_CHOICES, default="keep")
    ap.add_argument("--rephrase", choices=REPHRASE_CHOICES, default="both")
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929")
    ap.add_argument("--out", type=Path, default=Path("exp1b_results.csv"))
    ap.add_argument("--cache", type=Path, default=Path("exp1b_sparql_cache.pkl"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--preflight-timeout", type=int, default=30,
                    help="Per-gold timeout during screening. A gold needing "
                         "more than this is not a usable benchmark query.")
    ap.add_argument("--rescreen", action="store_true",
                    help="With --preflight: drop cached FAILURES, keep successes, "
                         "and retest only the ids not already in --gold-filter. "
                         "Use after a transform fix or a WDQS outage.")
    ap.add_argument("--preflight", action="store_true",
                    help="Only test gold queries; write usable ids to --gold-filter. "
                         "No LLM calls.")
    ap.add_argument("--gold-filter", type=Path, default=Path("gold_usable.json"))
    ap.add_argument("--split-filter", type=Path, default=None,
                    help="JSON list of indices (e.g. test_indices.json from "
                         "make_test_split.py). Restricts evaluation to those.")
    ap.add_argument("--use-gold-filter", action="store_true",
                    help="Sample only from questions listed in --gold-filter.")
    ap.add_argument("--gen-timeout", type=int, default=30,
                    help="Execution budget for GENERATED queries. A generated "
                         "query slower than this cannot match a fast gold anyway.")
    ap.add_argument("--inject-prefixes", action="store_true",
                    help="Declare standard prefixes explicitly. Leave OFF for "
                         "WDQS: it predefines them and its geof: signatures differ.")
    ap.add_argument("--api-key", default=None,
                    help="Anthropic API key. Prefer ANTHROPIC_API_KEY env var or "
                         "--api-key-file; a key passed here is visible in shell "
                         "history and in the process list.")
    ap.add_argument("--api-key-file", type=Path, default=None,
                    help="Path to a file containing only the API key.")
    args = ap.parse_args()

    global INJECT_PREFIXES, GEN_TIMEOUT
    INJECT_PREFIXES = args.inject_prefixes
    GEN_TIMEOUT = args.gen_timeout

    key = args.api_key
    if not key and args.api_key_file:
        try:
            key = args.api_key_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"ERROR: could not read --api-key-file: {e}")
            return 2
    if not key:
        key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ERROR: no API key. Use one of:\n"
              "  set ANTHROPIC_API_KEY=sk-...        (recommended)\n"
              "  --api-key-file key.txt\n"
              "  --api-key sk-...                    (leaks into shell history)")
        return 2
    os.environ["ANTHROPIC_API_KEY"] = key   # anthropic.Anthropic() reads this
    if args.api_key and not args.api_key_file:
        print("WARNING: --api-key was passed on the command line; it is now in your "
              "shell history. Rotate it when the run finishes.")

    questions, gold = load_data(args.queries, args.instructions)
    pool = list(range(len(questions)))
    if args.sample == "random":
        random.Random(args.seed).shuffle(pool)
    else:
        pool = pool[args.start:]
    if args.split_filter and args.split_filter.exists():
        keep = set(json.loads(args.split_filter.read_text(encoding="utf-8")))
        pool = [i for i in pool if i in keep]
        print(f"split filter: restricted to {len(pool)} questions from {args.split_filter}")

    if args.use_gold_filter and args.gold_filter.exists():
        keep = set(json.loads(args.gold_filter.read_text(encoding="utf-8")))
        pool = [i for i in pool if i in keep]
        print(f"gold filter: restricted to {len(pool)} questions with usable gold")

    idxs = pool[:args.limit] if args.limit > 0 else pool
    already = done_indices(args.out) if args.resume else set()
    idxs = [i for i in idxs if i not in already]
    print(f"{len(questions)} questions loaded; running {len(idxs)} "
          f"(skipping {len(already)} done)  scope={args.anon_scope} "
          f"labels={args.label_mode} rephrase={args.rephrase}")
    if args.preflight:
        print(f"PREFLIGHT: screening {len(idxs)} gold queries, no LLM calls. "
              f"Worst case ~{len(idxs) * args.preflight_timeout / 3600:.1f}h if all "
              f"time out; typically far less.")
    else:
        print(f"est. {len(idxs) * (4 + (1 if args.rephrase != 'none' else 0))} LLM calls, "
              f"~{len(idxs) * 5} SPARQL executions")

    if args.preflight:
        args.limit = args.limit if args.limit > 0 else 0
        install_global_api_throttle()
        ctx = PipelineHandles(cache_path=args.cache)
        ctx.label_cache_path = args.cache.with_name(args.cache.stem + "_labels.pkl")
        if args.cache.exists():
            try:
                ctx.exec_cache = pickle.load(args.cache.open("rb"))
            except Exception:
                pass
        usable, reasons = [], {}
        if args.rescreen:
            purged = [k for k, v in ctx.exec_cache.items() if v is None]
            for k in purged:
                ctx.exec_cache.pop(k, None)
                ctx.exec_errors.pop(k, None)
            print(f"  rescreen: purged {len(purged)} cached failures "
                  f"({len(ctx.exec_cache)} successes kept)")
            if args.gold_filter.exists():
                prev = json.loads(args.gold_filter.read_text(encoding="utf-8"))
                usable = list(prev)
                idxs = [i for i in idxs if i not in set(prev)]
                print(f"  rescreen: {len(prev)} already usable, retesting {len(idxs)}")
        for n, qi in enumerate(idxs, 1):
            a = execute_sparql(gold[qi], ctx, timeout=args.preflight_timeout,
                               retries=1, cap_rows=True)
            if a is None:
                r = ("timeout" if "time" in ctx.last_exec_error.lower()
                     else "too_many_rows" if "too many rows" in ctx.last_exec_error.lower()
                     else "error")
                reasons[r] = reasons.get(r, 0) + 1
            else:
                usable.append(qi)
            ok = "ok " if a is not None else "FAIL"
            print(f"  [{n}/{len(idxs)}] Q{qi:<5} {ok} "
                  f"({len(a) if a is not None else ctx.last_exec_error[:60]})",
                  flush=True)
            if n % 50 == 0:
                print(f"  --- {n}/{len(idxs)} screened, {len(usable)} usable {reasons}",
                      flush=True)
                save_cache(ctx)
        args.gold_filter.write_text(json.dumps(sorted(usable)), encoding="utf-8")
        save_cache(ctx)
        print(f"\nusable gold: {len(usable)}/{len(idxs)} ({100*len(usable)/max(1,len(idxs)):.1f}%)")
        print(f"excluded: {reasons}\nwrote -> {args.gold_filter}")
        return 0

    ctx = init_pipeline(args.model, args.label_mode, args.cache)
    if args.split_filter and args.split_filter.exists():
        ctx.retrieval_exclude = {str(i) for i in
                                 json.loads(args.split_filter.read_text(encoding='utf-8'))}
        print(f"retrieval pool: excluding {len(ctx.retrieval_exclude)} held-out ids "
              f"({len(questions) - len(ctx.retrieval_exclude)} remain retrievable)")
    anon = Anonymizer(seed=args.seed, scope=args.anon_scope)

    new_file = not args.out.exists() or not args.resume
    fh = args.out.open("w" if new_file else "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLS)
    if new_file:
        writer.writeheader()

    records: List[RunRecord] = []
    try:
        for n, qi in enumerate(idxs, 1):
            print(f"\n[{n}/{len(idxs)}] Q{qi}: {questions[qi][:88]}")
            try:
                rs = run_one_question(qi, questions[qi], gold[qi], ctx, anon, args.rephrase)
                records.extend(rs)
                for r in rs:
                    writer.writerow(row_of(r))
                fh.flush()
            except Exception as e:
                print(f"  [!] aborted Q{qi}: {type(e).__name__}: {e}")
                traceback.print_exc()
            if n % 25 == 0:
                save_cache(ctx)
                print(f"  ... {ctx.n_llm_calls} LLM, {ctx.n_sparql_calls} SPARQL, "
                      f"${(ctx.tok_in/1e6)*3 + (ctx.tok_out/1e6)*15:.2f} spent, "
                      f"{ctx.n_parse_failures} parse-fallbacks, "
                      f"{len(ctx.prop_labels)} labels cached so far")
            if args.sleep:
                time.sleep(args.sleep)
    finally:
        fh.close()
        save_cache(ctx)

    cost = (ctx.tok_in / 1e6) * 3.0 + (ctx.tok_out / 1e6) * 15.0
    print(f"\nAPI usage: {ctx.n_llm_calls} calls  "
          f"{ctx.tok_in:,} in / {ctx.tok_out:,} out tokens")
    print(f"Estimated cost: ${cost:.2f}  (Sonnet 4.5 @ $3/$15 per MTok)")
    print(f"\nWrote {len(records)} rows -> {args.out}")
    fails = [r for r in records if not r.executed]
    kinds = {}
    for r in fails:
        e = r.error.lower()
        k = ("timeout" if "time" in e else
             "malformed" if "badformed" in e else
             "too_many_rows" if "too many rows" in e else
             "other" if e else "unknown")
        kinds[k] = kinds.get(k, 0) + 1
    print(f"generated queries that did not execute: {len(fails)}/{len(records)} -> {kinds}")
    print(f"gold unusable: {ctx.n_gold_failures} excluded from accuracy "
          f"-> {dict(ctx.gold_fail_reasons)}")
    print(f"self-retrievals blocked (same id):      {ctx.n_self_retrievals}")
    print(f"duplicate-gold blocked (same content):  {ctx.n_dup_retrievals}")
    print(f"held-out examples dropped from retrieval: {ctx.n_heldout_retrievals}")
    print(f"utilExpre parse fallbacks: {ctx.n_parse_failures} "
          f"(regex harvest used; candidates unaffected)")
    print_table(aggregate(records, False), "ALL QUERIES (unconditioned)")
    print_table(aggregate(records, True), "CORRECT METHOD QUERIES ONLY (conditioned)")
    print("\nSanity gate: anon_bare accuracy must collapse toward 0.")
    print("If it does not, check the 'leaks' column -- real IDs are surviving anonymization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
