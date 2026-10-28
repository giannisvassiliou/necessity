import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
# Config / paths
# -----------------------------
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
CACHE_DIR = Path("./_wikidata_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Global toggle set by --refresh-cache
REFRESH_CACHE = False

# ---------------------------
# -----------------------------
# Utils
# -----------------------------
def _sanitize(s: str) -> str:
    """Safe string for filenames."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def safe_str(v) -> str:
    """Convert possibly-NaN/None values to a clean string."""
    if v is None:
        return ""
    if isinstance(v, float):
        try:
            if math.isnan(v):
                return ""
        except Exception:
            pass
    return str(v).strip()

def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

# -----------------------------
# Data fetching
# -----------------------------
def fetch_properties(language: str = "en") -> pd.DataFrame:
    """
    Fetch all Wikidata properties (label/description/aliases) for 'language'.
    Caches to ./_wikidata_cache/wikidata_properties_<lang>.csv
    """
    cache_path = CACHE_DIR / f"wikidata_properties_{language}.csv"

    # Load from cache (unless refresh requested)
    if cache_path.exists() and not REFRESH_CACHE:
        df = pd.read_csv(cache_path, dtype=str).fillna("")
        if "pid" not in df.columns:
            df["pid"] = df["prop"].apply(lambda u: u.rsplit("/", 1)[-1])
        if "url" not in df.columns:
            df["url"] = df["pid"].apply(lambda pid: f"https://www.wikidata.org/wiki/Property:{pid}")
        return df

    query = f"""
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT ?prop ?propLabel ?propDescription
           (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases)
    WHERE {{
      ?prop a wikibase:Property .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language}". }}
      OPTIONAL {{ ?prop skos:altLabel ?alias FILTER(LANG(?alias) = "{language}") }}
    }}
    GROUP BY ?prop ?propLabel ?propDescription
    """

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "wikidata-predicate-search/1.0 (+https://www.wikidata.org/)"
    }
    resp = requests.get(SPARQL_ENDPOINT, params={"query": query}, headers=headers, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for b in data["results"]["bindings"]:
        prop = b["prop"]["value"]  # e.g., https://www.wikidata.org/entity/P25
        label = b.get("propLabel", {}).get("value", "")
        desc = b.get("propDescription", {}).get("value", "")
        aliases = b.get("aliases", {}).get("value", "")
        rows.append({"prop": prop, "propLabel": label, "propDescription": desc, "aliases": aliases})

    df = pd.DataFrame(rows, dtype=str).fillna("")
    df["pid"] = df["prop"].apply(lambda u: u.rsplit("/", 1)[-1])  # P###
    df["url"] = df["pid"].apply(lambda pid: f"https://www.wikidata.org/wiki/Property:{pid}")

    # Persist cache
    df.to_csv(cache_path, index=False)
    return df

def build_property_texts(df: pd.DataFrame) -> List[str]:
    """Concatenate label + description + aliases into a single text per property."""
    for col in ("propLabel", "propDescription", "aliases"):
        if col not in df.columns:
            df[col] = ""

    texts = []
    for _, r in df.iterrows():
        label = safe_str(r.get("propLabel"))
        desc = safe_str(r.get("propDescription"))
        aliases = safe_str(r.get("aliases"))
        if aliases:
            aliases = aliases.replace("|", " | ")

        parts = [p for p in (label, desc, aliases) if p]
        text = " | ".join(parts) if parts else label
        texts.append(text)
    return texts

# -----------------------------
# Embedding / search
# -----------------------------
def pick_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' if available, else 'cpu'."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def embed_properties(df: pd.DataFrame, model_name: str, language: str,
                     device: str, batch_size: int, show_progress: bool,
                     force_rebuild: bool = False) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model_tag = _sanitize(model_name)
    cache_npy = CACHE_DIR / f"prop_vecs_{language}_{model_tag}.npy"

    if cache_npy.exists() and not force_rebuild:
        return np.load(cache_npy)

    resolved_device = pick_device(device)
    model = SentenceTransformer(model_name, device=resolved_device)
    texts = build_property_texts(df)
    prop_vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True
    ).astype("float32")
    np.save(cache_npy, prop_vecs)
    return prop_vecs

def embed_queries(terms: List[str], model_name: str, device: str,
                  batch_size: int, show_progress: bool) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    resolved_device = pick_device(device)
    model = SentenceTransformer(model_name, device=resolved_device)
    q_vecs = model.encode(
        terms,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True
    ).astype("float32")
    return q_vecs

def search(terms: List[str], df: pd.DataFrame, prop_vecs: np.ndarray, q_vecs: np.ndarray, topk: int
           ) -> Dict[str, List[Dict[str, Any]]]:
    prop_vecs = prop_vecs.astype("float32")
    q_vecs = normalize_rows(q_vecs.astype("float32"))
    sims = q_vecs @ prop_vecs.T  # [n_terms, n_props]

    out: Dict[str, List[Dict[str, Any]]] = {}
    for i, term in enumerate(terms):
        k = min(topk, sims.shape[1])
        idx = np.argpartition(-sims[i], k-1)[:k]
        idx = idx[np.argsort(-sims[i, idx])]
        results = []
        for j in idx:
            results.append({
                "score": float(sims[i, j]),
                "pid": df.iloc[j]["pid"],
                "label": df.iloc[j]["propLabel"],
                "description": df.iloc[j]["propDescription"],
                "url": df.iloc[j]["url"],
            })
        out[term] = results
    return out
