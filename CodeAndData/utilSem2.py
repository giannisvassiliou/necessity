#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import numpy as np

# ------------------------------
# Networking setup (robust HTTP)
# ------------------------------
SESSION = requests.Session()

# Retries for transient errors
retries = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
SESSION.mount("https://", HTTPAdapter(max_retries=retries))

# Provide a real contact in User-Agent if you can, per Wikimedia etiquette.
HEADERS = {
    "User-Agent": "MyWikidataLookupScript/1.0 (your.email@example.com)"
}

API_URL = "https://www.wikidata.org/w/api.php"

# ------------------------------
# Model (downloads on first run)
# ------------------------------
print("Loading semantic similarity model...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast + effective
print("Model loaded!\n")


# ------------------------------
# Helpers
# ------------------------------
def fuzzy_match_score(search_term: str, label: str) -> float:
    """
    Fuzzy string matching score between search term and label (0-1).
    """
    if not search_term or not label:
        return 0.0
    return SequenceMatcher(None, search_term.lower(), label.lower()).ratio()


def phrase_contains_label(phrase: str, label: str) -> bool:
    """
    Case-insensitive substring check: is label mentioned inside phrase?
    """
    if not phrase or not label:
        return False
    return label.lower() in phrase.lower()


def is_generic_entity(description: str) -> bool:
    """
    Penalize generic entity types (given name, family name, etc.).
    """
    if not description:
        return False
    generic_types = [
        "given name",
        "family name",
        "surname",
        "first name",
        "male given name",
        "female given name",
        "unisex given name",
    ]
    dl = description.lower()
    return any(g in dl for g in generic_types)


# ------------------------------
# Wikidata API layer
# ------------------------------
def get_wikidata_id(name: str, language: str = "en", limit: int = 10):
    """
    Search Wikidata by name and return raw wbsearchentities results.
    """
    if not name:
        return []

    params = {
        "action": "wbsearchentities",
        "language": language,  # language of label/desc to return
        "uselang": language,   # UI language; sometimes influences match formatting
        "format": "json",
        "type": "item",
        "limit": limit,
        "search": name,
    }
    resp = SESSION.get(API_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("search", []) or []


def enrich_entities_with_details(items, language: str = "en"):
    """
    Given wbsearchentities items, fetch full labels/descriptions/aliases in one batch.
    Guarantees presence of 'label' and provides best-effort 'description' and 'aliases'.
    """
    if not items:
        return []

    ids = [it["id"] for it in items]
    params = {
        "action": "wbgetentities",
        "ids": "|".join(ids),
        "props": "labels|descriptions|aliases",
        "languages": language,
        "format": "json",
    }
    resp = SESSION.get(API_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    entities = resp.json().get("entities", {})

    enriched = []
    for it in items:
        ent = entities.get(it["id"], {})
        label = (ent.get("labels", {}).get(language) or {}).get("value") or it.get("label", "")
        desc = (ent.get("descriptions", {}).get(language) or {}).get("value") or it.get("description", "") or ""
        aliases = [a.get("value", "") for a in ent.get("aliases", {}).get(language, [])]
        enriched.append(
            {
                "id": it["id"],
                "label": label or "",
                "description": desc,  # may be ""
                "aliases": aliases,   # may be []
                "match": it.get("match", {}),
            }
        )
    return enriched


# ------------------------------
# Scoring
# ------------------------------
def calculate_combined_similarity(
    search_term: str,
    user_phrase: str,
    results: list,
    label_weight: float = 0.2,
    semantic_weight: float = 0.5,
    label_phrase_weight: float = 0.3,
):
    """
    Combined similarity using:
    1. Label matching (fuzzy search vs label) -> label_weight
    2. Semantic similarity (phrase vs description or label/aliases fallback) -> semantic_weight
    3. Label-phrase semantic similarity (phrase vs label) -> label_phrase_weight

    Bonuses:
      +0.4 if phrase explicitly contains the label
    Penalties:
      -0.3 for generic entity description
    """
    if not results:
        return []

    # Prepare text used for semantic similarity to description; fall back to label/aliases
    texts_for_semantic = []
    labels = []

    for r in results:
        labels.append(r.get("label", "") or "")
        if r.get("description"):
            texts_for_semantic.append(r["description"])
        else:
            alias_str = "; ".join(r.get("aliases", []) or [])
            fallback = (r.get("label", "") + (" — " + alias_str if alias_str else "")).strip()
            texts_for_semantic.append(fallback)

    # Encode in batches (fast)
    user_emb = model.encode([user_phrase])  # shape (1, d)
    desc_embs = model.encode(texts_for_semantic)  # shape (n, d)
    label_embs = model.encode(labels)  # shape (n, d)

    # Cosine similarities
    desc_sims = cosine_similarity(user_emb, desc_embs).flatten()  # [-1,1]
    label_phrase_sims = cosine_similarity(user_emb, label_embs).flatten()  # [-1,1]

    # Score
    scored = []
    for i, item in enumerate(results):
        item_label = item.get("label", "") or ""
        item_desc = item.get("description", "") or ""

        label_score = fuzzy_match_score(search_term, item_label)

        # Normalize cosine from [-1,1] -> [0,1] for weighted sum
        sem_norm = (float(desc_sims[i]) + 1) / 2.0
        label_phrase_norm = (float(label_phrase_sims[i]) + 1) / 2.0

        phrase_mention_bonus = 0.4 if phrase_contains_label(user_phrase, item_label) else 0.0
        generic_penalty = 0.3 if is_generic_entity(item_desc) else 0.0

        combined = (
            label_weight * label_score
            + semantic_weight * sem_norm
            + label_phrase_weight * label_phrase_norm
            + phrase_mention_bonus
            - generic_penalty
        )

        # Store detailed breakdown
        enriched_item = dict(item)
        enriched_item.update(
            {
                "label_score": float(label_score),
                "semantic_similarity": float(desc_sims[i]),
                "label_phrase_similarity": float(label_phrase_sims[i]),
                "phrase_mention_bonus": float(phrase_mention_bonus),
                "generic_penalty": float(generic_penalty),
                "combined_score": float(combined),
            }
        )
        scored.append(enriched_item)

    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    return scored


# ------------------------------
# CLI / Script entry
# ------------------------------
def main():
    try:
        # Read inputs
        if sys.stdin.isatty():
            search_term = input("Enter a search term (e.g., 'Hercules'): ").strip()
        else:
            search_term = sys.stdin.readline().strip()

        if not search_term:
            print("No search term provided.")
            return

        if sys.stdin.isatty():
            user_phrase = input("Enter a phrase to compare (e.g., 'strong warrior from mythology'): ").strip()
        else:
            user_phrase = sys.stdin.readline().strip()

        if not user_phrase:
            print("No comparison phrase provided.")
            return

        language = "en"
        limit = 10

        print(f"\nSearching for '{search_term}'...")
        raw_results = get_wikidata_id(search_term, language=language, limit=limit)

        if not raw_results:
            print("No results found.")
            return

        # Enrich results to ensure we have labels/descriptions/aliases
        results = enrich_entities_with_details(raw_results, language=language)

        # Rank results
        ranked_results = calculate_combined_similarity(
            search_term=search_term,
            user_phrase=user_phrase,
            results=results,
            label_weight=0.2,       # 20% label fuzzy match
            semantic_weight=0.5,    # 50% phrase vs description/alias fallback
            label_phrase_weight=0.3 # 30% phrase vs label
        )

        if not ranked_results:
            print("No results could be ranked.")
            return

        # Pretty print
        print(f"\nResults ranked by combined score:")
        print(f"Phrase: '{user_phrase}'")
        print("=" * 130)
        for i, item in enumerate(ranked_results, 1):
            bonus_str = f" +Bonus: {item['phrase_mention_bonus']:.2f}" if item["phrase_mention_bonus"] > 0 else ""
            penalty_str = f" -Penalty: {item['generic_penalty']:.2f}" if item["generic_penalty"] > 0 else ""
            print(
                f"{i}. {item['label']} ({item['id']}) - Combined Score: {item['combined_score']:.4f}"
                f"{bonus_str}{penalty_str}"
            )
            print(
                f"   Label Match: {item['label_score']:.4f} | "
                f"Desc Similarity: {item['semantic_similarity']:.4f} | "
                f"Label-Phrase: {item['label_phrase_similarity']:.4f}"
            )
            desc = item.get("description") or ""
            aliases = item.get("aliases") or []
            if desc:
                print(f"   Description: {desc}")
            if aliases:
                print(f"   Aliases: {', '.join(a for a in aliases if a)[:400]}")
            print("-" * 130)

    except requests.HTTPError as e:
        print("HTTP error:", e, "\nBody:", getattr(e.response, "text", "")[:400])
    except requests.RequestException as e:
        print("Request failed:", e)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
