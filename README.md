# Necessity, Not Provision — Wikidata experiments

Code for the Wikidata experiments in *Necessity, Not Provision: Counterfactual Attribution for Retrieval-Augmented SPARQL Generation* (anonymous submission).

The paper asks how much a retrieval-augmented text-to-SPARQL system actually depends on the identifiers its retriever supplies. For each question the same model generates two queries:

- the **bare arm**, with no retrieved context;
- the **method arm**, with the retrieved candidates.

The identifiers in the method-arm query are then partitioned into three sets:

| Set | Definition | Rate |
|---|---|---|
| **N** necessary | supplied by retrieval, absent from the bare query | necessity ν = \|N\| / \|C(Q_m)\| |
| **R** redundant | supplied by retrieval, also in the bare query | redundancy ρ = \|R\| / \|C(Q_m)\| |
| **L** memory | not supplied by retrieval | memory μ = \|L\| / \|C(Q_m)\| |

Provision = ν + ρ. Necessity needs no reference query, no execution and no judge model: just one extra generation and a set comparison.

---

## Repository contents

### Experiment scripts

| File | Purpose | Paper |
|---|---|---|
| `exp1b_wired.py` | Main Wikidata runner: retrieval pipeline, bare and method arms, attribution, answer-set scoring, gold screening, per-question anonymisation | Table 1 (Bare, Full pipeline), Table 6, Fig. 2, Table 7, Table 12 / App. A.7 |
| `oracle_placebo.py` | Oracle (exactly the reference identifiers) and placebo (size-matched identifiers from other questions) | Table 1 (Oracle, Placebo), Table 3 rows 1 and 4 |
| `oracle_distractors.py` | Oracle + distractors (gold padded to deployed size) and small placebo (gold-sized, no gold), completing the 2×2 | Table 3 rows 2 and 3 |
| `build_instruction_embeddings.py` | Builds the example-retrieval index from `instructions.json` | setup |
| `analysis.py` | Recomputes all Wikidata statistics from the CSVs above (no LLM or SPARQL calls) | Tables 1, 2, 3, 7, 12, Fig. 2, App. A.3 |

### Retrieval helper modules (imported by `exp1b_wired.py`)

| Module | Used for |
|---|---|
| `utilLookup.py` | Wikidata property search (`wbsearchentities`, type=property) |
| `utilSem2.py` | Entity search and re-ranking by label match and MiniLM similarity |
| `utilsPre.py` | Downloads all Wikidata properties and embeds them with `all-MiniLM-L6-v2`; nearest-property search |
| `utilExpre.py` | rdflib-based extraction of entities and predicates from retrieved example queries |

### Data

| File | Description |
|---|---|
| `query.json` | Instruct-to-SPARQL reference queries, `{id: sparql}`, 2,838 items, ids 0–2837 |
| `instructions.json` | Natural-language paraphrases, `{id: [question, ...]}`; the first paraphrase is the test question, all paraphrases are indexed for example retrieval |
| `test_indices.json` | Test-split indices (518); these items are evaluated and also removed from the retrieval pool |
| `gold_usable.json` | The 366 test items whose reference query executes (output of the screening step, shipped so it can be skipped) |

Dataset: Ben Amor et al., *Instruct-to-SPARQL: A Text-to-SPARQL Dataset for Training SPARQL Agents*, CHIIR 2025.

### Generated on first run (not stored in the repository)

| File | Produced by |
|---|---|
| `embeddings_output/instructions_embeddings.pkl` | `build_instruction_embeddings.py` |
| `_wikidata_cache/wikidata_properties_en.csv` | `utilsPre.fetch_properties()`, from the live Wikidata Query Service |
| `_wikidata_cache/prop_vecs_en_sentence-transformers_all-MiniLM-L6-v2.npy` | `utilsPre.embed_properties()` |
| `exp1b_sparql_cache.pkl`, `exp1b_sparql_cache_labels.pkl` | SPARQL-result and label caches, shared by all scripts |

### Not in this release

Results for the following are reported in the paper, but their code is **not** included here:

- `entprop` / `examples` retrieval ablations (Table 1);
- Llama-3.1-8B runs (Table 4, App. A.4);
- DBpedia / DB25 (App. A.5);
- CK25 and CK25-R (App. A.6);
- the ARUQULA evaluation (Section 4.5).

---

## Requirements

- Python ≥ 3.10
- An Anthropic API key (the primary generator is `claude-sonnet-4-5-20250929`)
- Network access to the Wikidata Query Service and the Wikidata Action API

```bash
pip install anthropic SPARQLWrapper requests numpy pandas fastembed spacy \
            sentence-transformers torch scikit-learn rdflib scipy
python -m spacy download en_core_web_lg
```

`scipy` is used only by `analysis.py` for exact p-values; without it, p-values fall back to a normal approximation.

## Configuration

The three experiment scripts take the Anthropic API key from, in order of precedence:

1. `--api-key sk-...`
2. `--api-key-file key.txt` (a file containing only the key)
3. the `ANTHROPIC_API_KEY` environment variable

```bash
export ANTHROPIC_API_KEY=sk-...          # Windows: set ANTHROPIC_API_KEY=sk-...
```

The environment variable or a key file is recommended: a key passed with `--api-key` is stored in shell history. If you use a key file, keep it out of version control (see `.gitignore` below).

Before running, edit `USER_AGENT` at the top of `exp1b_wired.py` to include your own contact address. Wikimedia's API policy requires a real contact in the User-Agent.

All scripts call the model at temperature 0, with at most 1,200 output tokens per generation.

---

## Reproducing the Wikidata results

### 0. Build the example-retrieval index (once)

```bash
python build_instruction_embeddings.py --parallel 0
```

This embeds all 14,190 paraphrases in `instructions.json` (5 per question) with `BAAI/bge-large-en-v1.5` and writes `embeddings_output/instructions_embeddings.pkl`. The first run downloads the model (about 1.3 GB). Embedding on a CPU takes roughly 10–40 minutes and prints progress every 500 paraphrases. `--parallel 0` uses all CPU cores; omit it if multiprocessing fails on your platform.

### 1. Screen the reference queries (no LLM calls)

Each test reference query is executed once, with a 30 s timeout and a 2,001-row cap. Queries that time out, exceed 2,000 rows or fail to parse are excluded from every condition.

```bash
python exp1b_wired.py --preflight --split-filter test_indices.json \
    --limit 0 --preflight-timeout 30 --gold-filter gold_usable.json
```

Expected: 366 of 518 queries usable (Table 6). This step needs only `requests` and `SPARQLWrapper`; it does not load the retrieval modules. The shipped `gold_usable.json` is the result of this step, so it can be skipped.

### 2. Run the bare and method arms

```bash
python exp1b_wired.py --split-filter test_indices.json --use-gold-filter \
    --limit 0 --label-mode keep --rephrase both --out r_keep.csv --resume
```

On first use this also downloads and embeds the Wikidata property list into `_wikidata_cache/`, which takes a few minutes.

The run writes four rows per question to `r_keep.csv`, one each for `public_bare`, `public_method`, `anon_bare` and `anon_method`:
- the public rows give the Bare and Full-pipeline results;
- the anonymised rows give the per-question anonymisation experiment (App. A.7). For the `strip` variant, rerun with `--label-mode strip --out r_strip.csv`.

Each question costs 5 LLM calls (one rephrase plus four generations) and about 5 SPARQL executions. A full run of 366 questions is roughly 1,830 calls.

Leakage controls are applied during example retrieval:
- every test item is removed from the retrieval pool (`--split-filter`);
- examples sharing the test query's id are dropped;
- examples whose normalised SPARQL is identical to the reference are dropped (78 queries occur under two to four ids).

The run reports how many examples each filter removed.

### 3. Oracle and placebo

```bash
python oracle_placebo.py --results r_keep.csv --out r_oracle_placebo.csv --resume
```

This script does not rerun the bare arm; it reads the `public_bare` row from `r_keep.csv`. Two generations per question. It does not load the retrieval modules.

### 4. Oracle + distractors and small placebo

```bash
python oracle_distractors.py --results r_keep.csv \
    --oracle r_oracle_placebo.csv --out r_distractors.csv --resume
```

The summary it prints is restricted to the items present in all four conditions.

### 5. Statistics

```bash
python analysis.py --results r_keep.csv --oracle r_oracle_placebo.csv \
    --distractors r_distractors.csv --queries query.json --json stats.json
```

This makes no LLM or SPARQL calls and runs in seconds. `--oracle` and `--distractors` are optional; their sections are skipped if the files are missing. It prints seven sections and writes every number to `stats.json`:

| Section | Paper |
|---|---|
| 1. Attribution and answer quality of the full pipeline | Table 1 |
| 2. Correlations with element-level F1 gain, Williams/Steiger tests, partial correlations, regression, median split, answer-level correlations | Table 2, App. A.3 |
| 3. Attribution by candidate recall | Fig. 2, Table 7 |
| 4. Decoding-noise floor | App. A.3 |
| 5. Oracle and placebo | Table 1 |
| 6. Reference presence × candidate-set size | Table 3 |
| 7. Per-question anonymisation | Table 12, App. A.7 |

Definitions used:
- **element F1**: F1 between the identifier set of a generated query and that of the reference query;
- **gain**: element F1 of the method arm minus that of the bare arm;
- **divergence** δ = \|C(Q_m) \ C(Q_b)\| / \|C(Q_m)\|, and its unsupplied part μ_new, so that δ = ν + μ_new;
- **decoding-noise floor** φ, computed between the `public_bare` and `anon_bare` arms. The bare arm receives no retrieved context, so these two arms use the same prompt and differ only in being separate API calls.

Confidence intervals are 95% percentile intervals from a paired bootstrap over questions (10,000 resamples, `--seed 0`).

### Smoke test

To check the setup on a single question before a full run:

```bash
python exp1b_wired.py --split-filter test_indices.json --limit 1 --out smoke.csv
```

`smoke.csv` should contain four rows. The printed method context should contain all three of:
- `Found: …` lines (entity linking);
- score-prefixed property lines (embedding property search);
- `qid:` example lines.

If only `qid:` lines appear, a retrieval module is failing silently (see *Reproducibility notes*).

---

## How a question is processed

```
question
 ├─ rephrase into Wikidata-oriented form (1 LLM call, shared by both arms)
 ├─ build retrieval context
 │    ├─ noun/verb phrases (spaCy)  → property search        (utilLookup)
 │    ├─ named entities (spaCy)     → entity linking         (utilSem2)
 │    ├─ phrases                    → MiniLM property search (utilsPre)
 │    └─ question                   → top-3 similar examples (bge-large index)
 │                                    + their identifiers and property labels
 ├─ execute the reference query
 ├─ bare arm:   generate SPARQL from the question alone → execute → score
 ├─ method arm: generate SPARQL with the context        → execute → score
 └─ attribution: partition method-arm identifiers into N / R / L
```

Every identifier that appears anywhere in the retrieved context counts as a candidate, including those inside retrieved examples.

---

## Output columns

Columns shared by all scripts:

| Column | Meaning |
|---|---|
| `query_index` | Item id in `query.json` |
| `condition` | `public_bare`, `public_method`, `anon_bare`, `anon_method`, `oracle`, `placebo`, `oracle_distractors`, `placebo_small` |
| `n_candidates` | \|Cand(q)\| supplied to the method arm |
| `n_elements` | \|C(Q_m)\|, identifiers in the generated query |
| `provision`, `necessity`, `redundancy`, `memory` | Attribution rates (method-arm rows only) |
| `necessary_ids`, `redundant_ids`, `memory_ids` | The identifiers in N, R and L |
| `gold_recall` | Recall of the candidate set against the reference identifiers |
| `f1` | Answer-set F1 against the reference answers (0 when execution fails) |
| `executed` | Whether the generated query ran within the timeout and row cap |
| `sparql`, `error` | Generated query and any execution error |

Columns specific to `exp1b_wired.py`: `n_tokens_used`, `n_real_leaked`, `memory_override` (override rate λ under anonymisation) and `leaks` (real ids surviving in the anonymised context).

Columns specific to the oracle and distractor scripts: `divergence` (Jaccard distance between the bare and method identifier sets), and `n_gold` / `n_distractors` for the distractor run.

Note that the `divergence` column is a symmetric Jaccard distance, used for the placebo-versus-oracle comparison. The divergence δ in Table 2 is the asymmetric quantity defined above, computed by `analysis.py` from the stored queries.

---

## Implementation notes

**Identifier extraction.** Identifiers are matched lexically, including inside comments, and are not parsed: `\bQ\d+\b` and `\bP\d+\b`. All prefix forms map to the bare id, so `wd:`, `wdt:`, `p:`, `ps:`, `pq:` and full IRIs all count.

**Answer scoring.** Before execution, both the reference and the generated query are rewritten: `SERVICE wikibase:label { … }` blocks are removed, and label-suffixed variables are dropped from `SELECT`, `GROUP BY` and `ORDER BY` (but never from the query body). Results are compared as flattened value sets. An empty-vs-empty result scores 0.

**Rephrasing.** The question is rewritten once into Wikidata-oriented form, and the same rewrite is used by both arms (`--rephrase both`).

**Placebo sampling.** Placebo and distractor identifiers are drawn from the identifiers of other questions' reference queries. Both the current reference identifiers and the bare-arm identifiers are excluded from sampling, so placebo redundancy is 0 by construction.

**Prompts.**
- Oracle, placebo and distractor candidates are rendered as `Found: <label> (<id>) - <description>`, sorted by type and number.
- The oracle uses the header *"USE THESE IDENTIFIERS. They are the ones the query needs:"*.
- The placebo, small-placebo and oracle + distractors conditions use the deployed pipeline's header.

---

## Reproducibility notes

- **Live property list.** `_wikidata_cache/` is built from the live Wikidata Query Service on first run. Wikidata properties are added and deprecated over time, so property-search candidates can differ slightly from the reported runs. The one-off download has a 180 s timeout; if it fails under load, rerun.
- **API throttling.** `exp1b_wired.py` paces calls made through `requests.get`. `utilLookup` and `utilSem2` use their own `requests.Session`, so the pacing does not reach them:
  - `utilSem2` retries on HTTP 429;
  - `utilLookup` does not retry. On a 429 it prints `Error searching` and returns no results, so property lookup contributes nothing for that phrase.

  Candidate sets can therefore vary slightly with Wikidata API load.
- **Live endpoint.** Reference and generated queries execute against the live Wikidata Query Service, so answer sets reflect the state of Wikidata at run time.
- **Generator.** Results were obtained with `claude-sonnet-4-5-20250929` in August 2026. Even at temperature 0, API outputs are not guaranteed to be bit-identical across runs.

Expect the provision/necessity gap and the direction of every effect to reproduce, with individual values differing slightly from the paper.

---



## License

MIT for the code. Instruct-to-SPARQL and Wikidata content remain under their own licenses.
