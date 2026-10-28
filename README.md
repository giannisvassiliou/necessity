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

## What is in this release

| File | Purpose | Paper |
|---|---|---|
| `exp1b_wired.py` | Main Wikidata runner: retrieval pipeline, bare and method arms, attribution, answer-set scoring, gold screening, per-question anonymisation | Table 1 (Bare, Full pipeline), Table 6, Fig. 2, Table 7, Table 12 / App. A.7 |
| `oracle_placebo.py` | Oracle (exactly the reference identifiers) and placebo (size-matched identifiers from other questions) | Table 1 (Oracle, Placebo), Table 3 rows 1 and 4 |
| `oracle_distractors.py` | Oracle + distractors (gold padded to deployed size) and small placebo (gold-sized, no gold), completing the 2×2 | Table 3 rows 2 and 3 |

### Not yet in this release

Results for the following are reported in the paper, but their code is **not** in this repository yet:

- `entprop` / `examples` retrieval ablations (Table 1);
- element-level F1, divergence, correlation and regression analyses, Steiger/Williams tests, paired bootstrap CIs (Table 2, App. A.3);
- decoding-noise floor estimation;
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
pip install anthropic SPARQLWrapper requests numpy fastembed spacy sentence-transformers rdflib pandas
python -m spacy download en_core_web_lg
```

### Pipeline helper modules

`exp1b_wired.py` imports four helper modules from the original retrieval pipeline. They must sit in the same directory:

| Module | Used for |
|---|---|
| `utilLookup.py` | Wikidata property search (`wbsearchentities`) |
| `utilSem2.py` | Entity linking and ranking |
| `utilsPre.py` | Property table download and MiniLM property embeddings |
| `utilExpre.py` | rdflib-based extraction of entities and predicates from example queries |

### Data

| File | Description |
|---|---|
| `query.json` | Instruct-to-SPARQL reference queries, `{id: sparql}` |
| `instructions.json` | Natural-language paraphrases, `{id: [question, ...]}`; the first paraphrase is used |
| `embeddings_output/instructions_embeddings.pkl` | `bge-large-en-v1.5` embeddings of all paraphrases (example retrieval index) |
| `test_indices.json` | Test-split indices; these are also excluded from the retrieval pool |
| `gold_usable.json` | Produced by the screening step below |

Dataset: Ben Amor et al., *Instruct-to-SPARQL*, CHIIR 2025.

---

## Configuration

Set the API key as an environment variable. Do not put it on the command line:

```bash
export ANTHROPIC_API_KEY=sk-...          # Windows: set ANTHROPIC_API_KEY=sk-...
```

Before running, edit `USER_AGENT` at the top of `exp1b_wired.py` to include your own contact address. Wikimedia's API policy requires a real contact in the User-Agent.

All scripts call the model at temperature 0. They throttle every request to `wikidata.org` and cache SPARQL results and labels in `exp1b_sparql_cache.pkl` and `exp1b_sparql_cache_labels.pkl`.

---

## Reproducing the Wikidata results

### 1. Screen the reference queries (no LLM calls)

Each test reference query is executed once, with a 30 s timeout and a 2,001-row cap. Queries that time out, exceed 2,000 rows or fail to parse are excluded from every condition.

```bash
python exp1b_wired.py --preflight --split-filter test_indices.json \
    --limit 0 --preflight-timeout 30 --gold-filter gold_usable.json
```

Expected: 366 of 518 queries usable (Table 6).

### 2. Run the bare and method arms (public identifiers)

```bash
python exp1b_wired.py --split-filter test_indices.json --use-gold-filter \
    --limit 0 --label-mode keep --rephrase both --out r_keep.csv --resume
```

This writes four rows per question to `r_keep.csv`, one each for `public_bare`, `public_method`, `anon_bare` and `anon_method`. The public rows give the Bare and Full-pipeline results. The anonymised rows give the per-question anonymisation experiment (App. A.7); for the `strip` variant, rerun with `--label-mode strip --out r_strip.csv`.

Leakage controls are applied during example retrieval:

- every test item is removed from the retrieval pool (`--split-filter`);
- examples sharing the test query's id are dropped;
- examples whose normalised SPARQL is identical to the reference are dropped (78 queries occur under several ids).

The run reports how many examples each filter removed.

### 3. Oracle and placebo

```bash
python oracle_placebo.py --results r_keep.csv --out r_oracle_placebo.csv --resume
```

This script does not rerun the bare arm; it reads the `public_bare` row from `r_keep.csv`. Two generations per question.

### 4. Oracle + distractors and small placebo

```bash
python oracle_distractors.py --results r_keep.csv \
    --oracle r_oracle_placebo.csv --out r_distractors.csv --resume
```

The summary it prints is restricted to the items present in all four conditions.

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

**Cost.** One full Wikidata run (366 questions, rephrase plus four arms) is about 1,830 Sonnet calls.

---

## Citation

```bibtex
@inproceedings{necessity2027,
  title  = {Necessity, Not Provision: Counterfactual Attribution for
            Retrieval-Augmented {SPARQL} Generation},
  author = {Anonymous},
  year   = {2027},
  note   = {Under review}
}
```

## License

MIT for the code. Instruct-to-SPARQL and Wikidata content remain under their own licenses.
