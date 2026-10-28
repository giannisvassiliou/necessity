#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recompute the Wikidata statistics from the CSVs written by the experiment
scripts. No LLM or SPARQL calls; runs in seconds.

    python analysis.py --results r_keep.csv \
        --oracle r_oracle_placebo.csv --distractors r_distractors.csv \
        --queries query.json --json stats.json

--oracle and --distractors are optional; their sections are skipped if absent.

Sections
  1. Attribution and answer quality, full pipeline      (Table 1)
  2. Correlation with element-level F1 gain             (Table 2, App. A.3)
     - Pearson r for necessity, provision, divergence, unsupplied divergence
     - Williams' T2 test for dependent correlations (Steiger, 1980)
     - partial correlations, regression, nested-model F test
     - median split on necessity
     - answer-level correlations
  3. Attribution by candidate recall                    (Fig. 2, Table 7)
  4. Decoding-noise floor                               (App. A.3)
  5. Oracle and placebo                                 (Table 1)
  6. 2x2: reference presence x candidate-set size       (Table 3)
  7. Per-question anonymisation                         (Table 12, App. A.7)

Definitions (identifiers are extracted lexically, \\b[QP]\\d+\\b, exactly as in
the experiment scripts):
  C(Q)           identifier set of query Q
  element F1     F1 between C(Q) and C(Q*) of the reference query
  gain           element F1 (method arm) - element F1 (bare arm)
  divergence d   |C(Qm) \\ C(Qb)| / |C(Qm)|
  mu_new         |C(Qm) \\ C(Qb) \\ Cand| / |C(Qm)|     (d = necessity + mu_new)
  phi            |C(Qb') \\ C(Qb)| / |C(Qb')| between two independent bare runs

Confidence intervals are 95% percentile intervals from a paired bootstrap over
questions (default 10,000 resamples, seed 0).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

RE_ID = re.compile(r"\b[QP]\d+\b")
RESULTS: Dict[str, object] = {}


# =============================================================================
# Loading
# =============================================================================

def load_rows(path: Path) -> Dict[str, Dict[str, dict]]:
    """{condition: {query_index: row}}; later rows win (handles --resume reruns)."""
    out: Dict[str, Dict[str, dict]] = {}
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["condition"], {})[str(r["query_index"])] = r
    return out


def ids(text: Optional[str]) -> Set[str]:
    return set(RE_ID.findall(text or ""))


def idlist(field: Optional[str]) -> Set[str]:
    return {x for x in (field or "").split("|") if x}


def num(r: dict, k: str, default: float = 0.0) -> float:
    try:
        return float(r.get(k, ""))
    except (TypeError, ValueError):
        return default


def flag(r: dict, k: str) -> bool:
    return str(r.get(k, "0")).strip() in ("1", "True", "true")


# =============================================================================
# Statistics helpers
# =============================================================================

try:
    from scipy import stats as _sps
except Exception:          # scipy optional: fall back to normal approximation
    _sps = None


def p_from_t(t: float, df: int) -> float:
    if not math.isfinite(t):
        return 0.0
    if _sps is not None and df > 0:
        return float(2 * _sps.t.sf(abs(t), df))
    return math.erfc(abs(t) / math.sqrt(2))


def p_from_f(f: float, d1: int, d2: int) -> float:
    if _sps is not None:
        return float(_sps.f.sf(f, d1, d2))
    return p_from_t(math.sqrt(max(f, 0.0)), d2)   # d1 == 1 here


def pearson(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, int]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan"), n
    r = float(np.corrcoef(x, y)[0, 1])
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return r, p_from_t(t, n - 2), n


def williams_t2(r_jk: float, r_jh: float, r_kh: float, n: int) -> Tuple[float, float]:
    """Compare r(j,k) with r(j,h) sharing variable j (Williams' T2; Steiger 1980).

    Here j = gain, k and h are the two predictors, r_kh their correlation.
    """
    det = 1 - r_jk ** 2 - r_jh ** 2 - r_kh ** 2 + 2 * r_jk * r_jh * r_kh
    rbar = (r_jk + r_jh) / 2
    denom = 2 * ((n - 1) / (n - 3)) * det + rbar ** 2 * (1 - r_kh) ** 3
    t = (r_jk - r_jh) * math.sqrt((n - 1) * (1 + r_kh) / denom)
    return t, p_from_t(t, n - 3)


def partial_r(r_xy: float, r_xz: float, r_yz: float, n: int) -> Tuple[float, float]:
    den = (1 - r_xz ** 2) * (1 - r_yz ** 2)
    if not (den > 1e-12):
        return float("nan"), float("nan")
    r = (r_xy - r_xz * r_yz) / math.sqrt(den)
    t = r * math.sqrt((n - 3) / max(1e-12, 1 - r * r))
    return r, p_from_t(t, n - 3)


def zscore(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, float)
    sd = a.std(ddof=1)
    return (a - a.mean()) / sd if sd > 0 else a * 0


def ols(y: np.ndarray, cols: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float]:
    """OLS with intercept -> (coefficients, covariance matrix, R^2)."""
    X = np.column_stack([np.ones(len(y))] + cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return beta, cov, r2


class Boot:
    """Paired percentile bootstrap over questions."""

    def __init__(self, n_boot: int, seed: int):
        self.n_boot, self.seed = n_boot, seed

    def ci(self, values: Sequence[float]) -> Tuple[float, float, float]:
        v = np.asarray(values, float)
        if len(v) == 0:
            return float("nan"), float("nan"), float("nan")
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(v), size=(self.n_boot, len(v)))
        means = v[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return float(v.mean()), float(lo), float(hi)

    def diff(self, a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
        """mean(a - b), paired."""
        return self.ci(np.asarray(a, float) - np.asarray(b, float))

    def group_diff(self, values: Sequence[float], groups: Sequence[bool]
                   ) -> Tuple[float, float, float]:
        """mean(values | group) - mean(values | not group), resampling questions."""
        v, g = np.asarray(values, float), np.asarray(groups, bool)
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(v), size=(self.n_boot, len(v)))
        vv, gg = v[idx], g[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            hi = (vv * gg).sum(1) / gg.sum(1)
            lo = (vv * ~gg).sum(1) / (~gg).sum(1)
        d = hi - lo
        d = d[np.isfinite(d)]
        point = float(v[g].mean() - v[~g].mean())
        a, b = np.percentile(d, [2.5, 97.5])
        return point, float(a), float(b)

    def corr_diff(self, x1, x2, y) -> Tuple[float, float, float]:
        """r(x1,y) - r(x2,y), resampling questions."""
        x1, x2, y = (np.asarray(a, float) for a in (x1, x2, y))
        rng = np.random.default_rng(self.seed)
        n = len(y)
        out = []
        for _ in range(self.n_boot):
            i = rng.integers(0, n, n)
            a, b, c = x1[i], x2[i], y[i]
            if a.std() == 0 or b.std() == 0 or c.std() == 0:
                continue
            out.append(np.corrcoef(a, c)[0, 1] - np.corrcoef(b, c)[0, 1])
        point = pearson(x1, y)[0] - pearson(x2, y)[0]
        lo, hi = np.percentile(out, [2.5, 97.5])
        return float(point), float(lo), float(hi)


def fmt_ci(t: Tuple[float, float, float], d: int = 3, sign: bool = False) -> str:
    m, lo, hi = t
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]"


def stars(p: float) -> str:
    return "*" if (p == p and p < 0.05) else " "


def header(title: str) -> None:
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def record(key: str, value) -> None:
    RESULTS[key] = value


# =============================================================================
# Per-question measures for the deployed pipeline
# =============================================================================

def element_f1(gen: Set[str], gold: Set[str]) -> float:
    if not gen or not gold:
        return 0.0
    tp = len(gen & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(gen), tp / len(gold)
    return 2 * p * r / (p + r)


def build_items(rows, gold: Dict[str, str], keep: Optional[Set[str]]) -> List[dict]:
    pm, pb = rows.get("public_method", {}), rows.get("public_bare", {})
    items = []
    n_mismatch = 0
    for q in sorted(set(pm) & set(pb), key=int):
        if keep is not None and q not in keep:
            continue
        m, b = pm[q], pb[q]
        cm, cb = ids(m.get("sparql")), ids(b.get("sparql"))
        cg = ids(gold.get(q, ""))
        mem = idlist(m.get("memory_ids"))
        nec_ids = idlist(m.get("necessary_ids"))
        it = dict(q=q, cm=cm, cb=cb, cg=cg,
                  n_el=int(num(m, "n_elements")),
                  n_cand=num(m, "n_candidates"),
                  n_nec=len(nec_ids),
                  nec=num(m, "necessity"), prov=num(m, "provision"),
                  red=num(m, "redundancy"), mem=num(m, "memory"),
                  recall=num(m, "gold_recall"),
                  f1_m=num(m, "f1"), f1_b=num(b, "f1"),
                  ex_m=flag(m, "executed"), ex_b=flag(b, "executed"),
                  gold_ok=flag(b, "gold_ok") if "gold_ok" in b else True)
        it["ef_m"], it["ef_b"] = element_f1(cm, cg), element_f1(cb, cg)
        it["gain"] = it["ef_m"] - it["ef_b"]
        if cm:
            it["delta"] = len(cm - cb) / len(cm)
            it["mu_new"] = len(mem - cb) / len(cm)
            if abs(len(nec_ids) / len(cm) - it["nec"]) > 1e-3:
                n_mismatch += 1
        else:
            it["delta"] = it["mu_new"] = 0.0
        items.append(it)
    if n_mismatch:
        print(f"[warning] {n_mismatch} items: necessity column disagrees with "
              f"necessary_ids / |C(Qm)| recomputed from the SPARQL")
    return items


# =============================================================================
# Sections
# =============================================================================

def s1_table1(items: List[dict], boot: Boot) -> None:
    header("1. Attribution and answer quality, full pipeline (Table 1)")
    att = [i for i in items if i["n_el"] > 0]
    acc = [i for i in items if i["gold_ok"]]
    print(f"attribution items (|C(Qm)|>0): {len(att)}     answer items (gold executed): {len(acc)}")
    out = {}
    for k, lab in (("n_cand", "|Cand|"), ("prov", "provision"), ("nec", "necessity"),
                   ("red", "redundancy"), ("mem", "memory")):
        out[k] = boot.ci([i[k] for i in att])
        print(f"  {lab:<12} {fmt_ci(out[k], 3 if k != 'n_cand' else 1)}")
    ratio = out["prov"][0] / out["nec"][0] if out["nec"][0] else float("nan")
    print(f"  prov/nec     {ratio:.2f}")
    for k, lab in (("f1_b", "answer F1, bare"), ("f1_m", "answer F1, method"),
                   ("ef_b", "element F1, bare"), ("ef_m", "element F1, method")):
        src = acc if k.startswith("f1") else att
        out[k] = boot.ci([i[k] for i in src])
        print(f"  {lab:<20} {fmt_ci(out[k])}")
    out["f1_gain"] = boot.diff([i["f1_m"] for i in acc], [i["f1_b"] for i in acc])
    print(f"  answer F1 gain       {fmt_ci(out['f1_gain'], sign=True)}")
    exe = [i for i in items]
    print(f"  executed: bare {np.mean([i['ex_b'] for i in exe]):.3f}  "
          f"method {np.mean([i['ex_m'] for i in exe]):.3f}")

    print("\n  ratio provision/necessity on subsets")
    for lab, sub in (("|C|>=3", [i for i in att if i["n_el"] >= 3]),
                     ("method executed", [i for i in att if i["ex_m"]]),
                     ("both executed", [i for i in att if i["ex_m"] and i["ex_b"]])):
        if sub:
            pv, nv = np.mean([i["prov"] for i in sub]), np.mean([i["nec"] for i in sub])
            print(f"    {lab:<18} n={len(sub):<4} {pv / nv if nv else float('nan'):.2f}")
    record("table1", {k: list(v) if isinstance(v, tuple) else v for k, v in out.items()}
           | {"prov_nec_ratio": ratio, "n_att": len(att), "n_acc": len(acc)})


def s2_correlations(items: List[dict], boot: Boot) -> None:
    header("2. Correlation with per-question element-level F1 gain (Table 2, App. A.3)")
    att = [i for i in items if i["n_el"] > 0]
    subsets = [
        ("All items", att),
        ("|C|>=3", [i for i in att if i["n_el"] >= 3]),
        ("Method arm executed", [i for i in att if i["ex_m"]]),
        ("Both arms executed", [i for i in att if i["ex_m"] and i["ex_b"]]),
        ("Reference |C*|>=3", [i for i in att if len(i["cg"]) >= 3]),
    ]
    print(f"{'subset':<22}{'n':>5}{'nu':>9}{'prov':>9}{'delta':>9}{'mu_new':>9}"
          f"{'  T2 nu-prov':>15}{'  T2 nu-delta':>15}")
    print("-" * 100)
    tab = {}
    for lab, sub in subsets:
        if len(sub) < 5:
            continue
        g = [i["gain"] for i in sub]
        rn, pn, n = pearson([i["nec"] for i in sub], g)
        rp, pp, _ = pearson([i["prov"] for i in sub], g)
        rd, pd, _ = pearson([i["delta"] for i in sub], g)
        rm, pmn, _ = pearson([i["mu_new"] for i in sub], g)
        r_np = pearson([i["nec"] for i in sub], [i["prov"] for i in sub])[0]
        r_nd = pearson([i["nec"] for i in sub], [i["delta"] for i in sub])[0]
        t1, p1 = williams_t2(rn, rp, r_np, n)
        t2, p2 = williams_t2(rn, rd, r_nd, n)
        print(f"{lab:<22}{n:>5}{rn:>+8.3f}{stars(pn)}{rp:>+8.3f}{stars(pp)}"
              f"{rd:>+8.3f}{stars(pd)}{rm:>+8.3f}{stars(pmn)}"
              f"{t1:>+9.2f} p={p1:<5.3f}{t2:>+9.2f} p={p2:<5.3f}")
        tab[lab] = dict(n=n, r_nu=rn, p_nu=pn, r_prov=rp, p_prov=pp, r_delta=rd,
                        p_delta=pd, r_mu_new=rm, p_mu_new=pmn,
                        williams_nu_prov=(t1, p1), williams_nu_delta=(t2, p2))
    print("  * p < 0.05.  T2 = Williams' test for dependent correlations (Steiger, 1980).")
    record("table2", tab)

    # ---- all-items extras -------------------------------------------------
    g = np.array([i["gain"] for i in att])
    nu = np.array([i["nec"] for i in att])
    pv = np.array([i["prov"] for i in att])
    de = np.array([i["delta"] for i in att])
    mn = np.array([i["mu_new"] for i in att])
    n = len(att)

    print("\n  Necessity vs divergence (all items)")
    r_nd = pearson(nu, de)[0]
    r_ng, r_dg = pearson(nu, g)[0], pearson(de, g)[0]
    pr1, pp1 = partial_r(r_ng, r_nd, r_dg, n)
    pr2, pp2 = partial_r(r_dg, r_nd, r_ng, n)
    print(f"    r(nu, delta)                       {r_nd:+.3f}")
    print(f"    partial r(nu, gain | delta)        {pr1:+.3f}  p={pp1:.3g}")
    print(f"    partial r(delta, gain | nu)        {pr2:+.3f}  p={pp2:.3g}")
    cd = boot.corr_diff(nu, de, g)
    print(f"    r(nu,gain) - r(delta,gain)         {fmt_ci(cd, sign=True)}")
    print(f"    mean mu_new                        {mn.mean():.3f}")
    n_div = sum(len(i["cm"] - i["cb"]) for i in att)
    n_sup = sum(i["n_nec"] for i in att)
    if n_div:
        print(f"    share of divergent ids that were supplied   {n_sup / n_div:.3f}")
    eq = np.mean([abs(i["nec"] - i["delta"]) < 1e-9 for i in att])
    print(f"    share of items with nu == delta             {eq:.3f}")

    # regression gain ~ nu + mu_new (standardised)
    beta = [float("nan")] * 3
    t_d = float("nan")
    if mn.std() > 0 and nu.std() > 0 and g.std() > 0:
        y = zscore(g)
        beta, cov, _ = ols(y, [zscore(nu), zscore(mn)])
        var_d = cov[1, 1] + cov[2, 2] - 2 * cov[1, 2]
        t_d = (beta[1] - beta[2]) / math.sqrt(var_d) if var_d > 0 else float("nan")
        print(f"    regression gain ~ nu + mu_new (standardised): "
              f"b_nu={beta[1]:+.3f}  b_mu_new={beta[2]:+.3f}  t(diff)={t_d:.2f}")
    else:
        print("    regression gain ~ nu + mu_new: skipped (a variable has zero variance)")

    # nested models
    _, _, r2_d = ols(g, [de])
    _, _, r2_dn = ols(g, [de, nu])
    _, _, r2_n = ols(g, [nu])
    resid_var = max(1e-12, (1 - r2_dn) / (n - 3))
    F1 = max(0.0, (r2_dn - r2_d) / resid_var)
    F2 = max(0.0, (r2_dn - r2_n) / resid_var)
    print(f"    R^2: delta {r2_d:.3f} -> delta+nu {r2_dn:.3f}  "
          f"(dR^2={r2_dn - r2_d:.3f}, F={F1:.1f}, p={p_from_f(F1, 1, n - 3):.3g})")
    print(f"    R^2: nu {r2_n:.3f} -> nu+delta {r2_dn:.3f}  "
          f"(dR^2={r2_dn - r2_n:.3f}, F={F2:.1f}, p={p_from_f(F2, 1, n - 3):.3g})")

    # median split
    med = float(np.median(nu))
    hi = nu > med
    print(f"\n  Median split on necessity (median {med:.3f})")
    print(f"    high-necessity gain {g[hi].mean():+.3f} (n={hi.sum()})   "
          f"low-necessity gain {g[~hi].mean():+.3f} (n={(~hi).sum()})")
    md = boot.group_diff(g, hi)
    print(f"    difference          {fmt_ci(md, sign=True)}")
    print(f"    provision: high {pv[hi].mean():.3f}   low {pv[~hi].mean():.3f}")
    medd = float(np.median(de))
    dd = boot.group_diff(g, de > medd)
    print(f"    same split on divergence: difference {fmt_ci(dd, sign=True)}")

    # answer-level
    acc = [i for i in att if i["gold_ok"]]
    ag = [i["f1_m"] - i["f1_b"] for i in acc]
    ran, pan, na = pearson([i["nec"] for i in acc], ag)
    rap, pap, _ = pearson([i["prov"] for i in acc], ag)
    print(f"\n  Answer-level F1 gain (n={na}): r(nu)={ran:+.3f} p={pan:.3g}   "
          f"r(prov)={rap:+.3f} p={pap:.3g}")
    record("table2_extras", dict(
        partial_nu_given_delta=(pr1, pp1), partial_delta_given_nu=(pr2, pp2),
        corr_diff_nu_minus_delta=cd, mean_mu_new=float(mn.mean()),
        regression=dict(b_nu=float(beta[1]), b_mu_new=float(beta[2]), t_diff=t_d),
        r2=dict(delta=r2_d, nu=r2_n, both=r2_dn, F_add_nu=F1, F_add_delta=F2),
        median_split=dict(median=med, high=float(g[hi].mean()), low=float(g[~hi].mean()),
                          diff=md, prov_high=float(pv[hi].mean()), prov_low=float(pv[~hi].mean())),
        answer_level=dict(n=na, r_nu=ran, p_nu=pan, r_prov=rap, p_prov=pap)))


def s3_recall(items: List[dict], boot: Boot) -> None:
    header("3. Attribution by candidate recall (Fig. 2, Table 7)")
    att = [i for i in items if i["n_el"] > 0 and i["cg"]]
    bins = [("0-.25", 0, .25), (".25-.5", .25, .5), (".5-.75", .5, .75),
            (".75-1", .75, 1.0)]
    print(f"{'recall bin':<12}{'n':>5}{'|Cand|':>9}{'prov':>8}{'nec':>8}{'red':>8}{'mem':>8}")
    rows = []
    for lab, lo, hi in bins:
        b = [i for i in att if lo <= i["recall"] < hi]
        rows.append((lab, b))
    rows.append(("1.0", [i for i in att if i["recall"] >= 1.0 - 1e-12]))
    out = {}
    for lab, b in rows:
        if not b:
            continue
        m = {k: float(np.mean([i[k] for i in b])) for k in ("n_cand", "prov", "nec", "red", "mem")}
        out[lab] = m | {"n": len(b)}
        print(f"{lab:<12}{len(b):>5}{m['n_cand']:>9.1f}{m['prov']:>8.3f}{m['nec']:>8.3f}"
              f"{m['red']:>8.3f}{m['mem']:>8.3f}")
    rec = [i["recall"] for i in att]
    rp, pp, _ = pearson(rec, [i["prov"] for i in att])
    rn, pn, _ = pearson(rec, [i["nec"] for i in att])
    print(f"  r(recall, provision) = {rp:+.3f} (p={pp:.3g})   "
          f"r(recall, necessity) = {rn:+.3f} (p={pn:.3g})")
    perf = [i for i in att if i["recall"] >= 1.0 - 1e-12]
    imp = [i for i in att if i["recall"] < 1.0 - 1e-12]
    for k in ("prov", "nec", "red", "mem"):
        a, b = [i[k] for i in perf], [i[k] for i in imp]
        if len(a) > 1 and len(b) > 1:
            se = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
            t = (st.mean(a) - st.mean(b)) / se if se else float("nan")
            print(f"    {k:<5} perfect {fmt_ci(boot.ci(a))}   imperfect {fmt_ci(boot.ci(b))}"
                  f"   Welch t={t:.2f}")
    record("table7", dict(bins=out, r_recall_prov=(rp, pp), r_recall_nec=(rn, pn)))


def s4_noise(rows, items: List[dict], boot: Boot) -> None:
    header("4. Decoding-noise floor (two independent bare runs)")
    pb, ab = rows.get("public_bare", {}), rows.get("anon_bare", {})
    keep = {i["q"] for i in items}
    phi, same = [], []
    for q in sorted(set(pb) & set(ab) & keep, key=int):
        c1, c2 = ids(pb[q].get("sparql")), ids(ab[q].get("sparql"))
        same.append(c1 == c2)
        if c2:
            phi.append(len(c2 - c1) / len(c2))
    if not phi:
        print("  no anon_bare rows: skipped")
        return
    print("  Second run = anon_bare: the bare arm sees no retrieved context, so its prompt")
    print("  is identical to public_bare; only the API call differs.")
    print(f"  identical identifier sets: {np.mean(same):.3f} of {len(same)} questions")
    ci = boot.ci(phi)
    print(f"  phi = {fmt_ci(ci)}  (n={len(phi)})")
    record("noise_floor", dict(phi=ci, n=len(phi), identical=float(np.mean(same))))


def s5_oracle(rows, ora, items: List[dict], boot: Boot) -> None:
    header("5. Oracle and placebo (Table 1)")
    base = {i["q"]: i for i in items}
    acc = {i["q"] for i in items if i["gold_ok"]}
    out = {}
    for cond in ("oracle", "placebo"):
        rs = {q: r for q, r in ora.get(cond, {}).items() if q in base}
        if not rs:
            continue
        att = [r for r in rs.values() if num(r, "n_elements") > 0]
        f1 = [num(r, "f1") for q, r in rs.items() if q in acc]
        m = {k: boot.ci([num(r, k) for r in att])
             for k in ("n_candidates", "provision", "necessity", "redundancy", "memory")}
        m["f1"] = boot.ci(f1)
        out[cond] = m
        print(f"  {cond:<8} n_att={len(att):<4} |Cand|={m['n_candidates'][0]:.1f}  "
              f"prov={m['provision'][0]:.3f}  nec={fmt_ci(m['necessity'])}  "
              f"red={m['redundancy'][0]:.3f}  mem={m['memory'][0]:.3f}  "
              f"F1={fmt_ci(m['f1'])}  (n_F1={len(f1)})")
        nz = sum(1 for r in att if num(r, "necessity") > 0)
        print(f"           items with nonzero necessity: {nz}")
    o, p = ora.get("oracle", {}), ora.get("placebo", {})
    common = sorted(set(o) & set(p) & set(base), key=int)
    if common:
        d = boot.diff([num(o[q], "divergence") for q in common],
                      [num(p[q], "divergence") for q in common])
        print(f"\n  Jaccard distance between arms: oracle {np.mean([num(o[q],'divergence') for q in common]):.3f}"
              f"  placebo {np.mean([num(p[q],'divergence') for q in common]):.3f}"
              f"  difference {fmt_ci(d, sign=True)}")
        out["jaccard_oracle_minus_placebo"] = d
        both = [q for q in common if num(o[q], "n_elements") > 0 and base[q]["n_el"] > 0]
        dn = boot.diff([num(o[q], "necessity") for q in both], [base[q]["nec"] for q in both])
        print(f"  oracle - deployed necessity: {fmt_ci(dn, sign=True)} (n={len(both)})")
    record("oracle_placebo", out)


def s6_factorial(ora, dis, items: List[dict], boot: Boot) -> None:
    header("6. 2x2: reference presence x candidate-set size (Table 3)")
    base = {i["q"]: i for i in items}
    acc = {i["q"] for i in items if i["gold_ok"]}
    conds = {"oracle": ora.get("oracle", {}), "oracle_distractors": dis.get("oracle_distractors", {}),
             "placebo_small": dis.get("placebo_small", {}), "placebo": ora.get("placebo", {})}
    common = set(base)
    for c in conds.values():
        common &= set(c)
    common = sorted(common, key=int)
    if not common:
        print("  no items present in all four conditions: skipped")
        return
    acc_c = [q for q in common if q in acc]
    print(f"  items in all conditions: {len(common)}  (answer F1 over {len(acc_c)})")
    print(f"{'condition':<22}{'|Cand|':>8}{'F1':>8}{'prov':>8}{'nec':>8}{'red':>8}")
    tab = {}
    for c, rs in conds.items():
        att = [rs[q] for q in common if num(rs[q], "n_elements") > 0]
        m = dict(n_cand=np.mean([num(rs[q], "n_candidates") for q in common]),
                 f1=np.mean([num(rs[q], "f1") for q in acc_c]),
                 prov=np.mean([num(r, "provision") for r in att]),
                 nec=np.mean([num(r, "necessity") for r in att]),
                 red=np.mean([num(r, "redundancy") for r in att]))
        tab[c] = {k: float(v) for k, v in m.items()}
        print(f"{c:<22}{m['n_cand']:>8.1f}{m['f1']:>8.3f}{m['prov']:>8.3f}{m['nec']:>8.3f}{m['red']:>8.3f}")

    def f1d(a, b):
        A = conds[a] if a in conds else None
        B = conds[b] if b in conds else None
        va = [num(A[q], "f1") if A else base[q]["f1_m"] for q in acc_c]
        vb = [num(B[q], "f1") if B else base[q]["f1_m"] for q in acc_c]
        return boot.diff(va, vb)

    def necd(a, b):
        A = conds.get(a); B = conds.get(b)
        qs = [q for q in common
              if (num(A[q], "n_elements") if A else base[q]["n_el"]) > 0
              and (num(B[q], "n_elements") if B else base[q]["n_el"]) > 0]
        va = [num(A[q], "necessity") if A else base[q]["nec"] for q in qs]
        vb = [num(B[q], "necessity") if B else base[q]["nec"] for q in qs]
        return boot.diff(va, vb)

    print("\n  paired differences (95% bootstrap CI)")
    eff = {
        "add reference ids at deployed size (oracle+distr - deployed)":
            (f1d("oracle_distractors", "deployed"), necd("oracle_distractors", "deployed")),
        "shrink set with perfect recall (oracle - oracle+distr)":
            (f1d("oracle", "oracle_distractors"), necd("oracle", "oracle_distractors")),
        "shrink placebo set (placebo_small - placebo)":
            (f1d("placebo_small", "placebo"), None),
    }
    for lab, (f, n_) in eff.items():
        line = f"    {lab}\n        F1 {fmt_ci(f, sign=True)}"
        if n_:
            line += f"   necessity {fmt_ci(n_, sign=True)}"
        print(line)
    record("table3", dict(means=tab, effects={k: dict(f1=v[0], nec=v[1]) for k, v in eff.items()}))


def s7_anon(rows, items: List[dict], boot: Boot) -> None:
    header("7. Per-question anonymisation (Table 12, App. A.7)")
    am, ab = rows.get("anon_method", {}), rows.get("anon_bare", {})
    base = {i["q"]: i for i in items}
    qs = sorted(set(am) & set(base), key=int)
    if not qs:
        print("  no anon_method rows: skipped")
        return
    acc = [q for q in qs if base[q]["gold_ok"]]
    f1a = boot.ci([num(am[q], "f1") for q in acc])
    f1ab = boot.ci([num(ab[q], "f1") for q in acc if q in ab])
    d = boot.diff([num(am[q], "f1") for q in acc], [base[q]["f1_m"] for q in acc])
    att = [am[q] for q in qs if num(am[q], "n_elements") > 0]
    print(f"  answer F1: public method {np.mean([base[q]['f1_m'] for q in acc]):.3f}   "
          f"anon method {fmt_ci(f1a)}   anon bare {f1ab[0]:.3f}")
    print(f"  anon - public method F1: {fmt_ci(d, sign=True)}")
    print(f"  execution rate: anon method {np.mean([flag(am[q], 'executed') for q in qs]):.3f}   "
          f"anon bare {np.mean([flag(ab[q], 'executed') for q in qs if q in ab]):.3f}")
    print(f"  mean |C|: public method {np.mean([base[q]['n_el'] for q in qs]):.2f}   "
          f"anon method {np.mean([num(am[q], 'n_elements') for q in qs]):.2f}")
    print(f"  attribution (not interpreted in the paper): prov "
          f"{np.mean([num(r,'provision') for r in att]):.3f}  nec "
          f"{np.mean([num(r,'necessity') for r in att]):.3f}  red "
          f"{np.mean([num(r,'redundancy') for r in att]):.3f}")
    lam = np.mean([num(am[q], "memory_override") for q in qs])
    print(f"  override rate lambda: {lam:.3f}")
    record("anonymisation", dict(f1_anon_method=f1a, diff_vs_public=d, lam=float(lam)))


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("r_keep.csv"))
    ap.add_argument("--oracle", type=Path, default=Path("r_oracle_placebo.csv"))
    ap.add_argument("--distractors", type=Path, default=Path("r_distractors.csv"))
    ap.add_argument("--queries", type=Path, default=Path("query.json"))
    ap.add_argument("--gold-filter", type=Path, default=None,
                    help="optional JSON list of question ids to restrict to")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="also write all numbers here")
    args = ap.parse_args()

    if not args.results.exists():
        print(f"ERROR: {args.results} not found"); return 2
    gold = {str(k): v for k, v in json.loads(args.queries.read_text(encoding="utf-8")).items()}
    keep = None
    if args.gold_filter and args.gold_filter.exists():
        keep = {str(i) for i in json.loads(args.gold_filter.read_text(encoding="utf-8"))}

    rows = load_rows(args.results)
    ora = load_rows(args.oracle) if args.oracle.exists() else {}
    dis = load_rows(args.distractors) if args.distractors.exists() else {}
    boot = Boot(args.n_boot, args.seed)
    if _sps is None:
        print("[note] scipy not installed: p-values use a normal approximation")

    items = build_items(rows, gold, keep)
    print(f"{len(items)} questions with both public arms in {args.results}")

    s1_table1(items, boot)
    s2_correlations(items, boot)
    s3_recall(items, boot)
    s4_noise(rows, items, boot)
    if ora:
        s5_oracle(rows, ora, items, boot)
    else:
        print(f"\n[{args.oracle} not found: oracle/placebo section skipped]")
    if ora and dis:
        s6_factorial(ora, dis, items, boot)
    else:
        print(f"[{args.distractors} not found: 2x2 section skipped]")
    s7_anon(rows, items, boot)

    if args.json:
        def conv(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, tuple):
                return list(o)
            raise TypeError(type(o))
        args.json.write_text(json.dumps(RESULTS, indent=2, default=conv), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
