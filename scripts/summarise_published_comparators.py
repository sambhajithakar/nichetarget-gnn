"""Tables (manuscript Table 1 / 3 / S5 format, new rows only) for the published comparators.

    python scripts/summarise_published_comparators.py --root results/published_comparators
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

GENES = ["L1", "R1", "L2", "R2", "D1"]


def fmt_rank(x):
    return f"{x:g}"


def sim_table(root):
    rows, per_seed = [], []
    for tag, scen in [("full-lr", "Complete LR"), ("hidden-L1-R1", "L1–R1 missing")]:
        path = os.path.join(root, "simulation", f"all_results_{tag}.json")
        if not os.path.exists(path):
            continue
        R = json.load(open(path))
        methods = list(R[0]["metrics"].keys())
        for m in methods:
            v = [r["metrics"][m] for r in R]
            au = np.array([x["auprc"] for x in v])
            top25 = [sum(rk <= 25 for rk in x["ranks"].values()) for x in v]
            rows.append({"scenario": scen, "method": m, "n_tissues": len(v),
                         "AUPRC_mean": au.mean(), "AUPRC_sd": au.std(ddof=1),
                         "P@10_mean": np.mean([x["p@10"] for x in v]),
                         **{f"median_rank_{g}": float(np.median([x["ranks"][g] for x in v])) for g in GENES},
                         "top25_mean": float(np.mean(top25))})
            for r, x, t in zip(R, v, top25):
                per_seed.append({"scenario": scen, "seed": r["seed"], "method": m, "auprc": x["auprc"],
                                 "p@10": x["p@10"], **{f"rank_{g}": x["ranks"][g] for g in GENES}, "top25": t})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, "simulation", "table1_format_published_comparators.csv"), index=False)
    pd.DataFrame(per_seed).to_csv(os.path.join(root, "simulation", "per_seed_metrics.csv"), index=False)
    md = ["| Scenario | Method | AUPRC | P@10 | L1 | R1 | L2 | R2 | D1 | Top-25 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r.scenario} | {r.method} | {r.AUPRC_mean:.3f} ± {r.AUPRC_sd:.3f} | {r['P@10_mean']:.2f} | "
                  + " | ".join(fmt_rank(r[f'median_rank_{g}']) for g in GENES) + f" | {r.top25_mean:.1f} |")
    return "\n".join(md)


def xenium_table(root):
    ev_p = os.path.join(root, "xenium_evaluation", "evaluation_published_comparators.csv")
    if not os.path.exists(ev_p):
        return ""
    ev = pd.read_csv(ev_p)
    q = pd.read_csv(os.path.join(root, "xenium_evaluation", "evidence_tests_bh_published_comparators.csv"))
    qd = {(r.section, r.method, r.evidence_set): (r.q_bh_new_family, r.q_bh_extended_family) for _, r in q.iterrows()}

    def pf(p):
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    md = ["| Section | Method | Open Targets breast, AUPRC; top 25 (*p*; *q* new family; *q* extended) | "
          "Clinical-stage drug target, AUPRC; top 25 (*p*; *q*; *q* ext.) | DepMap breast dependency, AUPRC; top 25 "
          "(*p*; *q*; *q* ext.) | ρ with OT score |", "|---|---|---|---|---|---|"]
    for _, r in ev.iterrows():
        cells = []
        for s in ("OT_breast", "OT_clinical", "DepMap_breast"):
            qn, qe = qd[(r.section, r.method, s)]
            cells.append(f"{r[f'{s}_AUPRC']:.3f}; {int(r[f'{s}_top25'])} ({pf(r[f'{s}_top25_p'])}; {pf(qn)}; {pf(qe)})")
        sec = "Rep1" if r.section.endswith("rep1") else "Rep2"
        md.append(f"| {sec} | {r.method} | " + " | ".join(cells) + f" | {r.spearman_with_OT_score:.2f} |")
    prev = ev.groupby("section")[["OT_breast_prevalence", "OT_clinical_prevalence", "DepMap_breast_prevalence"]].first()
    md.append("")
    md.append("Prevalence: " + "; ".join(f"{s}: " + ", ".join(f"{c.replace('_prevalence', '')} {v:.3f}" for c, v in row.items())
                                      for s, row in prev.iterrows()))
    return "\n".join(md)


def perturbmap_table(root):
    p = os.path.join(root, "perturbmap_celcomen", "perturbmap_validation_celcomen.csv")
    if not os.path.exists(p):
        return ""
    v = pd.read_csv(p)
    ci = pd.read_csv(os.path.join(root, "perturbmap_celcomen", "spearman_bootstrap_ci_celcomen.csv"))
    import ast
    md = ["| Knockout (spots) | Method | Spearman (95% CI) | Rank (of candidates) | Percentile | IFN-γ genes predicted down "
          "| Five best-matching candidate genes |", "|---|---|---|---|---|---|---|"]
    for _, r in v.iterrows():
        c = ci[(ci.knockout == r.knockout)].iloc[0]
        top5 = ", ".join(ast.literal_eval(r.top5_genes)) if isinstance(r.top5_genes, str) else ""
        md.append(f"| {r.knockout} ({int(r.n_ko_spots)}) | {r.method} | {r.spearman_pred_vs_obs:.2f} "
                  f"({c.ci95_low:.2f} to {c.ci95_high:.2f}) | {int(r.rank_of_true_gene)} ({int(r.n_candidates)}) | "
                  f"{r.percentile:.2f} | {r.frac_ifng_genes_predicted_down:.2f} | {top5} |")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/published_comparators")
    a = ap.parse_args()
    out = ["## Simulation (Table 1 format)", sim_table(a.root), "", "## Xenium (Table 3 format)", xenium_table(a.root),
           "", "## Perturb-map (Table S5 format)", perturbmap_table(a.root)]
    open(os.path.join(a.root, "tables_published_comparators.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
