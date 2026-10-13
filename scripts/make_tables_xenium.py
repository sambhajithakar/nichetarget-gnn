"""Markdown tables for the manuscript (Table 3, Table S4) from results/xenium_evaluation."""
import sys
import pandas as pd

ev = pd.read_csv(sys.argv[1] + "/evaluation_all.csv")
rc = pd.read_csv(sys.argv[1] + "/replicate_concordance.csv")
MAIN = ["DE", "LR-communication (niche-stratified)", "NicheTarget-GNN (single model)",
        "NicheTarget-GNN (ensemble of 3)", "Hybrid (GNN ensemble + LR-communication)"]
SETS = [("OT_breast", "Open Targets breast"), ("OT_clinical", "Clinical-stage drug target"), ("DepMap_breast", "DepMap breast dependency")]


def p(v):
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def table(methods):
    h = "| Section | Method | " + " | ".join(f"{l} AUPRC | {l} Top-25 (p)" for _, l in SETS) + " | Spearman with OT score |"
    out = [h, "|" + "---|" * (2 + 2 * len(SETS) + 1)]
    for _, r in ev[ev.method.isin(methods)].iterrows():
        cells = [r.section.replace("xenium_rep", "Rep"), r.method]
        for s, _ in SETS:
            cells += [f"{r[s + '_AUPRC']:.3f} ({r[s + '_prevalence']:.3f})", f"{int(r[s + '_top25'])} ({p(r[s + '_top25_p'])})"]
        cells.append(f"{r.spearman_with_OT_score:.3f}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


print("TABLE3\n" + table(MAIN))
print("\nTABLES4\n" + table(list(dict.fromkeys(ev.method))))
out = ["| Method | Spearman Rep1 vs Rep2 | Top-25 overlap | Shared top-25 genes |", "|---|---|---|---|"]
for _, r in rc.iterrows():
    out.append(f"| {r.method} | {r.spearman_rep1_rep2:.2f} | {r.top25_overlap} | {r.shared_top_genes.replace(';', ', ')} |")
print("\nCONCORDANCE\n" + "\n".join(out))
out = ["| Section | Method | Top 25 genes |", "|---|---|---|"]
for _, r in ev[ev.method.isin(MAIN)].iterrows():
    out.append(f"| {r.section.replace('xenium_rep', 'Rep')} | {r.method} | {r.top25.replace(';', ', ')} |")
print("\nTOP25\n" + "\n".join(out))
