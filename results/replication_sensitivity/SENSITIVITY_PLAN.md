# Sensitivity analysis: malignant-specific read-out (written 28 Oct 2026, before any sensitivity run)

Status: POST HOC. Motivated by the pre-specified replication (results/replication/summary), in which the
GNN ranking did not replicate and an exploratory diagnostic (scripts/readout_purity.py) linked failure to
read-out programmes dominated by stromal/immune genes (spillover into tumour cells at tumour-stroma borders).

Single change: `--readout-rule malignant_specific`. The 15 read-out genes are the top target-vs-reference DE
genes among genes whose mean expression in malignant cells exceeds their mean in every non-malignant lineage.
Everything else (graph, niches, lineage labelling, tumour-state split, candidates, 500 steps, single model,
seed) is unchanged.

Runs: discovery Rep1 (single model, same rule) and the 17 donors that passed the original gate
(B1_1, B2-B15, B17, Sample #2). B1_2/B1_3 (replicates) and B16 (failed gate) are not re-run.

Endpoints (same as the pre-specified analysis): per donor, Spearman and top-25 overlap of the GNN ranking with
the discovery ranking; AUPRC vs prevalence for Open Targets breast, clinical-stage targets and DepMap
dependencies; across donors median/IQR and one-sided sign tests. Read-out purity is recomputed.

Reporting rule: both analyses are reported. The pre-specified result stays the primary result whatever the
sensitivity analysis shows; the sensitivity result is labelled post hoc.
