# Cross-patient replication: pre-specified analysis (written 2026-10-28, before any replication model was trained)

Data: GSE311609 (17 donors, 19 Xenium FFPE breast sections, 280-gene base panel; B1 has three serial sections)
and Janesick et al. Sample #2 (GSM7780155, 280-gene panel). Discovery = Rep1 (and Rep2, same tumour).

Pipeline: unchanged `run_xenium.py --section other` (marker-based lineages, CEACAM6/SERPINA3/TACSTD2 median split
into target vs reference tumour states, top-15 DE read-out, k = 10 graph, 5 composition niches, same model
hyper-parameters, steps = 500). One model per section (the "single model" column of the discovery runs is the
comparator). GSE311609 has no Xenium clustering, so clusters come from `prepare_gse311609.py` (PCA 20, k-means 20).
Only change to shared code: lineage markers absent from the 280-gene panel (KRT15, DERL3) are skipped.

Quality gate (applied to every section before looking at rankings): >= 2,000 malignant-lineage cells and
>= 500 cells in each tumour state. Sections failing it are reported and excluded from replication statistics.

Units: donor. B1 is summarised by B1_1 (primary); B1_2 and B1_3 are reported as within-tumour technical replicates.

Candidates: the 265 genes that are candidates in Rep1 (U297) and on the 280-gene panel, minus that section's read-out.

Replication endpoints (per donor, vs Rep1 single-model and vs Rep1 3-model ensemble):
1. Spearman correlation of GNN scores with Rep1 on shared candidates; top-25 overlap with hypergeometric p.
2. Same for niche-stratified LR-communication and DE (to show whether the GNN is more or less reproducible).
3. Evidence enrichment (OT-breast, OT-clinical, DepMap-breast; same files as the JBI evaluation): AUPRC and top-25 hits.
Across donors: median and IQR; sign test of (GNN AUPRC > prevalence) across donors; no patient-level claim beyond
these donors. Subtype is not available (no receptor status); ESR1/PGR/ERBB2 expression is reported as inferred only.
