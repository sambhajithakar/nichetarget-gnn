# Revision analysis: contamination-aware read-out, purity QC gate, external validation (pre-specified)

Written 5 Nov 2026, before any model was trained on the external cohort and before the 180-gene discovery
and seed runs below. Committed to GitHub before the first such run so the timestamp is public.

## Why

The pre-specified cross-patient replication (results/replication/PRESPECIFIED.md, GSE311609 17 donors) was
negative for the GNN ranking (median top-25 overlap 1/25 with discovery). A post hoc analysis
(results/replication_sensitivity/SENSITIVITY_PLAN.md) linked the failure to read-out programmes dominated by
stromal/immune genes that spill into tumour cells, and showed that a malignant-specific read-out raised the
GNN overlap to 7/25 (DE baseline 11/25). Both earlier results remain as reported; the first stays primary
for that cohort. This plan tests the two post hoc ideas as fixed method components on data never used.

## Method components locked by this plan (no further tuning)

1. Contamination-aware read-out: `--readout-rule malignant_specific` exactly as in run_xenium.py at commit
   b9391b9 (read-out = top-15 target-vs-reference DE genes among genes whose malignant-lineage mean exceeds the
   mean of every other lineage).
2. Read-out purity QC gate: purity = fraction of the 15 read-out genes most expressed in the malignant lineage
   (scripts/readout_purity.py). Gate: purity >= 0.5 = "reliable read-out"; < 0.5 = "flag: contaminated
   read-out, ranking not interpretable". Threshold fixed from the 17-donor analysis (0.5 split already reported).

## External cohort

GSE350571 (Xenium Prime 5K, human primary breast tumours): P17 (GSM10116352), P22 (GSM10116353),
P24 (GSM10116354), P26 (GSM10116355), P29 (GSM10116356), P32 (GSM10116357). Six tumours, one section each,
not seen by any earlier analysis. Receptor status is not used (not known to us); no subtype is assumed.
Files used: `*_cell_feature_matrix.h5`, `*_cells.parquet.gz` only.

## Gene restriction (applies to the external tumours AND to a re-run of discovery)

Shared gene set G180 = the 280-gene base panel (GSE311609) intersected with the 5K panel gene list
(gene_panel.json of P17, descriptor "gene"): 180 genes. Counts are restricted to G180 before log-normalisation
and clustering, for every section in this plan. Consequences, fixed now:
- Lineage markers absent from G180 are skipped (KRT7, KRT14, ACTG2, C1QA, TRAC, TPSAB1); every lineage keeps
  at least one marker.
- Tumour-state split: of the three reference-state markers (CEACAM6, SERPINA3, TACSTD2) only SERPINA3 is in
  G180, so the median split of malignant clusters uses SERPINA3 alone. This is a known deviation and is reported.
- Discovery is re-run on Rep1 restricted to G180 (fixed Rep1 cluster map, `--section rep1`), so discovery and
  external runs use identical inputs. This "Rep1-G180" run is the primary comparator; the original Rep1
  rankings are a secondary comparator.
- Candidates: Rep1 U265 candidates in G180, minus each section's read-out.
- Clustering for external tumours: the GSE311609 rule (prepare_gse311609.py: PCA 20, k-means 20, seed 0,
  cells with >= 10 counts), applied to G180 counts.

## Runs

- Discovery Rep1-G180: both read-out rules, seeds 1, 2, 3 (6 runs; single model, 500 steps).
- External: 6 tumours x 2 read-out rules, seed 1 (12 runs; single model, 500 steps).
Everything else unchanged from the replication analysis (k = 10, 5 niches, model hyper-parameters).

Quality gate before looking at rankings (unchanged): >= 2,000 malignant-lineage cells and >= 500 cells in each
tumour state. Failing tumours are reported and excluded.

## Hypotheses and endpoints

Unit = tumour. Comparator = Rep1-G180 with the same read-out rule and seed 1.

H1 (primary): under the contamination-aware rule, the GNN ranking of an unseen tumour agrees with discovery more
than chance. Per tumour: top-25 overlap with hypergeometric p, Spearman on shared candidates. Across tumours:
one-sided sign test of (overlap > hypergeometric expectation). Supported if >= 5 of 6 tumours exceed
expectation (sign test p = 0.11 at 5/6, 0.016 at 6/6; reported as such).

H2: the contamination-aware rule is more reproducible than the default DE rule. Paired per-tumour difference in
top-25 overlap and Spearman; one-sided Wilcoxon signed-rank (n = 6, minimum attainable p = 0.016).

H3: the purity gate predicts reproducibility. Spearman of default-rule read-out purity with GNN overlap,
pooled over the 17 GSE311609 donors (already measured) and the 6 new tumours, and in the 6 new tumours alone
(descriptive, n = 6). Overlap compared between gated (purity >= 0.5) and flagged sections.

H4 (stability): top-25 Jaccard of the Rep1-G180 GNN ranking across seeds 1-3, for each rule, compared with the
between-tumour overlap. Reported descriptively.

Baselines (reported whatever they show, same endpoints): DE, DE (niche-stratified), LR-communication
(niche-stratified). We do not expect or claim that the GNN beats these; if DE is more reproducible, that is
reported as the result.

Evidence enrichment (secondary, same files as before): AUPRC vs prevalence for Open Targets breast,
clinical-stage targets and DepMap breast dependencies, restricted to G180 candidates; across tumours a
one-sided sign test of AUPRC > prevalence.

## Reporting rule

All runs and all endpoints are reported, including any that fail. Earlier results (pre-specified negative
replication; post hoc sensitivity) are not changed or re-labelled. Any analysis not listed here is labelled
exploratory.
