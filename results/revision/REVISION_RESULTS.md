# Revision results: contamination-aware read-out, purity QC gate, external validation

Analysis exactly as locked in `PRESPECIFIED_REVISION.md` (commit dd3ed9f, pushed before any run).
Run date 5-6 Nov 2026. All 18 runs finished: Rep1-G180 discovery (2 rules x seeds 1-3) and the six unseen
GSE350571 tumours (2 rules, seed 1). All six tumours passed the pre-specified quality gate
(>= 2,000 malignant cells, >= 500 cells per state). Numbers: `summary/revision_endpoints.txt`,
`summary/external_per_tumour.csv`, `summary/{rule}/replication_*.csv`.

Comparator: Rep1-G180 with the same read-out rule, seed 1. Top-25 overlap expected by chance about 4/25
(155 shared candidates on average).


## CORRECTION (7 Nov 2026): the headline below was wrong

While preparing the manuscript we found two artefacts in the top-25 overlap used for H1, H2 and the baselines.
Audit: `scripts/audit_ties.py`, tables in `tie_audit/`.

1. **Masked DE baseline.** With `--readout-rule malignant_specific`, `run_xenium.py` stored the read-out selection
   vector, in which every gene that is not malignant-specific was set to minus infinity, as the "DE" baseline. The
   DE baseline therefore ranked only the malignant-specific genes (about 15-20 of 165 candidates here), and the
   rest of its top 25 was filled in alphabetical order, the same in every tumour. The reported improvement of DE
   from 4.5 to 18.5 of 25 is an artefact. DE does not depend on the read-out; recomputed without the mask it
   shares a median of 4.5 of 25 with discovery under both rules (chance about 4). The code is fixed (the DE
   baseline is never masked).
2. **Ties.** `top()` broke ties by input (alphabetical) order. With 180 genes, LR-communication gives the same
   zero score to most candidates, and the GNN gives zero to some, so tied genes entered the top 25 in the same
   order everywhere. With ties broken at random (1,000 draws), LR-communication drops from 23 to 18.4 (default)
   and from 22 to 19.3 (contamination-aware), and the GNN from 5 to 4.0.

Corrected results (median over six tumours, tie-aware top-25 overlap; chance about 4.0):

| Method | Default read-out | Contamination-aware read-out |
|---|---|---|
| DE (unmasked) | 4.5 | 4.5 |
| DE (niche-stratified) | 7.0 | 6.5 |
| NicheTarget-GNN | 3.7 | 4.0 (above chance in 3/6) |
| Neighbourhood correlation (niche) | 5.0 | 4.5 |
| LR-communication (niche) | 18.4 | 19.3 |

- **H1 not supported.** Under the contamination-aware read-out the GNN overlap equals chance (above chance in
  3 of 6 tumours).
- **H2 not supported.** GNN paired change median +1, Wilcoxon one-sided p = 0.063; no baseline improved.
- H3 and H4 are unaffected (H4 stands: GNN seed overlap 19-21 of 25 in discovery).
- The 17-donor analyses were re-audited with the same script: no tie problem (every method has at least 25
  distinct scores), so the pre-specified result (GNN 1 of 25) and the post hoc GNN gain to 7 of 25 stand. The
  post hoc "DE 11 of 25" was the same masked baseline (it ranked only about 55 malignant-specific genes);
  unmasked DE shares 4 of 25. Any text saying the malignant-specific read-out made DE more reproducible is wrong.

Conclusion after correction: the post hoc GNN gain seen in the 17 donors (1 to 7 of 25) did not replicate in six
unseen tumours on a different panel restricted to 180 genes. The contamination-aware read-out is not shown to
improve cross-patient reproducibility of any method. The only reproducible ranking is LR-communication, whose
scores are driven by the curated ligand-receptor list.

---

# Original write-up of 6 Nov 2026 (SUPERSEDED by the correction above; kept unchanged for the record)

## Headline

1. **The contamination-aware read-out made spatial target ranking reproducible across patients.** With the
   default DE read-out, the DE target ranking of an unseen tumour shared a median 4.5/25 targets with
   discovery (chance about 4; above chance in 3/6 tumours; Spearman 0.08). With the contamination-aware
   read-out it shared 18.5/25 (above chance 6/6, sign p = 0.016; Spearman 0.49, positive 6/6). Paired
   improvement 6/6 tumours, median +13.5 targets, Wilcoxon one-sided p = 0.016 (minimum attainable at n = 6).
2. **The GNN ranking did not become reproducible.** Under the contamination-aware rule the GNN overlap was a
   median 5/25 (chance 4). It exceeded chance in 5/6 tumours, which meets the pre-specified count for H1, but
   the sign test is p = 0.11, the excess is about one gene, and rank agreement is absent (Spearman median
   -0.01, positive in 3/6). We report H1 as formally met on the count rule but not supported in substance.
3. The purity gate did not separate the new tumours (H3): five of six were above 0.5 and purity did not
   track GNN overlap within them (rho -0.03, n = 6). The pooled correlation (rho 0.70, p = 0.0001, n = 25) is
   driven by the earlier 19 GSE311609 sections.
4. Seed noise does not explain the GNN result (H4): within Rep1-G180 the GNN top-25 overlapped 19-21/25 across
   seeds (Jaccard 0.61-0.72), far above its 3.5-5/25 between-tumour overlap.

## H1: GNN, contamination-aware rule (primary)

| Tumour | Top-25 overlap | Spearman |
|---|---|---|
| P17 | 5 | -0.07 |
| P22 | 4 | -0.09 |
| P24 | 9 | 0.02 |
| P26 | 5 | 0.01 |
| P29 | 4 | 0.02 |
| P32 | 5 | -0.02 |

Above expectation (overlap > 25 x 25 / n_common, about 4.0): 5/6, sign p = 0.109. Pre-specified rule
"supported if >= 5 of 6" is met; the effect size is about one gene over chance and rank correlation is null.

## Reproducibility of every method (median top-25 overlap / median Spearman, 6 tumours)

| Method | Default rule | Contamination-aware rule | Paired change (overlap) |
|---|---|---|---|
| DE | 4.5 / 0.08 (3/6 above chance) | **18.5 / 0.49 (6/6, p = 0.016)** | +13.5, 6/6 improved, p = 0.016 |
| DE (niche-stratified) | 7.0 / 0.10 (5/6) | 6.5 / 0.17 (6/6, p = 0.016) | +0.5, 3/6, p = 0.25 |
| NicheTarget-GNN | 3.5 / -0.04 (1/6) | 5.0 / -0.01 (5/6, p = 0.11) | +1.5, 5/6, p = 0.031 |
| Neighbour-correlation (niche) | 5.0 / -0.04 (3/6) | 4.5 / 0.02 (3/6) | 0, 3/6, p = 0.64 |
| LR-communication (niche) | 23.0 / 0.94 (6/6) | 22.0 / 0.88 (6/6) | +0.5, 3/6, p = 0.72 |

LR-communication is reproducible under both rules because its candidate scores come mostly from the curated
ligand-receptor list rather than from the tumour (same caveat as the 17-donor analysis).

## H2 (contamination-aware vs default rule)

GNN top-25 overlap improved in 5/6 tumours (median +1.5, Wilcoxon p = 0.031); GNN Spearman did not
(3/6, p = 0.50). DE improved in 6/6 on both endpoints (p = 0.016 each).

## H3 (purity gate)

Default-rule read-out purity in the new tumours: P17 0.73, P22 0.53, P24 0.60, P26 0.67, P29 0.33 (flagged),
P32 0.73. Spearman(purity, GNN overlap) = -0.03 (n = 6). Pooled with the 19 GSE311609 sections that passed
the gate: rho = 0.70, p = 0.0001 (n = 25). Gated tumours median GNN overlap 3; the one flagged tumour 4.
Under the contamination-aware rule all six read-outs had purity 1.0, by construction.

## H4 (seed stability, Rep1-G180)

Contamination-aware: top-25 overlap 20, 21, 21 (Jaccard 0.67, 0.72, 0.72). Default: 19, 20, 21
(Jaccard 0.61, 0.67, 0.72).

## Evidence enrichment (secondary; tumours with AUPRC above prevalence, sign p)

Contamination-aware rule: DE ranked DepMap breast dependencies above prevalence in 6/6 (p = 0.016; median
AUPRC 0.096 vs prevalence 0.016) but Open Targets breast in 0/6; GNN Open Targets breast 5/6 (p = 0.11),
DepMap 4/6; LR-communication Open Targets breast and clinical 6/6 (p = 0.016), DepMap 0/6;
neighbour-correlation DepMap 6/6 (p = 0.016). Default rule: DE DepMap 1/6; GNN Open Targets breast 5/6,
DepMap 3/6. Full table in `summary/{rule}/replication_across_donors.csv`.

## Deviations and caveats

- Analysis restricted to 180 genes shared by both panels; tumour-state split uses SERPINA3 only (pre-specified).
- Large tumours scanned in 2,000-cell chunks (exact; implementation note in the plan); models trained before an
  out-of-memory stop were reused, not retrained.
- Six tumours give limited power; the minimum attainable one-sided p is 0.016.
- Receptor status of the GSE350571 tumours was not used. ERBB2 mean expression in malignant cells was higher
  in P17, P22 and P24 than in P26, P29 and P32 (QC table), which we report without assigning subtypes.
- The 17-donor pre-specified negative result and the post hoc sensitivity analysis are unchanged.

## What this means for the paper

The reproducible, pre-registered finding is about the read-out, not the GNN: removing stromal and immune
spillover from the read-out programme turns a non-reproducible target ranking (4.5/25) into a reproducible one
(18.5/25) in six tumours never used before, from a different platform (Xenium Prime 5K). The GNN does not share
this gain and must not be presented as better than baselines.
