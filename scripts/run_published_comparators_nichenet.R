# NicheNet ligand activity (nichenetr, saeyslab) on one Xenium section.
#
# Inputs written by run_published_comparators_xenium.py prepare:
#   <dir>/nichenet_inputs_pct_expressed.csv  (fraction of cells with >= 1 transcript, per lineage and in receivers)
# Prior model: NicheNet v2 (Zenodo 10.5281/zenodo.7074291): ligand_target_matrix_nsga2r_final.rds and
#   lr_network_human_21122021.rds (md5 b09606b04b2d4490418d9028c0e58b9f / 2a155f81e9ffffd5d5e709fe66bcc465).
#
# Pre-specified settings (standard NicheNet workflow, sender-agnostic):
#   receiver = invasive (target-state) malignant cells
#   expressed  = detected in >= 10% of the cells of a population (NicheNet default pct = 0.10)
#   background = genes expressed in receivers, restricted to the rows of the ligand-target matrix
#   gene set of interest = the section's read-out programme (15 genes most up-regulated in the target state;
#                          the same set the in-house methods use), restricted to the background
#   potential ligands = ligands of the NicheNet v2 LR network expressed in >= 1 lineage (any lineage, including
#                       malignant cells) whose receptor is expressed in receivers
#   ranking metric = aupr_corrected (NicheNet v2 default)
# Gene ranking rule (mirrors the in-house LR rule): a ligand receives its activity; a receptor receives the
#   maximum activity of the potential ligands it pairs with in the LR network; genes not scored receive 0
#   (scores are floored at 0 by taking max(0, .)). A ligands-only variant is also written.
#
#   Rscript scripts/run_published_comparators_nichenet.R <dir> <ligand_target_matrix.rds> <lr_network.rds>
suppressPackageStartupMessages({library(nichenetr); library(dplyr)})
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
ltm <- readRDS(args[2])
lr_network <- readRDS(args[3]) %>% distinct(from, to)
pct <- read.csv(file.path(dir, "nichenet_inputs_pct_expressed.csv"), check.names = FALSE)
lin_cols <- setdiff(grep("^pct_", colnames(pct), value = TRUE), "pct_receiver")
expressed_receiver <- pct$gene[pct$pct_receiver >= 0.10]
expressed_sender <- pct$gene[apply(as.matrix(pct[, lin_cols]), 1, max) >= 0.10]
background <- intersect(expressed_receiver, rownames(ltm))
geneset_oi <- intersect(pct$gene[pct$is_readout %in% c("True", "TRUE", TRUE)], background)
lr_exp <- lr_network %>% filter(from %in% expressed_sender, to %in% expressed_receiver, from %in% colnames(ltm))
potential_ligands <- unique(lr_exp$from)
cat("background", length(background), "geneset_oi", length(geneset_oi), paste(geneset_oi, collapse = ","),
    "potential_ligands", length(potential_ligands), "\n")
set.seed(1)
act <- predict_ligand_activities(geneset = geneset_oi, background_expressed_genes = background,
                                 ligand_target_matrix = ltm, potential_ligands = potential_ligands)
act <- act %>% arrange(desc(aupr_corrected))
write.csv(act, file.path(dir, "nichenet_ligand_activities.csv"), row.names = FALSE)
score <- setNames(rep(0, nrow(pct)), pct$gene)
lig_only <- score
for (i in seq_len(nrow(act))) {
  l <- act$test_ligand[i]; v <- act$aupr_corrected[i]
  if (l %in% names(score)) { score[l] <- max(score[l], v); lig_only[l] <- max(lig_only[l], v) }
  recs <- lr_exp$to[lr_exp$from == l]
  for (r in recs) if (r %in% names(score)) score[r] <- max(score[r], v)
}
out <- data.frame(gene = pct$gene, `NicheNet ligand activity (ligand + receptor)` = score,
                  `NicheNet ligand activity (ligands only)` = lig_only, check.names = FALSE)
write.csv(out, file.path(dir, "nichenet_gene_scores.csv"), row.names = FALSE)
writeLines(c(paste("nichenetr", as.character(packageVersion("nichenetr"))), R.version.string,
             paste("n_background", length(background)), paste("n_geneset_oi", length(geneset_oi)),
             paste("geneset_oi", paste(geneset_oi, collapse = ",")),
             paste("n_potential_ligands", length(potential_ligands)),
             paste("n_potential_ligands_on_panel", sum(potential_ligands %in% pct$gene)),
             paste("n_lr_pairs_expressed", nrow(lr_exp))),
           file.path(dir, "nichenet_run_info.txt"))
