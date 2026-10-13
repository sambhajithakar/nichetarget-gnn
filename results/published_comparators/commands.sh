#!/usr/bin/env bash
# Exact commands used for the published-comparator analyses (CPU only, 4 cores, 15 GB RAM).
# Environments were created with micromamba 2.9.0 from conda-forge/bioconda (PyPI/CRAN/GitHub codeload unavailable):
#   micromamba create -n cmp -c conda-forge -c bioconda python=3.11 liana scanpy pot anndata pandas scikit-learn networkx libpysal
#   micromamba install -n cmp -c conda-forge -c bioconda python-igraph leidenalg plotly seaborn karateclub python-louvain
#   git clone --depth 1 https://github.com/zcang/commot && pip install --no-deps --no-build-isolation ./commot      (env cmp)
#   micromamba create -n nn -c conda-forge -c bioconda r-base=4.4 r-tidyverse r-data.table r-igraph r-matrix r-fdrtool r-rocr \
#     r-catools r-hmisc r-caret r-randomforest r-diagrammer r-mlrmbo r-parallelmap r-emoa r-dicekriging r-e1071 r-seurat r-cowplot \
#     r-ggpubr r-circlize bioconductor-complexheatmap r-ggnewscale r-ggforce r-shadowtext
#   git clone --depth 1 https://github.com/saeyslab/nichenetr && R CMD INSTALL --no-docs ./nichenetr                (env nn)
#   curl -L -o ligand_target_matrix_nsga2r_final.rds https://zenodo.org/api/records/7074291/files/ligand_target_matrix_nsga2r_final.rds/content
#   curl -L -o lr_network_human_21122021.rds https://zenodo.org/api/records/7074291/files/lr_network_human_21122021.rds/content
#   micromamba create -n cel -c conda-forge python=3.11 "pytorch=*=cpu*" pytorch_geometric scikit-learn scanpy pandas tqdm flit-core
#   git clone --depth 1 https://github.com/Teichlab/celcomen && pip install --no-deps --no-build-isolation ./celcomen  (env cel)
set -euo pipefail
D=/home/user/data
NN=/home/user/tools/nichenet_v2
# 1. simulation (seeds 1-5, complete and L1-R1-missing databases), env cmp
python scripts/run_published_comparators_simulation.py --seeds 1 2 3 4 5 --out results/published_comparators/simulation
# 2. Xenium, env cmp (COMMOT + LIANA+) then env nn (NicheNet), then env cmp (evaluation)
python scripts/run_published_comparators_xenium.py prepare --data $D/rep1.pkl --section rep1 --run results/xenium_rep1 \
    --lr $D/lr/lr_pairs_human.csv --out results/published_comparators/xenium_rep1
python scripts/run_published_comparators_xenium.py prepare --data $D/rep2.pkl --section other --run results/xenium_rep2 \
    --lr $D/lr/lr_pairs_human.csv --out results/published_comparators/xenium_rep2
for s in rep1 rep2; do
  Rscript scripts/run_published_comparators_nichenet.R results/published_comparators/xenium_$s \
      $NN/ligand_target_matrix_nsga2r_final.rds $NN/lr_network_human_21122021.rds
done
python scripts/run_published_comparators_xenium.py evaluate --ot-assoc $D/opentargets/breast_carcinoma_associations.csv \
    --ot-drugs $D/opentargets/panel_druggability.csv --depmap $D/depmap/panel_breast_gene_effect.csv \
    --out results/published_comparators/xenium_evaluation
# 3. Perturb-map, env cel
P="--npz $D/perturbmap/npz --raw $D/perturbmap/raw --lr $D/lr/lr_pairs_mouse.csv --out results/published_comparators/perturbmap_celcomen --seeds 1"
python scripts/run_published_comparators_perturbmap_celcomen.py $P --check                        # synthetic equivalence check
# Celcomen seed 1 was trained (published train(), 200 epochs, 457 s) in a timing run; its 6 candidate effects were computed
# with an earlier closed-form calc_sphex that was NOT bit-identical to the published one and were deleted before the scan below.
python scripts/run_published_comparators_perturbmap_celcomen.py $P --threads 2 --kos Tgfbr2 --max-candidates 6
rm results/published_comparators/perturbmap_celcomen/effects_seed1_*.pkl
python scripts/run_published_comparators_perturbmap_celcomen.py $P --threads 2 --kos Tgfbr2 --check-real   # real-data check
python scripts/run_published_comparators_perturbmap_celcomen.py $P --threads 2 --kos Tgfbr2 &   # the two scans ran in parallel
python scripts/run_published_comparators_perturbmap_celcomen.py $P --threads 2 --kos Ifngr2 &
wait
python scripts/run_published_comparators_perturbmap_celcomen.py $P --evaluate-only
# 4. tables
python scripts/summarise_published_comparators.py --root results/published_comparators
