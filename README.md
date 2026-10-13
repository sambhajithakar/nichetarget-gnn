# NicheTarget-GNN

Code and data outputs for the manuscript *Niche-Aware Heterogeneous Graph Neural Networks with Counterfactual Perturbation for Discovering Cell-State-Specific Druggable Targets in the Tumor Microenvironment* (Thakar and Ghorpade, submitted to Computers in Biology and Medicine). License: MIT. Please cite the archived release (see `CITATION.cff`).

Niche-aware graph neural network with counterfactual target inhibition for ranking cell-state-specific, selective drug targets from spatial transcriptomics.

## Layout
| Path | What it is |
|---|---|
| `nichetarget/simulate.py` | Simulator with planted paracrine (L1→R1, L2→R2), intrinsic (D1), non-selective (E1), passenger and decoy genes |
| `nichetarget/graph.py` | k-NN cell graph, LR edge features, niche composition |
| `nichetarget/model.py` | Edge-aware graph attention + FiLM niche conditioning; masked-expression training |
| `nichetarget/counterfactual.py` | In-silico inhibition, niche-resolved efficacy, selectivity, final score |
| `nichetarget/baselines.py` | DE, neighbourhood correlation, LR-communication (global and niche-stratified) |
| `scripts/run_simulation.py` | Simulation benchmark (`--hide-lr L1 R1` for the incomplete-database scenario) |
| `scripts/summarise_simulation.py` | Tables and Figs 3–5 |
| `scripts/make_figures_static.py` | Figs 1–2 |
| `results/simulation/` | Raw outputs (JSON per scenario, per-gene rankings per seed, logs) |
| `results/simulation_v2/` | Ensemble and hybrid variants on seeds 6–10 |

## Reproduce
```bash
pip install -r requirements.txt
python scripts/run_simulation.py --seeds 1 2 3 4 5 --out results/simulation
python scripts/run_simulation.py --seeds 1 2 3 4 5 --main-only --hide-lr L1 R1 --out results/simulation
python scripts/summarise_simulation.py results/simulation ../manuscript/figures
# follow-up variants (3-model ensemble, hybrid with LR-communication) on fresh seeds
python scripts/run_simulation.py --seeds 6 7 8 9 10 --main-only --n-ensemble 3 --hide-lr L1 R1 --out results/simulation_v2
python scripts/run_simulation.py --seeds 6 7 8 9 10 --main-only --n-ensemble 3 --out results/simulation_v2
```
Seed 0 was used for development only. CPU is enough (about 10 min per seed on 4 cores).

## Real-data analyses (NumPy implementation)
PyTorch could not be installed in the environment used for the real-data runs, so `nichetarget/npgrad.py`, `np_model.py`, `np_graph.py`, `np_counterfactual.py` and `np_baselines.py` re-implement the same model, optimiser and counterfactual scan in NumPy only (gradients checked in `tests/test_npgrad.py`; parity on simulation in `results/parity_numpy/`). `nichetarget/h5lite.py` reads 10x `.h5` files without h5py (`tests/test_h5lite.py`).

| Path | What it is |
|---|---|
| `scripts/prepare_data.py` | Xenium outputs -> `rep*.pkl`; Perturb-map `.h5` -> `npz/`; DepMap breast summary |
| `scripts/run_xenium.py` | GNN ensemble, hybrid and baselines on a Xenium section (pre-specified design in the docstring) |
| `scripts/evaluate_xenium.py` | Open Targets / clinical-stage drug / DepMap enrichment and Rep1-vs-Rep2 concordance |
| `scripts/run_perturbmap.py` | Perturb-map in-silico knockout validation (pre-specified protocol in the docstring) |
| `scripts/make_fig_xenium.py`, `make_fig_perturbmap.py`, `make_tables_xenium.py` | Fig. 6, Fig. 7, Table 3 / S4 |
| `results/xenium_rep1/`, `results/xenium_rep2/` | Per-gene scores (`gene_scores.csv`), run info, logs, trained parameters |
| `results/xenium_evaluation/` | Evidence evaluation and replicate concordance |
| `results/perturbmap/` | `perturbmap_validation.csv`, per-read-out effects, logs |

Data: 10x Genomics "Xenium_FFPE_Human_Breast_Cancer_Rep1/Rep2"; GEO GSE193460 (Perturb-map); Open Targets Platform 26.09 (`fetch_ot.py`); DepMap 25Q2 (Zenodo 17161166); NicheNet v1/v2 and CellPhoneDB v2.1.7 ligand–receptor pairs (`build_lr.py`). Copies of the processed inputs are in `../data/`.

```bash
export OPENBLAS_NUM_THREADS=1   # run ensemble members as parallel processes instead
python scripts/prepare_data.py xenium --root xenium_rep1/x --cells xenium_rep1/rep1_cells.csv.gz --out rep1.pkl
python scripts/run_xenium.py --data rep1.pkl --section rep1 --lr lr_pairs_human.csv --out results/xenium_rep1 --n-ensemble 3 --steps 500
python scripts/run_xenium.py --data rep2.pkl --section other --lr lr_pairs_human.csv --out results/xenium_rep2 --n-ensemble 3 --steps 500
python scripts/evaluate_xenium.py --runs results/xenium_rep1 results/xenium_rep2 --ot-assoc breast_carcinoma_associations.csv \
    --ot-drugs panel_druggability.csv --depmap panel_breast_gene_effect.csv --out results/xenium_evaluation
python scripts/prepare_data.py perturbmap --raw perturbmap/raw --out perturbmap/npz
python scripts/run_perturbmap.py --npz perturbmap/npz --raw perturbmap/raw --lr lr_pairs_mouse.csv --out results/perturbmap \
    --seeds 1 2 3 --variants full no_spatial
```
On a 4-core CPU one Xenium ensemble member takes about 15–20 min to train and 25 min to scan 298 genes (use `--members 0 1 2 --no-combine` in separate processes, then run once more with `--members -1` to load them and combine); one Perturb-map model takes 15 min (no spatial) to 85 min (full).
