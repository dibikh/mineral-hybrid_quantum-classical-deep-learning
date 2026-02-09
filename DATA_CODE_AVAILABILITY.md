## Data & Code Availability

**Code.** The full source code for data preprocessing, model training/inference (BiLSTM, QBiLSTM, Hybrid), and scripts to reproduce Table 1 and Figs. 2–11 is available in this public repository.

**Data.** Raw mineral price data are not redistributed in this repository due to licensing/terms of use. Users can obtain the raw data from the original provider and place the CSV file in `data/raw/` as described in `data/README.md`. The provided preprocessing script (`data/preprocess.py`) generates the processed dataset used in the experiments.

**Reproducibility.** The repository includes a pinned environment specification (`environment.yml`) and fixed random seeds (default seed = 42). Running `scripts/run_all.sh` reproduces Table 1 and the figure set used in the paper.

**DOI.** A DOI-backed archive of this repository is provided via Zenodo (link/DOI to be inserted after release).
