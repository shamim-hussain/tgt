#! /bin/bash

mkdir -p data/PCQM
cd data/PCQM

# This file contains the training, valid, test-dev and test-challenge split indices
wget -O splits.npz           https://huggingface.co/datasets/shamim-hussain/pcqm/resolve/main/splits.npz?download=true
# This file contains the molecular graph data including node and edge features extracted via RDKIT according to OGB specification
wget -O records.parquet      https://huggingface.co/datasets/shamim-hussain/pcqm/resolve/main/records.parquet?download=true
# This file contains the DFT coordinates extracted from the SDF file provided by OGB
wget -O dft_coords.parquet   https://huggingface.co/datasets/shamim-hussain/pcqm/resolve/main/dft_coords.parquet?download=true

# This file is only used for the models that use the RDKIT coordinates in the distance predictor
# This file contains the RDKIT coordinates calculated by forming RDKIT (30) conformations and subsequent MMFF optimization
wget -O rdkit_coords.parquet https://huggingface.co/datasets/shamim-hussain/pcqm/resolve/main/rdkit_coords.parquet?download=true
