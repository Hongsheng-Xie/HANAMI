# HANAMI: Heterogeneous Graph Contrastive Learning for Drug-Gene-Disease Motif Prediction

## Overview
This project implements a drug–gene–disease motif prediction model using a multi-view deep graph learning framework with GraphSAGE convolution. The model predicts all seven drug-gene-disease motifs by integrating heterogeneous biomedical data and leveraging relation-aware topology encoding and contrastive learning to enable accurate and biologically meaningful predictions. It employs an architecture combining:

- Message Passing GraphSAGE Convolutional Networks
- Structure-Aware Pooling module
- Attention-based fusion mechanism
- N-pair Contrastive Learning strategy

---
![Main Model Architecture](image/main.png)
---

## 🔗 Pretrained Resources Used for Feature Initialisation

| Resource | Purpose in DREAM-GNN | Link |
| -------- | ------------------- | ---- |
| **ChemBERTa (ZINC100M, MLM & v1 base-zinc)** | 1152-dim SMILES embeddings for small-molecule drugs | [`DeepChem/ChemBERTa-100M-MLM`](https://huggingface.co/DeepChem/ChemBERTa-100M-MLM)&[`seyonec/ChemBERTa-zinc-base-v1`](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)|
| **MPNN** | 300-dim SMILES embeddings for small-molecule drugs | [`MPNN Class`](https://github.com/chemprop/chemprop/blob/main/chemprop/models/model.py)|
| **BioBERT (v1.1 large-cased-squad)** | 1024-dim biomedical text embeddings for disease terms | [`dmis-lab/biobert-large-cased-v1.1-squad`](https://huggingface.co/dmis-lab/biobert-large-cased-v1.1-squad)|
| **ClinicalBERT (base-cased-clinical)** | 768-dim biomedical text embeddings for disease terms | [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)|
| **Borzoi (replicate-0)** | 1536-dim genomic sequence embeddings for genes | [`johahi/borzoi-replicate-0`](https://huggingface.co/johahi/borzoi-replicate-0)|
| **Enformer (official-rough)** | 1536-dim genomic sequence embeddings for genes | [`EleutherAI/enformer-official-rough`](https://huggingface.co/EleutherAI/enformer-official-rough)|
| **DrugBank** | Curated drug metadata & identifiers | [DrugBank Online](https://go.drugbank.com/)|
| **MeSH** | Curated disease phenotype information | [MeSH.org](https://www.ncbi.nlm.nih.gov/mesh/)|
| **NCBI** | Curated genetic sequence information | [NCBI.gov](https://www.ncbi.nlm.nih.gov/gene/)|

---


## Files Description

- `Attention.py`: Implements a PyTorch-based attention mechanism
- `base_gcn.py`: Defines the neural network architectures and custom layers, including GraphSAGE layers, Structure-Aware Poolings, and Multi-Layer Perceptrons (MLPs)
- `create_data.py`: Manages the logic of assembling valid drug-gene-disease motifs and generating corresponding negative samples
- `embedding.py`: Leverages domain-specific pre-trained models to extract and process the initial high-dimensional feature representations for drugs, genes, and diseases.
- `main_tri_binary.py`: Main training script with contrastive learning, seed-based experiments, and model evaluations (AUROC, AUPR)
- `utils.py`: Utility functions for graph processing and logging

## Usage

### Basic Training

Run training with default parameters:

```bash
python train.py --data_name lrssl --device 0
```

## Model Architecture

1. **GCMC Module**: Processes drug-disease interaction graph with relation-specific transformations
2. **FGCN Module**: Processes drug and disease similarity graphs separately
3. **Attention Fusion**: Combines topology and feature representations
4. **MLP Decoder**: Predicts association scores

## Data Format

Input data should be in MATLAB (.mat) format containing:
- `didr`: Drug-disease association matrix
- `drug`: Drug similarity matrix
- `disease`: Disease similarity matrix
- `drug_embed`: Drug feature embeddings
- `disease_embed`: Disease feature embeddings
- `Wrname`: Drug identifiers

## Cold Start

### Overview
The cold start module handles completely unseen drugs and diseases using a retrieval-aggregation framework with disease-conditional attention. It uses raw feature similarity for neighbor selection and trained embeddings for context-aware aggregation to predict associations for entities not seen during training.

### Usage
First, train models and save embeddings:
```bash
python train.py --data_name Gdataset --device 0 --save_model
```

Then run cold start evaluation:
```bash
python cold_start.py
```
