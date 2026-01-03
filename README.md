# Representation-to-Creativity

📑 [**Representation-to-Creativity (R2C): Automated Holistic Scoring Model for Essay Creativity**](https://aclanthology.org/2025.findings-naacl.292.pdf)

## Introduction

we propose a novel **self-supervised learning model** that recognizes cluster patterns within the essay embedding space and leverages them for creativity scoring. This approach aims to automatically generate a **high-quality training set**, thereby facilitating the training of diverse language models.

<img src="./assets/Overview_of_the_proposed_method.png" width="700" style="display: block; margin: 0 auto;"/>

# 🛠️ Installation
```bash
git https://github.com/Deok-gi/Representation-to-Creativity.git
cd Representation-to-Creativity

conda create -n r2c python=3.7.16
conda activate r2c

pip install -r requirements.txt
```

# 💻 Usage
### 1. Essay-to-Vector (E2V)
```bash
cd src
python3 01. essay-to-vector.py
```
- The D1, D2, and D3 datasets presented in the paper are provided in the "datasets" folder.
- The model used in the code is as follows: https://huggingface.co/Deok-gi/longformer-bert-base-4096 (Recommend downloading the model locally.)
- When you run the code, the essay vectors are transformed into 2D vectors through PCA and saved as a .pt file.

### 2. Clustering of Essay Vectors
```bash
02-1. aic-bic_view.ipynb
02-2. clustering-essay-vectors_view.ipynb
```
- "aic-bic_view.ipynb" can be used to estimate the optimal number of clusters.
- "clustering-essay-vectors_view.ipynb" can be used to visualize the EM clustering results. For each cluster, the number of e2v instances, mean creativity score, and variance are displayed.
- We consider the cluster with the largest number of e2v elements to be the conventional cluster.

### 3. Training Transformer Using Essay Data
```bash
python3 03. transformer-train.py
```
- Tokenizes Korean text using Okt morphological analyzer
- Builds a vocabulary dictionary from all tokens
- Trains for 10 epochs using CrossEntropyLoss and Adam optimizer
- Saves trained weights to `transformer_weights.pth`

### 4. Automatic Labeling of Conventional, Creative, and Non-creative Clusters
```bash
python3 04. creativity-aware-distance_calc.py
```