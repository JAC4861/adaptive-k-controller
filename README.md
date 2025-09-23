# adaptive-k-controller
A theory-grounded Adaptive-K Controller to prevent geometric collapse and stabilize Hyperbolic GNNs.

# When Task Performance Deceives: Task-Geometry Decoupling in Learnable-Curvature Hyperbolic GNNs

This repository contains the official PyTorch implementation for the paper **"When Task Performance Deceives: Decoupling Task and Geometric Fidelity in Hyperbolic GNNs"**.

## 1. Overview

This repository provides the code to study the "task-geometry decoupling" problem and contains an implementation of our proposed **Adaptive-K Controller** applied to Hyperbolic Graph Convolutional Networks (HGCN) [1].

The library allows for training and evaluation of the following methods for the **link prediction (`lp`)** task:

#### Graph Neural Network (GNN) methods

* Graph Convolutional Networks (`GCN`) 
* Graph Attention Networks (`GAT`) 
* **Hyperbolic GNNs (based on HGCN)**
    * `Fixed-K (Oracle)`: HGCN with a fixed, optimal curvature.
    * `Free-K`: Standard HGCN with unconstrained learnable curvature.
    * `Adaptive-K (Ours)`: HGCN augmented with our proposed controller.

## 2. Setup

### 2.1 Installation

We recommend using Conda for environment management.

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repo-Name].git
    cd [Your-Repo-Name]
    ```
2.  **Create and activate the conda environment:**
    ```bash
    conda create -n hgcn_controller python=3.9
    conda activate hgcn_controller
    ```
3.  **Install dependencies:**
    Please first install the correct PyTorch version for your CUDA setup from the [official website](https://pytorch.org/get-started/locally/). Then, install all other requirements.
    ```bash
    # Example for CUDA 12.8
    # pip install torch==2.7.1+cu128 ...

    # Install all other requirements from the provided file
    pip install -r requirements.txt
    ```

### 2.2 Datasets

The `data/` folder should contain the source files for:
* Cora
* PubMed
* Airport
* Disease

The code will automatically process these datasets. Pre-computed node pairs for distortion evaluation (e.g., `cora_pairs.npy`) should also be placed in this folder.

## 3. Usage

The main script to run all experiments is `adaptive_train.py` (which is called by the scripts in Section 4). Below is a comprehensive list of its command-line arguments.

#### Training Configuration
* `--lr`: Learning rate (default: `0.01`).
* `--dropout`: Dropout probability (default: `0.0`).
* `--cuda`: CUDA device ID to use; `-1` for CPU (default: `0`).
* `--epochs`: Maximum number of epochs (default: `5000`).
* `--weight-decay`: L2 regularization strength (default: `0.0`).
* `--optimizer`: Optimizer to use, `Adam` or `RiemannianAdam` (default: `Adam`).
* `--patience`: Patience for early stopping (default: `100`).
* `--seed`: Random seed for training (default: `1234`).
* `--save`: Set to `1` to save model and logs (default: `0`).

#### Model & Data Configuration
* `--model`: Encoder model to use, e.g., `HGCN`, `GCN` (default: `HGCN`).
* `--dataset`: Dataset name, e.g., `cora`, `pubmed` (default: `cora`).
* `--dim`: Embedding dimension (default: `16`).
* `--manifold`: Manifold to use, `PoincareBall` or `Euclidean` (default: `PoincareBall`).
* `--c`: Hyperbolic curvature; set to `None` for trainable curvature (default: `None`).
* `--num-layers`: Number of hidden layers (default: `2`).
* `--act`: Activation function, e.g., `relu` (default: `relu`).
* `--bias`: Whether to use a bias term (`1` for yes, `0` for no) (default: `1`).

#### Adaptive Controller Configuration (Ours)
* `--lambda-reg`, `--mu-reg`: Penalty strengths for the high and low curvature boundaries.
* `--plateau-low`, `--plateau-high`: The lower and upper bounds of the target stable curvature band.
* `--plateau-mean`: The target mean distortion for the stable plateau.
* `--plateau-window`: Size of the sliding window to check for stability.
* `--plateau-var-th`: Variance threshold for distortion to trigger a freeze.
* `--curv-warmup`: Number of initial epochs to skip before the controller becomes active.
* `--curv-freeze`: Set to `1` to enable the curvature freezing mechanism.

## 4. Examples

We provide the exact commands used to generate the results in our paper. To reproduce the results, run these commands for all seeds and average the metrics from the resulting log files.

### 4.1 Cora Dataset

We provide the exact commands used to generate the results in our paper. To reproduce the results, run these commands for all seeds and average the metrics from the resulting log files.

### 4.1 Cora Dataset

* **Fixed-K (Oracle)**
    ```bash
    python -u adaptive_train.py --task lp --dataset cora --model HGCN --manifold PoincareBall --dim 16 --dropout 0.5 --weight-decay 0.001 --epochs 600 --patience 120  --lr 0.01 --c 0.77 --curv_lr_scale 0 --lambda-reg 0 --mu-reg 0 --save 1 --distortion-use-precomputed 1 --distortion-pairs-file data/cora_pairs.npy --distortion-pairs-limit 8000
    ```
* **Free-K**
    ```bash
    python -u adaptive_train.py --task lp --dataset cora --model HGCN --manifold PoincareBall --dim 16 --dropout 0.5 --weight-decay 0.001 --epochs 600 --patience 120  --lr 0.01 --c None --curv_lr_scale 1.0 --lambda-reg 0 --mu-reg 0 --save 1 --distortion-use-precomputed 1 --distortion-pairs-file data/cora_pairs.npy --distortion-pairs-limit 8000
    ```
* **Adaptive-K (Ours)**
    ```bash
    python -u adaptive_train.py --task lp --dataset cora --model HGCN --manifold PoincareBall --dim 16 --dropout 0.5 --weight-decay 0.001 --epochs 600 --patience 120  --lr 0.01 --c None --curv_lr_scale 1.0 --lambda-reg 1.5 --mu-reg 1.1925 --plateau-low 0.77 --plateau-high 0.82 --plateau-mean 0.505 --plateau-window 8 --plateau-var-th 0.01 --plateau-min-epochs 50 --curv-warmup 15 --curv-freeze 1 --save 1 --distortion-use-precomputed 1 --distortion-pairs-file data/cora_pairs.npy --distortion-pairs-limit 8000
    ```


### 4.2 Airport Dataset

* **Fixed-K (Oracle)**
    ```bash
    python -u adaptive_train.py --task lp --dataset airport --model HGCN --manifold PoincareBall --dim 16 --dropout 0.0 --weight-decay 0 --epochs 1500 --patience 200  --lr 0.01 --c 0.90 --curv_lr_scale 0 --lambda-reg 0 --mu-reg 0 --save 1 --distortion-use-precomputed 1 --distortion-pairs-file data/airport_pairs.npy --distortion-pairs-limit 4000
    ```
* **Free-K**
    ```bash
    python -u adaptive_train.py --task lp --dataset airport --model HGCN --manifold PoincareBall --dim 16 --dropout 0.0 --weight-decay 0 --epochs 1500 --patience 200  --lr 0.01 --c None --curv_lr_scale 1.0 --lambda-reg 0 --mu-reg 0 --save 1 --distortion-use-precomputed 1 --distortion-pairs-file data/airport_pairs.npy --distortion-pairs-limit 4000
    ```
* **Adaptive-K (Ours)**
    ```bash
    python -u adaptive_train.py --task lp --dataset airport --model HGCN --manifold PoincareBall --dim 16 --dropout 0.0 --weight-decay 0 --epochs 1500 --patience 200  --lr 0.01 --c None --curv_lr_scale 1.0 --lambda-reg 1.5 --mu-reg 1.36 --plateau-low 0.88 --plateau-high 0.92 --plateau-mean 0.261 --plateau-window 12 --plateau-var-th 0.01 --plateau-min-epochs 60 --curv-warmup 20 --curv-freeze 1 --adapt-fail-epoch 120 --adapt-lambda-scale 1.5 --save 1 --distortion-use-precomputed 1 --distortion-pairs-file data/airport_pairs.npy --distortion-pairs-limit 4000
    ```


## 5. Citation

If you find our work useful in your research, please consider citing our paper:
```bibtex
@misc{chen2025decoupling,
  title={When Task Performance Deceives: Decoupling Task and Geometric Fidelity in Hyperbolic GNNs},
  author={Chen, Lixian and Wang, Jingchao and Dai, Zhaorong and Liu, Hanqian and Ai, Danxiang and Yang, Shi},
  year={2025},
  howpublished={Technical Report},
  note={Preprint}
}
