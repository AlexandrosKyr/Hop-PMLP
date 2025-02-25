# Hop-PMLP: Single-Layer HopGNN-Enhanced PMLP for Sparse Heterophilic Graphs

![GitHub license](https://img.shields.io/github/license/AlexandrosKyr/Hop-PMLP)
![GitHub stars](https://img.shields.io/github/stars/AlexandrosKyr/Hop-PMLP?style=social)
![GitHub forks](https://img.shields.io/github/forks/AlexandrosKyr/Hop-PMLP?style=social)

## Overview
Hop-PMLP is a hybrid deep learning model that **combines Propagational Multi-Layer Perceptron (PMLP) with Hop Graph Neural Networks (HopGNN)** to improve performance on **heterophilic and sparse graphs**.

### Key Features:
- **Single HopGNN layer** in training to capture graph structure.
- **Full HopGNN message-passing** in inference for accuracy boost.
- **Works well on heterophilic graphs** (Wisconsin, Texas, Cornell).

---

## Table of Contents
- [Overview](#-overview)
- [Motivation](#-motivation)
- [Based on Previous Work](#-based-on-previous-work)
- [Code and Implementation](#️-code-and-implementation)
- [Installation](#-installation)
- [Usage](#-usage)
- [Datasets](#-datasets)
- [Results](#-results)
- [Limitations & Future Work](#️-limitations--future-work)
---

## Motivation
Traditional **Graph Neural Networks (GNNs)** struggle with:
- **Over-smoothing** in deep architectures.
- **High computational cost** due to message passing.
- **Poor performance on heterophilic graphs** where neighboring nodes have dissimilar features.

 **Hop-PMLP addresses these problems** by combining **PMLP’s efficiency** with **HopGNN’s multi-hop aggregation** to improve performance on sparse, heterophilic graphs.

---
## Previous Work

This project builds upon existing research in **Graph Neural Networks (GNNs), Propagational MLPs (PMLPs), and HopGNNs**. Specifically, it is inspired by:

- **PMLP: Graph Neural Networks are Inherently Good Generalizers**  
  *Yang et al., 2023*  
  📄 [Paper Link](https://arxiv.org/abs/2212.09034)  
  🔗 [Code Repository](https://github.com/chr26195/PMLP)  

- **HopGNN: From Node Interaction to Hop Interaction**  
  *Chen et al., 2023*  
  📄 [Paper Link](https://arxiv.org/abs/2211.11761)  
  🔗 [Code Repository](https://github.com/JC-202/HopGNN)  

Our implementation extends these works by integrating **HopGNN’s message-passing abilities into PMLP**, focusing on **heterophilic, sparse graphs**.

---
## Code and Implementation
This repository builds upon existing code from **PMLP** and **HopGNN**:

- **PMLP Code:** Adapted from [Yang et al. (2023)](https://github.com/chr26195/PMLP)
- **HopGNN Code:** Borrowed key components from [Chen et al. (2023)](https://github.com/JC-202/HopGNN)

### Key Modifications:
- Integrated **HopGNN layer** into PMLP.
- Allowed **conditional activation** of HopGNN's message passing.
- Adapted **training/inference loops** from PMLP.

---

## Installation
### **Dependencies**
To run this project, install the required dependencies:

pip install -r requirements.txt

## Usage

### **Training the Model**
Run the following command to train Hop-PMLP:
```bash
python main.py --dataset <DATASET> --lr 0.01 --dropout 0.5 --num_layers 2
```
---

## Datasets
Hop-PMLP is evaluated on **heterophilic and homophilic graphs**:

| **Dataset**  | **Nodes** | **Edges** | **Heterophily** | **Classes** |
|-------------|---------:|---------:|--------------:|-----------:|
| **Wisconsin** | 251 | 515 | High | 5 |
| **Texas** | 183 | 325 | High | 5 |
| **Cornell** | 183 | 293 | High | 5 |
| **Cora** | 2,708 | 10,556 | Low | 7 |
| **Pubmed** | 19,717 | 88,651 | Low | 3 |

**Heterophilic Graphs:** Wisconsin, Texas, Cornell  
**Homophilic Graphs:** Cora, Pubmed
**Based on 5 runs**

## Results
Hop-PMLP consistently **outperforms PMLP and GNN** on heterophilic graphs.

| **Model**  | **Wisconsin** | **Texas** | **Cornell** | **Cora** | **Pubmed** |
|------------|-------------:|----------:|-----------:|--------:|---------:|
| **Hop-PMLP** | 81.96% ± 6.99 | 80.00% ± 10.57 | 72.97% ± 5.06 | 76.64% ± 2.31 | 76.62% ± 0.66 |
| **PMLP** | 59.61% ± 5.11 | 63.24% ± 5.92 | 55.14% ± 6.22 | 73.96% ± 0.59 | 76.00% ± 1.02 |
| **GNN** | 59.22% ± 5.44 | 59.46% ± 3.31 | 56.22% ± 10.00 | 74.36% ± 0.72 | 75.80% ± 0.93 |

**Trade-off:** Hop-PMLP improves accuracy but has **higher computational costs** than PMLP.

## Limitations & Future Work
**Computational Overhead:**  
- Adding the HopGNN layer improves accuracy but increases **training time** compared to standard PMLP.

 **Future Work:**  
To improve efficiency, we can:
- **Precompute** multi-hop features instead of computing them during training.
- Optimize HopGNN integration **to reduce computational complexity**.
- Investigate **lighter message-passing mechanisms** to maintain performance without the full HopGNN inference cost.

