# 🏆 Hop-PMLP: Single-Layer HopGNN-Enhanced PMLP for Sparse Heterophilic Graphs

![GitHub license](https://img.shields.io/github/license/AlexandrosKyr/Hop-PMLP)
![GitHub stars](https://img.shields.io/github/stars/AlexandrosKyr/Hop-PMLP?style=social)
![GitHub forks](https://img.shields.io/github/forks/AlexandrosKyr/Hop-PMLP?style=social)

## 🚀 Overview
Hop-PMLP is a hybrid deep learning model that **combines Propagational Multi-Layer Perceptron (PMLP) with Hop Graph Neural Networks (HopGNN)** to improve performance on **heterophilic and sparse graphs**.

### 🔥 Key Features:
- ✅ **Single HopGNN layer** in training to capture graph structure.
- ✅ **Full HopGNN message-passing** in inference for accuracy boost.
- ✅ **Works well on heterophilic graphs** (Wisconsin, Texas, Cornell).

🔗 **Project Repository:** [https://github.com/AlexandrosKyr/Hop-PMLP](https://github.com/AlexandrosKyr/Hop-PMLP)  
📄 **Paper:** [Link to Paper (if available)]  

---

## 📚 Table of Contents
- [🚀 Overview](#-overview)
- [📌 Motivation](#-motivation)
- [🛠️ Code and Implementation](#️-code-and-implementation)
- [📦 Installation](#-installation)
- [📝 Usage](#-usage)
- [📊 Datasets](#-datasets)
- [📈 Results](#-results)
- [⚠️ Limitations & Future Work](#️-limitations--future-work)
- [📖 Citation](#-citation)

---

## 📌 Motivation
Traditional **Graph Neural Networks (GNNs)** struggle with:
- ❌ **Over-smoothing** in deep architectures.
- ❌ **High computational cost** due to message passing.
- ❌ **Poor performance on heterophilic graphs** where neighboring nodes have dissimilar features.

🚀 **Hop-PMLP addresses these problems** by combining **PMLP’s efficiency** with **HopGNN’s multi-hop aggregation** to improve performance on sparse, heterophilic graphs.

---

## 🛠️ Code and Implementation
This repository builds upon existing code from **PMLP** and **HopGNN**:

- 🟢 **PMLP Code:** Adapted from [Yang et al. (2023)](https://github.com/chr26195/PMLP)
- 🔵 **HopGNN Code:** Borrowed key components from [Chen et al. (2023)](https://github.com/JC-202/HopGNN)

### 🔑 Key Modifications:
- ✅ Integrated **HopGNN layer** into PMLP.
- ✅ Allowed **conditional activation** of HopGNN's message passing.
- ✅ Adapted **training/inference loops** from PMLP.

---

## 📦 Installation
### 🔧 **Dependencies**
To run this project, install the required dependencies:

```bash
pip install -r requirements.txt
