# 🎬 Neural Collaborative Filtering Recommender System

This project implements a **deep learning-based recommender system** using **Neural Collaborative Filtering (NeuMF)** to predict user-item interactions. The goal is to generate personalized item recommendations — for example, recommending movies to users based on past ratings.

---

## 📚 What Is Neural Collaborative Filtering?

Neural Collaborative Filtering (NeuMF) is a hybrid recommendation model that combines:

- **Generalized Matrix Factorization (GMF)** — captures linear interactions between users and items.
- **Multi-Layer Perceptron (MLP)** — captures non-linear, complex patterns.

Both components use **separate embedding layers** for users and items. Their outputs are combined and passed through dense layers to produce a final prediction score. The architecture is trained end-to-end using supervised learning on implicit/explicit feedback data.

Core strengths of NeuMF:
- Learns **user-item interaction patterns**
- Supports **non-linear feature combinations**
- Scales well with embeddings

---

## ✅ Current Progress

### ✅ Model Architecture
- Implemented NeuMF from scratch using Keras
- Separate embeddings for user and item IDs
- Combined GMF + MLP architecture with:
  - Flatten, Dense, ReLU, and Sigmoid layers
  - Final output predicts interaction score

### ✅ Features Implemented
- Mapped user/item IDs to integer indices
- Trained on MovieLens 1M dataset (explicit feedback)
- Early stopping for better generalization
- Predictions and recommendation generation
- Model summary & prediction analysis
- Reverse ID mapping for interpretability

### 🧪 Currently Working On

- Preparing the model for evaluation with metrics like **Hit@K**, **Precision@K**

---

## 🔭 Future Enhancements

### 🚀 Model Evaluation
- Compute **Hit@K**, **Precision@K**, and **NDCG@K** to evaluate top-N recommendations
- Filter out already-watched items from recommendations

### 💡 Feature Improvements
- Include side features (e.g., genres, timestamps, user demographics)
- Build hybrid models combining content + collaborative signals

### 🧰 Tooling & Deployment
- Save model and serve it with **Streamlit** or **Flask**
- Package it in Docker for reproducible deployment
- Add an interactive frontend to demo user-based recommendations


---

## 📁 Dataset

- MovieLens 1M (https://grouplens.org/datasets/movielens/1m/)
- 1 million ratings across ~6,000 users and ~4,000 movies

## 📄 Reference

This implementation is based on the paper:

> **He et al., "Neural Collaborative Filtering", WWW 2017**  
> [https://arxiv.org/abs/1708.05031](https://arxiv.org/abs/1708.05031)

Key takeaways from the paper that influenced this implementation:
- Traditional matrix factorization assumes linearity in user-item interaction.
- NeuMF replaces dot-product with deep neural networks to model non-linear relationships.
- The final architecture combines GMF and MLP for robust performance.

This project aims to bridge theoretical research and practical recommendation systems.


---

## 🤝 Contributions & Learning

This project is a hands-on implementation to:
- Deepen understanding of **recommender systems**
- Build practical skills in **deep learning architecture**, **model evaluation**, and **data pipelines**
- Prepare for real-world machine learning interviews and portfolio work

---

