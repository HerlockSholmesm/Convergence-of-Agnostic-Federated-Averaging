# Convergence-of-Agnostic-Federated-Averaging
The experiments for the Paper
# Federated Learning with Skewed Participation

This repository contains experiments accompanying our paper, investigating the behavior of Federated Averaging (FedAvg) under skewed client participation.

We perform two primary experiments:

1. **MNIST Classification** using softmax regression (multiclass logistic regression).
2. **Synthetic Linear Regression** with convex loss.

In both settings, we simulate a federated environment with:
- \( N = 100 \) users
- \( M = 10 \) users per communication round
- Skewed client participation probabilities

## 📊 Skewed Participation Design

To introduce non-uniform user participation, each client \( i \in \{0, \dots, N-1\} \) is assigned a probability:

\[
q_i = \frac{\exp(-i / \tau)}{\sum_{j=0}^{N-1} \exp(-j / \tau)}, \quad \text{with } \tau = 10
\]

At each round, \( M \) users are sampled without replacement from this distribution. The **marginal participation probabilities** \( p_i \) are then **empirically estimated** by repeated sampling from \( q \), giving:

\[
p_i \approx \mathbb{P}[i \in \text{subset sampled from } q]
\]

This procedure simulates participation skew, where lower-indexed clients participate more frequently.

## 📁 Structure

- `mnist_fedavg.ipynb`: MNIST classification experiment
- `linear_regression_fedavg.ipynb`: Synthetic regression experiment
- `figures/`: Generated plots used in the paper
  - `fig2_mnist_loss_curve.png`
  - `fig3_regression_loss_curve.png`
  - `fig4_loss_vs_skew_scatter.png`
- `results/`: `.npz` and `.csv` files containing raw experimental results

## 📈 Plots

We report:
- Global test loss across rounds (log-scale)
- Final performance gap between **Weighted FedAvg** and **Agnostic FedAvg**
- Correlation with skew magnitude \( \|\mathbf{p} - \mathbf{u}\|_1 \)

## 📦 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
