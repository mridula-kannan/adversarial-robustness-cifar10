# Adversarial Robustness on CIFAR-10 — PGD Attack & Adversarial Training

A PyTorch implementation studying the vulnerability of CNNs to adversarial attacks and the effectiveness of PGD-based adversarial training as a defense mechanism — evaluated on the CIFAR-10 benchmark dataset.

## Overview

Deep learning models are highly susceptible to adversarial examples: inputs that are imperceptibly perturbed to a human eye but cause confident misclassification. This project:

1. Trains a **standard CNN baseline** on CIFAR-10
2. Attacks it using **Projected Gradient Descent (PGD)** to quantify vulnerability
3. Applies **PGD-based adversarial training** as a defense
4. Compares clean accuracy vs. adversarial robustness across both models

## Results

| Model | Clean Accuracy | Adversarial Accuracy (PGD) |
|---|---|---|
| Standard CNN | 76.25% | 2.29% |
| Adversarially Trained CNN | 49.46% | 31.48% |

**Key Finding:** Adversarial training improved PGD robustness by +29.19 percentage points, demonstrating a clear but costly defense — with a ~27-point drop in clean accuracy highlighting the well-documented robustness-accuracy trade-off.

## Architecture

A custom CNN built in PyTorch:

```
Input (3×32×32)
→ Conv2d(3→32) + BatchNorm + ReLU + MaxPool → 16×16
→ Conv2d(32→64) + BatchNorm + ReLU + MaxPool → 8×8
→ Conv2d(64→128) + BatchNorm + ReLU + MaxPool → 4×4
→ Dropout(0.3)
→ FC(2048→256) + ReLU
→ FC(256→10)
```

## Attack: Projected Gradient Descent (PGD)

PGD is an iterative white-box attack that generates adversarial examples by following the gradient of the loss function, clipped within an epsilon-ball:

**Attack parameters used:**

| Parameter | Training Attack | Evaluation Attack |
|---|---|---|
| Epsilon (ε) | 8/255 | 4/255 |
| Step size (α) | 2/255 | 1/255 |
| Iterations | 7 | 5 |

## Adversarial Training

During adversarial training, PGD-generated examples replace clean inputs in the training loop — forcing the model to learn features that are stable under perturbation. This approach follows the framework introduced by Madry et al. (2018).

## Setup & Usage

**Requirements**
```
torch
torchvision
numpy
matplotlib
```

Install:
```bash
pip install torch torchvision numpy matplotlib
```

**Run the notebook:**
```bash
jupyter notebook DEEP_LEARNING_FINAL_PROJECT.ipynb
```

CIFAR-10 downloads automatically on first run via torchvision.datasets.CIFAR10.

**Saved outputs:**
- saved_models/ — trained model weights
- results/ — accuracy logs
- report_figures/ — training loss curves and adversarial image comparisons

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss Function | Cross-Entropy |
| Batch Size | 128 |
| Epochs | 10 |
| Data Augmentation | Random crop (pad=4), horizontal flip |
| Device | CUDA / CPU (auto-detected) |

## Key Insights

1. A standard CNN with 76% clean accuracy collapses to 2.29% under PGD — demonstrating how fragile undefended models are against even modest perturbations.

2. Adversarial training provides a significant robustness boost of +29 percentage points but introduces a clean accuracy penalty of ~27 points, consistent with the robustness-accuracy trade-off documented in the literature.

3. The iterative nature of PGD training forces the model to learn more stable features, but this comes at the cost of generalization on unperturbed data.

4. Adversarial training roughly doubles computational cost compared to standard training due to the need to generate adversarial examples during each epoch.

## References

- Goodfellow, I., Shlens, J., & Szegedy, C. (2014). Explaining and Harnessing Adversarial Examples
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vlatt, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch Documentation: https://pytorch.org

## Author

**Mridula Kannan**
M.S. Computer and Information Science, University of Alabama at Birmingham
```
