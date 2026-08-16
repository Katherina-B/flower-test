## ML Engineering Pipeline — Flower Image Classification with MLOps
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Katherina-B/ml-engineering-mlops/blob/main/notebooks/run_pipeline.ipynb)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=flat&logo=WeightsAndBiases&logoColor=black)](https://wandb.ai/katherina-barbasheva-ntu-khpi/flower-classification-interpretability?nw=nwuserkatherinabarbasheva)
[![DagsHub](https://img.shields.io/badge/DagsHub-MLflow-orange)](https://dagshub.com/katherina.barbasheva/ml-engineering-mlops/experiments)

A complete ML engineering project built across 5 labs as part of the **ML Engineering course** at NTU KhPI (2024).

End-to-end image classification pipeline using **ResNet50** on the Oxford Flowers 102 dataset, with full MLOps tooling — from data versioning to model interpretability.
---

## Results

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **96.96%** (Epoch 19) |
| Best Validation Loss | 0.1296 |
| Test Set Size | 1,020 images |
| Training Set Size | 6,149 images |
| Classes | 102 flower categories |

---

## Model

- **Architecture:** ResNet50 (pretrained on ImageNet)
- **Dataset:** [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- **Optimizer:** AdamW (lr=0.0001)
- **Loss:** CrossEntropyLoss
- **Epochs:** 20
- **Framework:** PyTorch

---

## Pipeline Stages

| Stage | Focus | Tools |
|-------|-------|-------|
| 1 | Basic training pipeline | Python, logging, type hints, ruff, black |
| 2 | Automated dataset extension | Batch splitting, data registry |
| 3 | Data version control | DVC, params.yaml |
| 4 | Experiment tracking | MLflow, DagsHub |
| 5 | Advanced tracking + interpretability | Weights & Biases, Captum |

---

## Experiment Tracking

### MLflow via DagsHub
Full experiment tracking including parameters, metrics per epoch, and model artifacts.
View experiments: https://dagshub.com/katherina.barbasheva/ml-engineering-mlops/experiments

### Weights & Biases
GradCAM interpretability results logged as image galleries.
View runs: https://wandb.ai/katherina-barbasheva-ntu-khpi/flower-classification-interpretability?nw=nwuserkatherinabarbasheva

---

## Interpretability

Two experiments implemented using **Captum**:

**Experiment 1 — GradCAM across layers**
Visualises what ResNet50 learns at different depths: from simple edges (layer1) to complex flower patterns (layer4).

**Experiment 2 — GradCAM vs GradCAM++**
Side-by-side comparison of standard GradCAM and GradCAM++ — GradCAM++ produces sharper, more localised attributions especially for smaller flower regions.

---

## Project Structure

```
flower-classification/
├── src/
│   ├── train.py            # Training loop + ResNet50
│   ├── train_mlflow.py     # Training with MLflow tracking
│   ├── evaluate.py         # Test set evaluation
│   ├── interp.py           # GradCAM + GradCAM++ interpretability
│   ├── load_date.py        # Data loading and splitting
│   └── check_cuda.py       # GPU check
├── configs/
│   └── params.yaml         # All hyperparameters and paths
├── notebooks/
│   └── run_pipeline.ipynb  # Colab notebook
├── dvc.yaml                # DVC pipeline stages
├── setup_dagshub.py        # DagsHub credentials setup
├── .env.example            # Credentials template
├── .gitignore
└── pyproject.toml
```

---



## Tech Stack

- **Model:** ResNet50 (PyTorch / torchvision)
- **Dataset:** Oxford Flowers 102 (102 categories)
- **Experiment tracking:** Weights & Biases, MLflow
- **Interpretability:** Captum (GradCAM + Saliency Maps)
- **Config management:** YAML-based (`params.yaml`)
- **Dependency management:** Poetry (`pyproject.toml`)
- **Logging:** Python logging module
- **Code quality:** ruff, black, mypy, isort

---

## Interpretability

Lab 5 includes model interpretability using **Captum**:
- **Grad-CAM** — highlights regions most influential 
  for the model's prediction
- **Saliency Maps** — pixel-level attribution visualisation
- Results logged as images to **Weights & Biases**
- Correct and incorrect predictions saved separately   for error analysis

---
## Project Structure

    ├── train.py              # Training loop + ResNet50 model definition
    ├── load_date.py          # Data loading, splitting, augmentation
    ├── interp.py             # GradCAM + Saliency interpretability
    ├── check_cuda.py         # GPU availability check
    ├── params.yaml           # Centralised configuration
    ├── pyproject.toml        # Poetry dependency management
    ├── run_pipeline.ipynb    # Colab notebook to run full pipeline
    └── result.txt            # Training logs and metrics
---
## How to Run

### Option 1 — Google Colab (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Katherina-B/ml-engineering-mlops/blob/main/notebooks/run_pipeline.ipynb)

No local setup needed. GPU included.

### Option 2 — Local

```bash
git clone https://github.com/Katherina-B/ml-engineering-mlops.git
cd YOUR_REPO_NAME
poetry install
python src/train.py
python src/evaluate.py
python src/interp.py
```

> Update paths in `configs/params.yaml` before running locally.

---

## Tech Stack

Python · PyTorch · ResNet50 · Captum · DVC · MLflow · Weights & Biases · Poetry · Git

---


*Part of the ML Engineering course — NTU KhPI, 2024*
*Instructor: Maksym Tatariants, PhD (ML Engineer @ Toshiba)*
