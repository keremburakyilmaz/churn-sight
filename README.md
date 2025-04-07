# ChurnSight

ChurnSight is a complete end-to-end MLOps pipeline built to predict customer churn. It features fully custom-written models, hyperparameter optimization, model evaluation, and CI/CD integration.

---

## Features

- **Custom Models**
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - XGBoost (using custom Decision Tree)
  - MLP
  - Gaussian Naive Bayes
  - LightGBM (imported)

- **Meta Classifier**
  - Stacks multiple models for ensemble predictions

- **Training Utilities**
  - Modular `train_optuna.py` for hyperparameter tuning
  - Model training scripts for each algorithm
  - Evaluation with ROC-AUC and accuracy

- **FastAPI Inference API**
  - `/predict`: Single instance prediction
  - `/batch-predict`: CSV batch predictions
  - `/explain`: SHAP-based explanations

- **CI/CD**
  - GitHub Actions: Linting, testing, Docker build
  - Dockerized API: Production-ready containerization

---

## Directory Structure

```
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Train/test splits
├── models/                 # Saved models and metadata
├── src/
│   ├── api/                # FastAPI app and schemas
│   ├── data/               # Preprocessing pipeline
│   └── train/              # All training scripts
├── tests/                  # Pytest unit/integration tests
├── requirements.txt
├── Dockerfile
└── .github/workflows/     # CI/CD config
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess data
python src/data/preprocess.py

# Train with a model
python src/train/train_\*.py

# Start FastAPI server
uvicorn src.api.main:app --reload
```

---

## Evaluation

All models are evaluated on:
- Accuracy
- ROC-AUC

The meta-classifier currently achieves:
```
Accuracy: 0.7991
ROC-AUC:  0.8503
```

---

## Docker

```bash
docker build -t churn-sight-api .
docker run -p 8000:8000 churn-sight-api
```

---

## CI/CD

GitHub Actions CI pipeline runs on every push:
- Flake8 linting
- Pytest test suite
- Docker build verification

---

## Author

Kerem Burak Yilmaz

---

## Roadmap

- [ ] Add more unit tests.
- [ ] Implement a feedback API endpoint.
- [ ] UI development including but not limited to a form to predict a single customer and tracking the models' performance, using React and Tailwind CSS.
- [ ] Deploy the Docker on AWS and develop CD functions.
