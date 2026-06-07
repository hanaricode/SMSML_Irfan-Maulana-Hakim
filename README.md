# 🤖 ML System Monitoring — Tokopedia Product Reviews Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.12.7-FFE873)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.19.0-blue)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-Tracked-orange)](https://dagshub.com/hanaricode/SMSML_Irfan-Maulana-Hakim)
[![Flask](https://img.shields.io/badge/Flask-latest-000000)](https://flask.palletsprojects.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-latest-E6522C)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-v13.0-F46800)](https://grafana.com/)

## 📝 Description

This project is an end-to-end **Machine Learning System Monitoring** implementation using **Tokopedia product review data** for sentiment analysis. It covers the full MLOps pipeline from automated data preprocessing and model training with experiment tracking via MLflow and DagsHub, to containerized deployment with real-time monitoring via Prometheus and Grafana.

The project is structured across three interconnected GitHub repositories:

| Repository | Description |
| --- | --- |
| [SMSML_Irfan-Maulana-Hakim](https://github.com/hanaricode/SMSML_Irfan-Maulana-Hakim) | Main project repository (this repo) |
| [Eksperimen_SML_Irfan-Maulana-Hakim](https://github.com/hanaricode/Eksperimen_SML_Irfan-Maulana-Hakim) | Preprocessing & CI/CD experiment pipeline |
| [Workflow-CI](https://github.com/hanaricode/Workflow-CI) | MLflow-based CI workflow with GitHub Actions |

🔗 **DagsHub Experiment Tracking:** https://dagshub.com/hanaricode/SMSML_Irfan-Maulana-Hakim

🐳 **Docker Hub Image:** https://hub.docker.com/r/irfan174/sentiment-model

---

## 📦 Libraries Used

| Library | Version | Purpose |
| --- | --- | --- |
| `scikit-learn` | latest | Model training, TF-IDF vectorization, label encoding |
| `mlflow` | 2.19.0 | Experiment tracking & model/artifact logging |
| `dagshub` | latest | Remote MLflow tracking server integration |
| `flask` | latest | REST API for model inference service |
| `prometheus-client` | latest | Metrics collection and exposure (10 metrics) |
| `pandas` | latest | Data manipulation |
| `numpy` | latest | Numerical operations |
| `matplotlib` | latest | Visualization (confusion matrix, feature importance, tuning plot) |
| `seaborn` | latest | Heatmap for confusion matrix |
| `nltk` | latest | Indonesian stopwords removal & tokenization |
| `pickle` | built-in | Load saved model and TF-IDF vectorizer |

---


## 📁 Repository Structure

```
SMSML_Irfan-Maulana-Hakim/
├── .github/
│   └── workflows/
│       └── preprocessing.yml                 # GitHub Actions CI/CD workflow
├── preprocessing/
│   ├── automate_Irfan-Maulana-Hakim.py       # Automated preprocessing pipeline
│   ├── Eksperimen_Irfan_Maulana_Hakim.ipynb
│   └── tokopedia-preprocessing.csv           # Output of preprocessing (auto-updated by CI)
├── Membangun_model/
│   ├── modelling.py                          # Baseline model training (MLflow + DagsHub)
│   ├── modelling_tuning.py                   # Hyperparameter tuning (MLflow + DagsHub)
│   ├── requirements.txt                      # Dependencies for model training
│   ├── DagsHub.txt                           # DagsHub repository link
│   ├── model_local/
│   │   ├── model.pkl                         # Saved trained model
│   │   └── tfidf.pkl                         # Saved TF-IDF vectorizer
│   └── artifacts/                            # MLflow logged artifacts
├── Monitoring dan Logging/      
│   ├── 4.bukti monitoring Prometheus/        # Prometheus monitoring evidence
│   ├── 5.bukti monitoring Grafana/           # Grafana dashboard evidence
│   ├── 6.bukti alerting Grafana/             # Grafana alerting evidence
│   ├── 1.bukti_serving.png                   # Model serving evidence
│   ├── 2.prometheus.yml                      # Prometheus scrape configuration
│   ├── 3.prometheus_exporter.py              # Script to send periodic test requests
│   ├── 7.inference.py                        # Flask inference service with Prometheus metrics
│   └── DockerHub.txt                         # Docker Hub image link
├── tokopedia-product-reviews-2019.csv        # Raw dataset
├── Eksperimen_SML_Irfan-Maulana-Hakim.txt
└── Workflow-CI.txt
```

---

## 📌 Dataset

| Info | Detail |
| --- | --- |
| **Source** | Tokopedia Product Reviews 2019 |
| **Raw File** | `tokopedia-product-reviews-2019.csv` |
| **Preprocessed File** | `preprocessing/tokopedia-preprocessing.csv` |
| **Key Columns** | `text` (review text), `rating` (1–5) |
| **Target Column** | `sentiment` (derived from rating) |
| **Language** | Indonesian |
| **Task** | Sentiment Classification (3 classes: Positif, Netral, Negatif) |

---

## ⚙️ Preprocessing Pipeline

The preprocessing script (`automate_Irfan-Maulana-Hakim.py`) handles the full data cleaning and preparation pipeline from raw Tokopedia reviews to model-ready data. It is also automated via GitHub Actions CI/CD.

### Sentiment Mapping from Rating

| Rating | Sentiment |
| --- | --- |
| ≥ 4 | `Positif` |
| = 3 | `Netral` |
| ≤ 2 | `Negatif` |

### Preprocessing Steps

| Step | Method | Description |
| --- | --- | --- |
| Load Data | `pd.read_csv()` | Load raw dataset from `tokopedia-product-reviews-2019.csv` |
| Select Columns | `df[['text', 'rating']]` | Use only relevant columns |
| Drop NA | `dropna()` | Remove rows with missing `text` or `rating` |
| Drop Duplicates | `drop_duplicates()` | Remove duplicate rows |
| Filter Short Text | `len(text) >= 3` | Remove texts shorter than 3 characters |
| Sentiment Labeling | Custom function | Map rating to `Positif`, `Netral`, `Negatif` |
| Lowercasing | `str.lower()` | Convert all text to lowercase |
| Remove Numbers | `re.sub(r'\d+', '', text)` | Remove all numeric characters |
| Remove Punctuation | `str.maketrans()` | Remove all punctuation marks |
| Normalize Whitespace | `re.sub(r'\s+', ' ', text)` | Remove extra whitespace |
| Remove Stopwords | NLTK Indonesian stopwords | Remove common Indonesian stopwords |
| Label Encoding | `LabelEncoder()` | Encode sentiment labels to numeric values |
| Save Output | `to_csv()` | Save to `preprocessing/tokopedia-preprocessing.csv` |

---

## 🤖 Model Training & Experiment Tracking

All experiments are tracked using **MLflow** integrated with **DagsHub** for remote experiment management. Two training scripts are provided:

### `modelling.py` - Baseline Model

Trains a baseline Logistic Regression model and logs all parameters, metrics, and artifacts to DagsHub via MLflow.

```python
dagshub.init(repo_owner='hanaricode', repo_name='SMSML_Irfan-Maulana-Hakim', mlflow=True)
mlflow.set_experiment("Sentiment_Analysis_Tokopedia")
```

| Component | Detail |
| --- | --- |
| **Algorithm** | Logistic Regression |
| **Feature Extraction** | TF-IDF (`max_features=5000`, `ngram_range=(1,2)`) |
| **Data Split** | 80% train, 20% test (`stratify=y`) |
| **Run Name** | `LogisticRegression_baseline` |
| **Experiment Name** | `Sentiment_Analysis_Tokopedia` |
| **Tracking** | MLflow + DagsHub (remote) |

**Logged Parameters:**

| Parameter | Value |
| --- | --- |
| `max_iter` | 200 |
| `C` | 1.0 |
| `solver` | lbfgs |
| `max_features_tfidf` | 5000 |
| `ngram_range` | (1, 2) |
| `test_size` | 0.2 |
| `random_state` | 42 |

**Logged Metrics:** `accuracy`, `precision`, `recall`, `f1_score`

**Logged Artifacts:**

| Artifact | Description |
| --- | --- |
| `confusion_matrix.png` | Heatmap visualization of prediction results |
| `classification_report.json` | Detailed per-class precision, recall, F1 |
| `feature_importance.jpg` | Top 20 words per sentiment class (coefficient scores) |

---

### `modelling_tuning.py` - Hyperparameter Tuning

Runs multiple training experiments across a parameter grid, tracking each run separately in DagsHub via MLflow. The best model is selected based on the highest F1-score.

```python
dagshub.init(repo_owner='hanaricode', repo_name='SMSML_Irfan-Maulana-Hakim', mlflow=True)
mlflow.set_experiment("Sentiment_Analysis_Tokopedia_Tuning")
```

| Component | Detail |
| --- | --- |
| **Algorithm** | Logistic Regression |
| **Feature Extraction** | TF-IDF (`max_features=10000`, `ngram_range=(1,2)`) |
| **Data Split** | 80% train, 20% test (`stratify=y`) |
| **Experiment Name** | `Sentiment_Analysis_Tokopedia_Tuning` |
| **Tracking** | MLflow + DagsHub (remote) |
| **Best Model Selection** | Highest F1-score across all runs |

**Parameter Grid (5 runs):**

| Run Name | C | max_iter | solver |
| --- | --- | --- | --- |
| `LR_C0.1_iter100` | 0.1 | 100 | lbfgs |
| `LR_C0.5_iter200` | 0.5 | 200 | lbfgs |
| `LR_C1.0_iter200` | 1.0 | 200 | lbfgs |
| `LR_C2.0_iter300` | 2.0 | 300 | lbfgs |
| `LR_C5.0_iter300` | 5.0 | 300 | lbfgs |

**Logged Artifacts:**

| Artifact | Description |
| --- | --- |
| `tuning_result.csv` | Full results table of all 5 runs |
| `tuning_f1_plot.jpg` | F1-score comparison chart across C values |

---

## 🚀 Deployment & Monitoring Stack

The trained model is served via a **Flask inference service** (`7.inference.py`) and monitored in real-time using **Prometheus** and **Grafana**.

### Architecture

```
  User / prometheus_exporter.py (sends requests every 2s)
       │
       │ POST /predict  {"text": "..."}
       ▼
┌──────────────────────────┐
│   Flask Inference Service │  - loads model.pkl & tfidf.pkl
│       (port 5002)         │  - returns sentiment + confidence + latency
│     7.inference.py        │  - records 10 Prometheus metrics
└────────────┬─────────────┘
             │
             │ expose GET /metrics
             ▼
┌──────────────────────────┐
│       Prometheus          │  - scrapes /metrics every 10 seconds
│       (port 9090)         │  - target: host.docker.internal:5002
│                           │  - job_name: sentiment-model
└────────────┬──────────────┘
             │
             │ used as data source
             ▼
┌──────────────────────────┐
│         Grafana           │  - reads data from Prometheus
│       (port 3000)         │  - displays live monitoring dashboard
│                           │  - supports alerting configuration
└──────────────────────────┘
```

**How it works:**
- `prometheus_exporter.py` sends random Tokopedia review texts to the inference service every **2 seconds**, simulating real traffic for monitoring purposes.
- `7.inference.py` (Flask) receives each request, transforms the text using the saved TF-IDF vectorizer (`tfidf.pkl`), runs the Logistic Regression model (`model.pkl`), and returns the predicted sentiment with confidence score and latency.
- On every request, **10 Prometheus metrics** are updated automatically.
- Prometheus scrapes the `/metrics` endpoint every **10 seconds**, storing time-series monitoring data.
- Grafana connects to Prometheus as a data source and visualizes metrics as a live dashboard with alerting support.

### API Endpoints (Flask Inference Service — port 5002)

| Endpoint | Method | Description |
| --- | --- | --- |
| `/predict` | POST | Accepts `{"text": "..."}`, returns sentiment + confidence + latency |
| `/metrics` | GET | Exposes Prometheus metrics |
| `/health` | GET | Health check — returns `{"status": "healthy"}` |

### Prediction Input & Output

**Input:**
```json
{"text": "produk bagus banget, pengiriman cepat"}
```

**Output:**
```json
{
  "sentiment": "Positif",
  "confidence": 0.923,
  "latency": 0.003
}
```

---

## 📊 Prometheus Metrics

The inference service exposes **10 metrics** to Prometheus:

| Metric | Type | Description |
| --- | --- | --- |
| `request_count_total` | Counter | Total number of requests received |
| `request_latency_seconds` | Histogram | Time taken per request (in seconds) |
| `prediction_count_total` | Counter | Total predictions broken down by sentiment label |
| `positive_prediction_total` | Counter | Total predictions classified as `Positif` |
| `negative_prediction_total` | Counter | Total predictions classified as `Negatif` |
| `neutral_prediction_total` | Counter | Total predictions classified as `Netral` |
| `error_count_total` | Counter | Total number of errors during prediction |
| `input_text_length` | Histogram | Distribution of input text length (characters) |
| `active_requests` | Gauge | Number of currently active requests |
| `model_confidence` | Histogram | Distribution of model confidence scores |

**Prometheus Configuration (`2.prometheus.yml`):**
```yaml
global:
  scrape_interval: 10s
  evaluation_interval: 10s

scrape_configs:
  - job_name: 'sentiment-model'
    static_configs:
      - targets: ['host.docker.internal:5002']
    metrics_path: '/metrics'
```

---

## 📈 Grafana Dashboard

Grafana connects to Prometheus as a data source and visualizes the ML system metrics as a live dashboard with alerting support. Evidence of monitoring and alerting results can be found in:

| Folder | Contents |
| --- | --- |
| `Monitoring dan Logging/4.bukti monitoring Prometheus/` | Prometheus monitoring screenshots |
| `Monitoring dan Logging/5.bukti monitoring Grafana/` | Grafana dashboard screenshots |
| `Monitoring dan Logging/6.bukti alerting Grafana/` | Grafana alerting configuration & results |

---

## 🔁 CI/CD Pipeline

The project uses **GitHub Actions** for automated preprocessing triggered on every push to `main`.

**Workflow file:** `.github/workflows/preprocessing.yml`

| Step | Action | Description |
| --- | --- | --- |
| Checkout | `actions/checkout@v4` | Clone repository |
| Setup Python | `actions/setup-python@v5` | Python `3.12.7` |
| Install Dependencies | `pip install` | pandas, numpy, scikit-learn, nltk |
| Run Preprocessing | `python automate_Irfan-Maulana-Hakim.py` | Execute full preprocessing pipeline |
| Upload Artifact | `actions/upload-artifact@v4` | Save `tokopedia-preprocessing.csv` as workflow artifact |
| Commit & Push | `git push` | Auto-commit updated preprocessing result to repo |

---

## ⚙️ How to Run Locally

### Prerequisites

- Python 3.12.7
- Docker Desktop
- DagsHub account (for experiment tracking)

### 1. Clone this repository

```bash
git clone https://github.com/hanaricode/SMSML_Irfan-Maulana-Hakim.git
cd SMSML_Irfan-Maulana-Hakim
```

### 2. Install dependencies

```bash
pip install -r Membangun_model/requirements.txt
```

### 3. Run preprocessing

```bash
python preprocessing/automate_Irfan-Maulana-Hakim.py
```

### 4. Run baseline model training

```bash
python Membangun_model/modelling.py
```

### 5. Run hyperparameter tuning

```bash
python Membangun_model/modelling_tuning.py
```

### 6. Start Flask inference service

```bash
python "Monitoring dan Logging/7.inference.py"
```

### 7. Start Prometheus exporter (send test requests)

```bash
python "Monitoring dan Logging/3.prometheus_exporter.py"
```

### 8. Access the services

| Service | URL |
| --- | --- |
| Flask Inference API | http://localhost:5002/predict |
| Health Check | http://localhost:5002/health |
| Prometheus Metrics | http://localhost:5002/metrics |
| Prometheus UI | http://localhost:9090 |
| Grafana Dashboard | http://localhost:3000 |

---

## 👤 Author & 📄 License

- **Name** : Hanari
- **Platform** : Antigravity IDE 2.0.4
- © 2026 Hanari. Licensed under the [MIT License](LICENSE).
