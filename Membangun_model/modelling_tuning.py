import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import os

# setup Dagshub dan MLflow
dagshub.init(repo_owner='hanaricode', repo_name='SMSML_Irfan-Maulana-Hakim', mlflow=True)

# load dataset
df = pd.read_csv('tokopedia-preprocessing.csv')
df.dropna(subset=['text_clean', 'sentiment'], inplace=True)
print(f"dataset dimuat. Shape: {df.shape}")

# persiapan fitur
x = df['text_clean']
y = df['sentiment']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

# hyperparameter tuning
mlflow.set_experiment("Sentiment_Analysis_Tokopedia_Tuning")

param_grid = [
     {"C": 0.1, "max_iter": 100, "solver": "lbfgs"},
    {"C": 0.5, "max_iter": 200, "solver": "lbfgs"},
    {"C": 1.0, "max_iter": 200, "solver": "lbfgs"},
    {"C": 2.0, "max_iter": 300, "solver": "lbfgs"},
    {"C": 5.0, "max_iter": 300, "solver": "lbfgs"},]

best_f1 = 0
best_params = None
results = []

for params in param_grid:
    with mlflow.start_run(run_name=f"LR_C{params['C']}_iter{params['max_iter']}"):

        mlflow.log_params(params)

        model = LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            solver=params["solver"])
        
        model.fit(x_train_tfidf, y_train)
        y_pred = model.predict(x_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        results.append({**params, "accuracy": accuracy, "f1_score": f1})
        print(f"C={params['C']} | Accuracy={accuracy:.4f} | F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_model = model

        mlflow.sklearn.log_model(model, "model")


# menyimpan hasil tuning terbaik
os.makedirs("artifacts", exist_ok=True)

# artefak 1: hasil dari semua tuning
results_df = pd.DataFrame(results)
results_df.to_csv("artifacts/tuning_result.csv", index=False)
mlflow.log_artifact("artifacts/tuning_result.csv")


# artefak 2: plot perbandingan f1-score
plt.figure(figsize=(7, 4))
plt.plot([str(r["C"]) for r in results],
        [r["f1_score"] for r in results],
        marker='o', color='purple')
plt.title('F1-score per C (Hyperparamter tuning)')
plt.xlabel('nilai C')
plt.ylabel('F1-score')
plt.grid(True)
plt.tight_layout()
plt.savefig("artifacts/tuning_f1_plot.jpg")
plt.close()
mlflow.log_artifact("artifacts/tuning_f1_plot.jpg")

print("tuning selesai")
print(f"best params: {best_params}")
print(f"best F1-score: {best_f1:.4f}")