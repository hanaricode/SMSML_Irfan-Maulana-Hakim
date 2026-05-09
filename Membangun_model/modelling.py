import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


# setup Dagshub dan MLflow
dagshub.init(
    repo_owner="hanaricode", repo_name="SMSML_Irfan-Maulana-Hakim", mlflow=True
)

# load dataset
df = pd.read_csv("tokopedia-preprocessing.csv")
print(f"dataset dimuat. Shape: {df.shape}")

# drop baris nilai kosong
df.dropna(subset=["text_clean", "sentiment"], inplace=True)

# persiapan fitur
x = df["text_clean"]
y = df["sentiment"]

# split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# tf-idf vectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

# MLflow eksperiment
mlflow.set_experiment("Sentiment_Analysis_Tokopedia")

with mlflow.start_run(run_name="LogisticRegression_baseline"):
    # parameter model
    params = {
        "max_iter": 200,
        "C": 1.0,
        "solver": "lbfgs",
        "max_features_tfidf": 5000,
        "ngram_range": "(1, 2)",
        "test_size": 0.2,
        "random_state": 42,
    }

    # log parameter
    mlflow.log_params(params)

    # train model
    model = LogisticRegression(
        max_iter=params["max_iter"], C=params["C"], solver=params["solver"]
    )

    model.fit(x_train_tfidf, y_train)

    # prediksi
    y_pred = model.predict(x_test_tfidf)

    # metriks
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\nAkurasi: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # log metriks
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)


    # artefak 1: Confusion matrix
    os.makedirs("artifacts", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("artifacts/confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("artifacts/confusion_matrix.png")


    # artefak 2: Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    with open("artifacts/classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact("artifacts/classification_report.json")


    # artefak 3: feature importance
    feature_names = tfidf.get_feature_names_out()
    coefficients = model.coef_

    # plot kata perkelas sentiment
    fig,axes = plt.subplots(1, 3, figsize=(17, 4))
    classes = model.classes_

    for i, (cls, ax) in enumerate(zip(classes, axes)):
        top_indices = np.argsort(coefficients[i])[-20:]
        top_words = [feature_names[j] for j in top_indices]
        top_score = coefficients[i][top_indices]

        ax.barh(top_words, top_score, color='skyblue')
        ax.set_title(f'top 20 kata - {cls}')
        ax.set_xlabel('coefficients score')
        ax.tick_params(axis='y', labelsize=7)

    plt.suptitle('Feature importance perkelas sentiment', fontsize=15)
    plt.tight_layout()
    plt.savefig("artifacts/feature_importance.jpg")
    plt.close()
    mlflow.log_artifact("artifacts/feature_importance.jpg")


    # log model
    mlflow.sklearn.log_model(model, "model")

    print("\n MLflow logging selesai dan eksperimen berhasil disimpan di Dagshub.")
