import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# set tracking uri lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Sentiment_Local")

# load dataset
df = pd.read_csv('tokopedia-preprocessing.csv')
df.dropna(subset=['text_clean', 'sentiment'], inplace=True)

X = df['text_clean']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)

model = LogisticRegression(max_iter=200, C=1.0, solver='lbfgs')
model.fit(X_train_tfidf, y_train)

# menyimpan ke MLflow lokal
with mlflow.start_run(run_name="model_docker"):
    mlflow.sklearn.log_model(model, "model")
    run_id = mlflow.active_run().info.run_id
    print(f"model tersimpan Run ID: {run_id}")