import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# load dataset
df = pd.read_csv("tokopedia-preprocessing.csv")
df.dropna(subset=["text_clean", "sentiment"], inplace=True)

x = df["text_clean"]
y = df["sentiment"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
x_train_tfidf = tfidf.fit_transform(x_train)

model = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
model.fit(x_train_tfidf, y_train)

# simpan model dan tfidf
os.makedirs("model_local", exist_ok=True)
with open("model_local/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model_local/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("model berhasil disimpan di model_local/")
