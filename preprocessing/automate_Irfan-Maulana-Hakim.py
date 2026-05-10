import pandas as pd
import numpy as np
import re
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

# load dataset
df = pd.read_csv("tokopedia-product-reviews-2019.csv")
print(f"Dataset berhasil dimuat. Shape: {df.shape}")

# kolom
df_clean = df[["text", "rating"]].copy()

# missing values
df_clean.dropna(subset=["text", "rating"], inplace=True)
print(f"Shape setelah drop NA: {df_clean.shape}")

# hapus duplikat
df_clean.drop_duplicates(inplace=True)
print(f"Shape setelah drop duplikat: {df_clean.shape}")

# hapus teks terpendek
df_clean = df_clean[df_clean["text"].astype(str).apply(len) >= 3]
print(f"Shape setelah hapus teks pendek: {df_clean.shape}")


# konversi rating ke sentimen
def rating_to_sentiment(rating):
    if rating >= 4:
        return "Positif"
    elif rating == 3:
        return "Netral"
    else:
        return "Negatif"


df_clean["sentiment"] = df_clean["rating"].apply(rating_to_sentiment)
print("\nDistribusi Sentimen:")
print(df_clean["sentiment"].value_counts())


# text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


df_clean["text_clean"] = df_clean["text"].apply(clean_text)

# hapus stopwords
stop_words = set(stopwords.words("indonesian"))


def remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


df_clean["text_clean"] = df_clean["text_clean"].apply(remove_stopwords)

# label encoding
le = LabelEncoder()
df_clean["sentiment_encoded"] = le.fit_transform(df_clean["sentiment"])
print("\nLabel Encoding:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

# simpan hasil
os.makedirs("preprocessing", exist_ok=True)
output_path = "preprocessing/tokopedia-preprocessing.csv"
df_clean.to_csv(output_path, index=False)
print(f"\n Berhasil disimpan di: {output_path}")
