import pickle
import os
from flask import Flask, request, jsonify
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import time

app = Flask(__name__)

# load model dan tfidf
model_path = os.path.join(
    os.path.dirname(__file__), "../Membangun_model/model_local/model.pkl"
)
tfidf_path = os.path.join(
    os.path.dirname(__file__), "../Membangun_model/model_local/tfidf.pkl"
)

with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(tfidf_path, "rb") as f:
    tfidf = pickle.load(f)

# prometheus metrics
REQUEST_COUNT = Counter("request_count_total", "Total jumlah request")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Latency request dalam detik")
PREDICTION_COUNT = Counter("prediction_count_total", "Total prediksi", ["sentiment"])
POSITIVE_COUNT = Counter("positive_prediction_total", "Total prediksi positif")
NEGATIVE_COUNT = Counter("negative_prediction_total", "Total prediksi negatif")
NEUTRAL_COUNT = Counter("neutral_prediction_total", "Total prediksi netral")
ERROR_COUNT = Counter("error_count_total", "Total error")
TEXT_LENGTH = Histogram("input_text_length", "Panjang teks input")
ACTIVE_REQUESTS = Gauge("active_requests", "Jumlah request aktif")
MODEL_CONFIDENCE = Histogram("model_confidence", "Confidence score model")


@app.route("/predict", methods=["POST"])
def predict():
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    REQUEST_COUNT.inc()

    try:
        data = request.get_json()
        text = data.get("text", "")

        TEXT_LENGTH.observe(len(text))

        # preprocessing & prediksi
        text_tfidf = tfidf.transform([text])
        prediction = model.predict(text_tfidf)[0]
        proba = model.predict_proba(text_tfidf).max()

        MODEL_CONFIDENCE.observe(float(proba))
        PREDICTION_COUNT.labels(sentiment=prediction).inc()

        if prediction == "Positif":
            POSITIVE_COUNT.inc()
        elif prediction == "Negatif":
            NEGATIVE_COUNT.inc()
        else:
            NEUTRAL_COUNT.inc()

        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        ACTIVE_REQUESTS.dec()

        return jsonify(
            {"sentiment": prediction, "confidence": float(proba), "latency": latency})

    except Exception as e:
        ERROR_COUNT.inc()
        ACTIVE_REQUESTS.dec()
        return jsonify({"error": str(e)}), 500


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)
