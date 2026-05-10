import requests
import time
import random

# url inference service
INFERENCE_URL = "http://localhost:5002/predict"

# sample teks untuk testing
sample_texts = [
    "produk bagus banget, pengiriman cepat",
    "kecewa dengan kualitas produk ini",
    "biasa saja tidak terlalu bagus",
    "sangat puas dengan pembelian ini",
    "barang tidak sesuai deskripsi",
    "recommended seller, produk original",
    "pengiriman lama tapi produk oke",
    "mantap jiwa, beli lagi deh",
    "jelek banget, tidak sesuai ekspektasi",
    "cukup bagus untuk harganya"]

def send_requests():
    """mengirim request ke inference service secara periodik"""
    while True:
        try:
            text = random.choice(sample_texts)
            response = requests.post(
                INFERENCE_URL,
                json={"text": text},
                timeout=5)
                
            if response.status_code == 200:
                result = response.json()
                print(f"Text: {text[:30]}... | Sentiment: {result['sentiment']} | Confidence: {result['confidence']:.3f}")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(2)

if __name__ == '__main__':
    print("prometheus exporter berjalan...")
    print("mengirim request ke inference service...")
    send_requests()