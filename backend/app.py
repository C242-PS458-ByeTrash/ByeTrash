from flask import Flask, jsonify, request
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
import os
import numpy as np
import requests
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Tambahkan CORS untuk mengizinkan akses lintas sumber
CORS(app)

# Konfigurasi Swagger UI
SWAGGER_URL = "/swagger"  # URL endpoint untuk Swagger UI
API_URL = "/static/swagger.yaml"  # Lokasi file OpenAPI Specification

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={"app_name": "Recycling Prediction API"}
)

# Daftarkan blueprint Swagger UI
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# URLs Model di Google Cloud Storage
MODEL_URL = "https://storage.googleapis.com/model-machine-learning/model_weights.weights.h5"
JSON_URL = "https://storage.googleapis.com/model-machine-learning/model.json"

# Unduh model jika belum ada secara lokal
LOCAL_MODEL_PATH = "model_weights.weights.h5"
LOCAL_JSON_PATH = "model.json"

if not os.path.exists(LOCAL_JSON_PATH):
    print("Downloading model JSON...")
    response = requests.get(JSON_URL)
    with open(LOCAL_JSON_PATH, "wb") as f:
        f.write(response.content)

if not os.path.exists(LOCAL_MODEL_PATH):
    print("Downloading model weights...")
    response = requests.get(MODEL_URL)
    with open(LOCAL_MODEL_PATH, "wb") as f:
        f.write(response.content)

# Load model
with open(LOCAL_JSON_PATH, "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights(LOCAL_MODEL_PATH)

# Labels (ubah sesuai kategori Anda)
LABELS = ["non-organik", "berbahaya", "organik"]

# Preprocessing function
def preprocess_image(image_path, target_size=(128, 128)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)
    return image

# Endpoint contoh
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Recycling Prediction API!"})


# Endpoint untuk predict
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Simpan file sementara
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Prediksi menggunakan model
    try:
        image = preprocess_image(file_path)
        prediction = model.predict(image)
        predicted_label = LABELS[np.argmax(prediction)]
        confidence = float(np.max(prediction))  # Ambil confidence score tertinggi
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Hapus file setelah diproses
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({"prediction": predicted_label, "confidence": confidence})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
