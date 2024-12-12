from flask import Flask, jsonify, request
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import requests

app = Flask(__name__)
CORS(app)

# Konfigurasi Swagger UI
SWAGGER_URL = "/swagger"
API_URL = "/static/swagger.yaml"

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={"app_name": "ByeTrash Prediction API"}
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# URL Model dan path lokal
MODEL_URL = "https://storage.googleapis.com/model-machine-learning/model_kategorisampah.h5"
LOCAL_MODEL_PATH = "model_kategorisampah.h5"

# Unduh model jika belum ada
if not os.path.exists(LOCAL_MODEL_PATH):
    print("Downloading model...")
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        with open(LOCAL_MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}")
        # Tangani error download, misalnya, keluar atau gunakan model default

# Muat model
try:
    model = keras.models.load_model(LOCAL_MODEL_PATH)
except OSError as e:
    print(f"Error loading model: {e}")
    # Tangani error loading model


# Label Kelas
class_names = ['berbahaya', 'non-organik', 'organic']

def preprocess_image(image, target_size=(224, 224)):  # Ukuran input MobileNetV2
    image = image.resize(target_size)
    image = img_to_array(image)
    image = preprocess_input(image)  # Preprocessing MobileNetV2
    image = np.expand_dims(image, axis=0)
    return image

# Endpoint contoh
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the ByeTrash Prediction API!"})

# Endpoint prediksi
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image = load_img(BytesIO(file.read()), target_size=(224, 224)) # Ukuran input MobileNetV2
        image = preprocess_image(image)
        prediction = model.predict(image)

        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        confidence = prediction[0][predicted_class_index]

        result = {
            "prediction": predicted_class,
            "confidence": float(confidence)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
