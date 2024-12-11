from flask import Flask, jsonify, request
from flask_swagger_ui import get_swaggerui_blueprint
import os

app = Flask(__name__)

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

    # Dummy response (ganti ini dengan model prediksi Anda)
    try:
        # Di sini Anda dapat memanggil fungsi model prediksi
        predicted_label = "organik"  # Contoh label
        confidence = 0.95  # Contoh kepercayaan
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Hapus file setelah diproses
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({"prediction": predicted_label, "confidence": confidence})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
