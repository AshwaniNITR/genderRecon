from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import numpy as np
import cv2
import traceback
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def classify_gender(image_bytes, min_confidence=60):
    # Decode image from memory
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image data")

    result = DeepFace.analyze(
        img,
        actions=["gender"],
        detector_backend="mtcnn",
        enforce_detection=True,
        align=False
    )

    # Extract gender probabilities
    if isinstance(result, list):
        gender_scores = result[0]["gender"]
    else:
        gender_scores = result["gender"]
    
    predicted_gender = max(gender_scores, key=gender_scores.get)
    confidence = float(gender_scores[predicted_gender])


    # Clean up
    del img
    del np_img

    # Confidence check
    if confidence < min_confidence:
        raise ValueError(f"Low confidence ({confidence:.2f}%), please retry")

    return predicted_gender, confidence

@app.route("/predict_gender", methods=["POST"])
def predict_gender():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    try:
        gender, confidence = classify_gender(image_bytes)
        return jsonify({
            "gender": gender,
            "confidence": confidence
        })
    except ValueError as ve:
        print("❌ ValueError:", ve)
        traceback.print_exc()
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print("🔥 UNEXPECTED ERROR:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False,threaded=True)
