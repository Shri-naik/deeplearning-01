from http.server import BaseHTTPRequestHandler
import os
import io
import json
import urllib.parse
import cgi
import numpy as np
from PIL import Image

# Lazy imports to keep cold-start smaller until needed
tf = None

CLASS_NAMES = ["Clean", "Little Polluted", "Highly Polluted"]

def ensure_tf():
    global tf
    if tf is None:
        import tensorflow as _tf
        tf = _tf

def preprocess_image(image, img_size=(224, 224)):
    ensure_tf()
    img = image.resize(img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def analyze_water_rgb(image):
    img = np.array(image)
    avg_rgb = np.mean(img, axis=(0, 1))
    r, g, b = avg_rgb
    analysis = []
    if g > r and g > b and g > 100:
        analysis.append("High green: Possible algae growth.")
    if r > g and r > b and r > 100:
        analysis.append("High red/brown: Possible iron or suspended solids.")
    if b > r and b > g and b > 120:
        analysis.append("Strong blue: Likely clean water.")
    if np.mean(avg_rgb) < 60:
        analysis.append("Very dark: Possible industrial waste / oil contamination.")
    if not analysis:
        analysis.append("No major harmful constituents detected visually.")
    return avg_rgb.tolist(), analysis

def analyze_air_rgb(image):
    img = np.array(image)
    avg_rgb = np.mean(img, axis=(0, 1))
    r, g, b = avg_rgb
    analysis = []
    if np.mean(avg_rgb) < 80:
        analysis.append("Dark/gray tones: Possible smog or soot.")
    if r > 120 and r > g and r > b:
        analysis.append("Brown haze: Possible dust or industrial emissions.")
    if g > 120 and g > r and g > b:
        analysis.append("Greenish tint: Could indicate chemical pollutants.")
    if b > 130 and b > r and b > g:
        analysis.append("Clear sky with strong blue: Clean air likely.")
    if not analysis:
        analysis.append("No major harmful air constituents detected visually.")
    return avg_rgb.tolist(), analysis

def get_model_path(pollution_type: str) -> str:
    ensure_tf()
    # Try local first (ignored by .gitignore for repo cleanliness)
    local_map = {
        "water": os.path.join("/tmp", "water_best_model.h5"),
        "air": os.path.join("/tmp", "air_best_model.h5"),
    }
    path = local_map.get(pollution_type)
    if path and os.path.exists(path):
        return path

    # If not present locally, download from env-configured URLs
    url = os.getenv("WATER_MODEL_URL") if pollution_type == "water" else os.getenv("AIR_MODEL_URL")
    if not url:
        return None
    try:
        import requests
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    except Exception:
        return None

def predict_image(image: Image.Image, pollution_type: str):
    # Try ML model; if unavailable, use heuristic fallback so Vercel deploys still work
    model_path = None
    pred = None
    try:
        ensure_tf()
        model_path = get_model_path(pollution_type)
        if model_path:
            model = tf.keras.models.load_model(model_path)
            img_array = preprocess_image(image)
            pred = model.predict(img_array)
    except Exception:
        pred = None

    # Compute RGB-based analysis regardless (used by both paths)
    if pollution_type == "water":
        avg_rgb, analysis = analyze_water_rgb(image)
    else:
        avg_rgb, analysis = analyze_air_rgb(image)

    if pred is not None:
        confidence = float(np.max(pred))
        label_index = int(np.argmax(pred))
        prediction = CLASS_NAMES[label_index]
        probs = {CLASS_NAMES[i]: float(pred[0][i] * 100.0) for i in range(len(CLASS_NAMES))}
        class_name = "Low" if label_index == 0 else ("Medium" if label_index == 1 else "High")
    else:
        # Heuristic fallback when model is missing or fails
        r, g, b = avg_rgb
        mean_val = float(np.mean(avg_rgb))
        if pollution_type == "water":
            if b > 130 and b > r and b > g and mean_val > 120:
                class_name = "Low"; prediction = "Clean"
            elif g > 120 and g > r and g > b:
                class_name = "Medium"; prediction = "Little Polluted"
            elif mean_val < 70 or r > 120:
                class_name = "High"; prediction = "Highly Polluted"
            else:
                class_name = "Medium"; prediction = "Little Polluted"
        else:
            if b > 140 and b > r and b > g:
                class_name = "Low"; prediction = "Clean"
            elif mean_val < 80 or r > 130:
                class_name = "High"; prediction = "Highly Polluted"
            else:
                class_name = "Medium"; prediction = "Little Polluted"
        # Construct rough probabilities
        probs = {
            "Clean": 60.0 if class_name == "Low" else 20.0,
            "Little Polluted": 60.0 if class_name == "Medium" else 20.0,
            "Highly Polluted": 60.0 if class_name == "High" else 20.0,
        }
        confidence = max(probs.values()) / 100.0

    return {
        "prediction": prediction,
        "confidence": confidence,
        "class": class_name,
        "probs": probs,
        "analysis": analysis,
        "avg_rgb": avg_rgb,
        "model_used": bool(pred is not None),
    }


class handler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)
        self.wfile.write(b"{}")

    def do_POST(self):
        # Parse query params for type
        parsed = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(parsed.query)
        pollution_type = (q.get("type", ["water"])[0] or "water").lower()

        # Parse multipart form
        ctype, pdict = cgi.parse_header(self.headers.get("content-type"))
        if ctype != "multipart/form-data":
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Expected multipart/form-data"}).encode("utf-8"))
            return

        pdict["boundary"] = bytes(pdict["boundary"], "utf-8")
        pdict["CONTENT-LENGTH"] = int(self.headers.get("content-length"))
        try:
            fields = cgi.parse_multipart(self.rfile, pdict)
        except Exception as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": f"Malformed form-data: {e}"}).encode("utf-8"))
            return

        files = fields.get("file")
        if not files:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "No file uploaded"}).encode("utf-8"))
            return

        try:
            file_bytes = files[0]
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": f"Invalid image: {e}"}).encode("utf-8"))
            return

        result = predict_image(image, pollution_type)
        status = 200 if "error" not in result else 500
        self._set_headers(status)
        self.wfile.write(json.dumps(result).encode("utf-8"))