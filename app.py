"""
TraceIMEI-BJ — API ML v2.2
Moteur : Random Forest 70% + Isolation Forest 30%
Correctifs v2.2 :
  - Features opérateur supprimées (sim_swap, geoloc, repair, net_pat, age_diff)
  - Features fiables uniquement : luhn, tac_match, all_same, is_test, length_ok
  - Base TAC élargie à 40+ fabricants présents au Bénin
  - CORS ouvert (origins=*)
Auteur : Euriss FANOU & Thierry MEHOUNOU — GETECH 2026
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import time
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ────────────────────────────────────────────────────────────

MODEL = None
METRICS = {}

def load_model():
    global MODEL, METRICS
    try:
        MODEL = joblib.load('traceimei_model.pkl')
        print("✅ Modèle RF+IF chargé")
    except Exception as e:
        print(f"⚠️ Modèle non chargé : {e}")
        MODEL = None
    try:
        with open('model_metrics.json') as f:
            METRICS = json.load(f)
    except Exception:
        METRICS = {
            "model_version": "TraceIMEI-BJ v2.2-RF",
            "data_origin": "synthetic"
        }

load_model()

# ────────────────────────────────────────────────────────────
# BASE TAC ÉLARGIE (40+ fabricants au Bénin)
# ────────────────────────────────────────────────────────────

TAC_DB = {
    # Samsung
    "35674108": ("Samsung",   "Galaxy Series"),
    "35919004": ("Samsung",   "Galaxy A Series"),
    "35821804": ("Samsung",   "Galaxy S Series"),
    "35355810": ("Samsung",   "Galaxy M Series"),
    "35284608": ("Samsung",   "Galaxy F Series"),

    # Apple
    "35328004": ("Apple",     "iPhone Series"),
    "01326300": ("Apple",     "iPhone 14"),
    "35299406": ("Apple",     "iPhone 13"),
    "35469208": ("Apple",     "iPhone 15"),
    "35607204": ("Apple",     "iPhone 12"),

    # Tecno
    "35761904": ("Tecno",     "Spark Series"),
    "35445610": ("Tecno",     "Camon Series"),
    "35990410": ("Tecno",     "Pop Series"),
    "35221710": ("Tecno",     "Phantom Series"),
    "35119710": ("Tecno",     "Pova Series"),

    # Itel
    "35856910": ("Itel",      "A Series"),
    "35991610": ("Itel",      "Vision Series"),
    "35120310": ("Itel",      "P Series"),
    "35284510": ("Itel",      "S Series"),

    # Infinix
    "35231910": ("Infinix",   "Hot Series"),
    "35784510": ("Infinix",   "Note Series"),
    "35990610": ("Infinix",   "Zero Series"),
    "35221810": ("Infinix",   "Smart Series"),
    "35119610": ("Infinix",   "GT Series"),

    # Nokia
    "35842910": ("Nokia",     "G Series"),
    "35284710": ("Nokia",     "C Series"),
    "35119910": ("Nokia",     "X Series"),
    "35990910": ("Nokia",     "T Series"),

    # Huawei
    "86751904": ("Huawei",    "Y Series"),
    "86611102": ("Huawei",    "Nova Series"),
    "86498904": ("Huawei",    "P Series"),
    "86732204": ("Huawei",    "Mate Series"),
    "86521604": ("Huawei",    "Honor Series"),

    # Xiaomi
    "86498210": ("Xiaomi",    "Redmi Series"),
    "86739210": ("Xiaomi",    "Mi Series"),
    "86521910": ("Xiaomi",    "Poco Series"),
    "35284810": ("Xiaomi",    "Redmi Note Series"),
    "86739510": ("Xiaomi",    "12 Series"),

    # Oppo
    "35986710": ("Oppo",      "A Series"),
    "86738904": ("Oppo",      "Reno Series"),
    "86521810": ("Oppo",      "Find Series"),
    "35990110": ("Oppo",      "F Series"),

    # Vivo
    "35124510": ("Vivo",      "Y Series"),
    "86739110": ("Vivo",      "V Series"),
    "86521710": ("Vivo",      "X Series"),
    "35990210": ("Vivo",      "T Series"),

    # Realme
    "35284910": ("Realme",    "C Series"),
    "86739310": ("Realme",    "GT Series"),
    "35990710": ("Realme",    "Narzo Series"),
    "35119510": ("Realme",    "10 Series"),

    # Motorola
    "35354410": ("Motorola",  "Moto G Series"),
    "35119810": ("Motorola",  "Moto E Series"),
    "35990810": ("Motorola",  "Edge Series"),

    # Zte / Wiko / autres Bénin
    "86738710": ("ZTE",       "Blade Series"),
    "35990310": ("Wiko",      "T Series"),
    "35119410": ("Wiko",      "Y Series"),
}

TEST_IMEIS = {
    "000000000000000",
    "111111111111111",
    "123456789012345",
    "999999999999999",
}

# ────────────────────────────────────────────────────────────
# UTILITAIRES
# ────────────────────────────────────────────────────────────

def luhn_check(imei):
    if len(imei) != 15 or not imei.isdigit():
        return False
    digits = [int(d) for d in imei]
    odd_digits  = digits[-1::-2]
    even_digits = digits[-2::-2]
    total = sum(odd_digits)
    for d in even_digits:
        total += sum(divmod(d * 2, 10))
    return total % 10 == 0


def get_manufacturer(imei):
    tac = imei[:8] if len(imei) >= 8 else ""
    for prefix, (brand, series) in TAC_DB.items():
        if tac.startswith(prefix[:6]):
            return brand, series, True
    return "Inconnu", "Modèle inconnu", False


def build_features(imei):
    """
    Features v2.2 — uniquement features vérifiables sans données opérateur.
    5 features : luhn, tac_match, all_same_digits, is_test_imei, length_ok
    """
    luhn      = 1 if luhn_check(imei) else 0
    _, _, tac = get_manufacturer(imei)
    tac_match = 1 if tac else 0
    all_same  = 1 if imei and len(set(imei)) == 1 else 0
    is_test   = 1 if imei in TEST_IMEIS else 0
    length_ok = 1 if len(imei) == 15 else 0

    return [[luhn, tac_match, all_same, is_test, length_ok]]


def compute_fallback_score(imei):
    score = 0.0
    if not luhn_check(imei):         score += 0.40
    if len(imei) != 15:              score += 0.40
    if imei in TEST_IMEIS:           score += 0.50
    if imei and len(set(imei)) == 1: score += 0.35
    return round(min(score, 0.99), 3)


def compute_ml_score(imei):
    # Blacklist pré-inférence
    if imei in TEST_IMEIS:
        return 0.99, "blacklist", "high"
    if imei and len(imei) == 15 and len(set(imei)) == 1:
        return 0.99, "blacklist", "high"

    if MODEL is None:
        return compute_fallback_score(imei), "fallback_rules", "low"

    X = np.array(build_features(imei))

    rf      = MODEL['rf']
    iso     = MODEL['iso']
    iso_min = MODEL['iso_min']
    iso_max = MODEL['iso_max']

    rf_proba = rf.predict_proba(X)[0][1]
    iso_raw  = -iso.decision_function(X)[0]
    iso_norm = float(np.clip(
        (iso_raw - iso_min) / (iso_max - iso_min), 0, 1
    ))

    score = 0.70 * rf_proba + 0.30 * iso_norm
    return float(round(min(score, 0.99), 3)), "scoring_v2", "high"

# ────────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name":         "TraceIMEI-BJ API",
        "version":      "2.2.0",
        "engine":       "Random Forest 70% + Isolation Forest 30%",
        "features":     "luhn, tac_match, all_same, is_test, length_ok",
        "status":       "running",
        "model_loaded": MODEL is not None
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": MODEL is not None,
        "model_status": "calibration_terrain_en_cours",
        "scoring_mode": "scoring_v2" if MODEL else "fallback_rules",
        "version":      "2.2.0",
        "timestamp":    time.time()
    })


@app.route("/api/check-imei", methods=["POST"])
def check_imei():
    start = time.time()
    data  = request.get_json()

    if not data or "imei" not in data:
        return jsonify({"error": "IMEI manquant"}), 400

    imei = str(data.get("imei", "")).strip()

    luhn                         = luhn_check(imei)
    manufacturer, series, tac_ok = get_manufacturer(imei)
    score, mode, confidence      = compute_ml_score(imei)

    if score >= 0.80:   status = "vole"
    elif score >= 0.50: status = "suspect"
    else:               status = "legitime"

    elapsed = round((time.time() - start) * 1000, 2)

    return jsonify({
        "imei":         imei,
        "score":        score,
        "status":       status,
        "manufacturer": manufacturer,
        "model_series": series,

        "features": {
            "luhn_valid":             luhn,
            "imei_length_valid":      len(imei) == 15,
            "tac_code":               imei[:8] if len(imei) >= 8 else "",
            "tac_manufacturer_match": tac_ok,
            "all_same_digits":        len(set(imei)) == 1 if imei else False,
            "known_test_imei":        imei in TEST_IMEIS,
        },

        "scoring_mode":      mode,
        "confidence":        confidence,
        "model_status":      "calibration_terrain_en_cours",
        "response_time_ms":  elapsed,
        "model_version":     METRICS.get("model_version", "TraceIMEI-BJ v2.2-RF")
    })


@app.route("/api/batch-check", methods=["POST"])
def batch_check():
    data = request.get_json()
    if not data or "imeis" not in data:
        return jsonify({"error": "Liste IMEI manquante"}), 400

    imeis   = data.get("imeis", [])[:50]
    results = []

    for imei in imeis:
        imei = str(imei).strip()
        score, mode, confidence      = compute_ml_score(imei)
        manufacturer, series, tac_ok = get_manufacturer(imei)
        results.append({
            "imei":         imei,
            "score":        score,
            "status":       "vole" if score >= 0.80 else
                            "suspect" if score >= 0.50 else "legitime",
            "manufacturer": manufacturer,
            "tac_match":    tac_ok,
            "scoring_mode": mode,
            "confidence":   confidence,
        })

    return jsonify({
        "results":       results,
        "total":         len(results),
        "model_status":  "calibration_terrain_en_cours",
        "model_version": METRICS.get("model_version", "TraceIMEI-BJ v2.2-RF")
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
