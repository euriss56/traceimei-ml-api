"""
TraceIMEI-BJ — API ML v2.0
Modèle réel : Random Forest 70% + Isolation Forest 30%
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
CORS(app, origins=[
    "https://trace-benin-secure.vercel.app",
    "http://localhost:5173"
])

# ────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ────────────────────────────────────────────────────────────

MODEL = None
METRICS = {}

def load_model():
    global MODEL, METRICS
    try:
        MODEL = joblib.load('traceimei_model.pkl')
        print("✅ Modèle RF+IF chargé depuis traceimei_model.pkl")
    except Exception as e:
        print(f"⚠️  Modèle non trouvé : {e}")
        MODEL = None
    try:
        with open('model_metrics.json') as f:
            METRICS = json.load(f)
    except Exception:
        METRICS = {'auc_roc': 0.0, 'model_version': 'non chargé'}

load_model()

# ────────────────────────────────────────────────────────────
# BASE TAC (15 fabricants principaux au Bénin)
# ────────────────────────────────────────────────────────────

TAC_DB = {
    "35674108": ("Samsung",  "Galaxy Series"),
    "35328004": ("Apple",    "iPhone Series"),
    "35761904": ("Tecno",    "Spark Series"),
    "35856910": ("Itel",     "A Series"),
    "35231910": ("Infinix",  "Hot Series"),
    "35842910": ("Nokia",    "G Series"),
    "86751904": ("Huawei",   "Y Series"),
    "86498210": ("Xiaomi",   "Redmi Series"),
    "35986710": ("Oppo",     "A Series"),
    "35124510": ("Vivo",     "Y Series"),
    "35919004": ("Samsung",  "Galaxy A Series"),
    "01326300": ("Apple",    "iPhone 14"),
    "35445610": ("Tecno",    "Camon Series"),
    "35991610": ("Itel",     "Vision Series"),
    "86611102": ("Huawei",   "Nova Series"),
}

TEST_IMEIS = {
    "000000000000000",
    "123456789012345",
    "111111111111111",
    "999999999999999",
    "123456789000000",
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

def build_features(imei, extra=None):
    """
    Construit le vecteur de features pour l'inférence ML.
    Les features non observables (SIM swap, géoloc) utilisent
    des valeurs par défaut conservatrices.
    extra : dict optionnel avec des valeurs fournies par l'appelant
    """
    luhn  = 1 if luhn_check(imei) else 0
    _, _, tac_match = get_manufacturer(imei)
    tac_match_val = 1 if tac_match else 0

    # Valeurs par défaut (conservatrices — pas de données opérateur)
    sim_swap   = extra.get('sim_swap_frequency_30d', 1)   if extra else 1
    geoloc     = extra.get('geoloc_dispersion_km', 10)    if extra else 10
    repair     = extra.get('repair_history_count', 1)     if extra else 1
    net_pat    = extra.get('network_registration_pattern', 1) if extra else 1
    age_diff   = extra.get('imei_age_vs_model_age', 0.3)  if extra else 0.3
    photo      = extra.get('photo_model_mismatch_score', 0.0) if extra else 0.0

    return [[luhn, tac_match_val, sim_swap, geoloc,
             repair, net_pat, age_diff, photo]]

def compute_ml_score(imei, extra=None):
    """
    Calcule le score ensembliste RF 70% + IF 30%.
    Fallback sur règles métier si le modèle n'est pas chargé.
    """
    if MODEL is None:
        return compute_fallback_score(imei), "fallback"

    features = build_features(imei, extra)
    X = np.array(features)

    rf  = MODEL['rf']
    iso = MODEL['iso']
    iso_min = MODEL['iso_min']
    iso_max = MODEL['iso_max']

    rf_proba   = rf.predict_proba(X)[0][1]
    iso_raw    = -iso.decision_function(X)[0]
    iso_norm   = (iso_raw - iso_min) / (iso_max - iso_min)
    iso_norm   = float(np.clip(iso_norm, 0, 1))

    score = 0.70 * rf_proba + 0.30 * iso_norm
    return float(round(min(score, 0.99), 3)), "rf_iso_ensemble"

def compute_fallback_score(imei):
    """
    Score de secours basé sur règles métier (si modèle indisponible).
    """
    score = 0.0
    if not luhn_check(imei):         score += 0.40
    if len(imei) != 15:              score += 0.40
    if imei in TEST_IMEIS:           score += 0.50
    if imei and len(set(imei)) == 1: score += 0.35
    return round(min(score, 0.99), 3)

# ────────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "TraceIMEI-BJ ML API",
        "version": "2.0.0-RF",
        "model": "Random Forest 70% + Isolation Forest 30%",
        "status": "running",
        "model_loaded": MODEL is not None
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "model":         METRICS.get("model_version", "TraceIMEI-BJ v2.0-RF"),
        "model_loaded":  MODEL is not None,
        "auc_roc":       METRICS.get("auc_roc", 0.0),
        "auc_roc_terrain": METRICS.get("auc_roc_terrain"),
        "cv_mean":       METRICS.get("cv_mean"),
        "cv_std":        METRICS.get("cv_std"),
        "precision":     METRICS.get("precision"),
        "recall":        METRICS.get("recall"),
        "f1":            METRICS.get("f1"),
        "mode":          "rf_iso_ensemble" if MODEL else "fallback_rules",
        "timestamp":     time.time()
    })

@app.route("/api/check-imei", methods=["POST"])
def check_imei():
    start = time.time()
    data  = request.get_json()

    if not data or "imei" not in data:
        return jsonify({"error": "IMEI manquant"}), 400

    imei  = str(data.get("imei", "")).strip()
    extra = data.get("features", None)   # features optionnelles depuis le frontend

    luhn              = luhn_check(imei)
    manufacturer, series, tac_ok = get_manufacturer(imei)
    score, mode       = compute_ml_score(imei, extra)

    if score >= 0.80:
        status = "vole"
    elif score >= 0.50:
        status = "suspect"
    else:
        status = "legitime"

    elapsed = round((time.time() - start) * 1000, 2)

    return jsonify({
        "imei":            imei,
        "score":           score,
        "status":          status,
        "manufacturer":    manufacturer,
        "model_series":    series,
        "features": {
            "luhn_valid":                luhn,
            "imei_length_valid":         len(imei) == 15,
            "tac_code":                  imei[:8] if len(imei) >= 8 else "",
            "tac_manufacturer_match":    tac_ok,
            "all_same_digits":           len(set(imei)) == 1 if imei else False,
            "known_test_imei":           imei in TEST_IMEIS,
        },
        "auc_roc":         METRICS.get("auc_roc", 0.0),
        "scoring_mode":    mode,
        "response_time_ms": elapsed,
        "model_version":   METRICS.get("model_version", "TraceIMEI-BJ v2.0-RF")
    })

@app.route("/api/batch-check", methods=["POST"])
def batch_check():
    data = request.get_json()
    if not data or "imeis" not in data:
        return jsonify({"error": "Liste IMEI manquante"}), 400

    imeis   = data.get("imeis", [])[:50]
    results = []

    for imei in imeis:
        imei  = str(imei).strip()
        score, mode = compute_ml_score(imei)
        manufacturer, series, _ = get_manufacturer(imei)
        results.append({
            "imei":         imei,
            "score":        score,
            "status":       "vole"    if score >= 0.80 else
                            "suspect" if score >= 0.50 else "legitime",
            "manufacturer": manufacturer,
            "scoring_mode": mode,
        })

    return jsonify({
        "results":   results,
        "total":     len(results),
        "auc_roc":   METRICS.get("auc_roc", 0.0),
        "model_version": METRICS.get("model_version", "TraceIMEI-BJ v2.0-RF")
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
