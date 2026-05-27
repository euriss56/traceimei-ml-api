"""
TraceIMEI-BJ — API ML v2.1 (honest build)
Moteur : Random Forest 70% + Isolation Forest 30% (entraîné sur données synthétiques)
Scoring exposé comme "scoring_v1" jusqu'à validation terrain.

Correctifs v2.1-honest :
  - AUC retiré des réponses publiques (données synthétiques, non fiable en prod)
  - scoring_mode renommé en "scoring_v1" (terminologie neutre)
  - data_completeness ajouté (full / partial) pour transparence
  - confidence exposé : low si features opérateur absentes
  - /api/health expose model_status: "calibration_terrain_en_cours"
  - photo_model_mismatch_score retirée (importance=0%)
  - Valeurs fallback prudentes (sim_swap=3, geoloc=30km, repair=3)
  - sklearn==1.6.1 épinglé dans requirements.txt

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
        METRICS = {'model_version': 'non chargé', 'data_origin': 'unknown'}

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
    Construit le vecteur de features pour l'inférence ML (7 features, v2.1).
    photo_model_mismatch_score retirée (importance=0%).
    Valeurs par défaut prudentes — absence de données opérateur = profil plus risqué.
    """
    luhn      = 1 if luhn_check(imei) else 0
    _, _, tac_match = get_manufacturer(imei)
    tac_match_val = 1 if tac_match else 0

    sim_swap  = extra.get('sim_swap_frequency_30d', 3)          if extra else 3
    geoloc    = extra.get('geoloc_dispersion_km', 30)           if extra else 30
    repair    = extra.get('repair_history_count', 3)            if extra else 3
    net_pat   = extra.get('network_registration_pattern', 0.5)  if extra else 0.5
    age_diff  = extra.get('imei_age_vs_model_age', 0.3)         if extra else 0.3

    return [[luhn, tac_match_val, sim_swap, geoloc,
             repair, net_pat, age_diff]]

def compute_ml_score(imei, extra=None):
    """
    Calcule le score ensembliste RF 70% + IF 30%.
    Blacklist pré-inférence pour les IMEI de test connus.
    Fallback sur règles métier si le modèle n'est pas chargé.
    Retourne : (score, mode, confidence, data_completeness)
    """
    # ── Blacklist pré-inférence ──────────────────────────────
    if imei in TEST_IMEIS:
        return 0.99, "blacklist", "high", "full"
    if imei and len(imei) == 15 and len(set(imei)) == 1:
        return 0.99, "blacklist", "high", "full"

    data_completeness = "full" if extra is not None else "partial"
    confidence        = "medium" if extra is not None else "low"

    if MODEL is None:
        return compute_fallback_score(imei), "fallback_rules", "low", data_completeness

    features = build_features(imei, extra)
    X = np.array(features)

    rf      = MODEL['rf']
    iso     = MODEL['iso']
    iso_min = MODEL['iso_min']
    iso_max = MODEL['iso_max']

    rf_proba  = rf.predict_proba(X)[0][1]
    iso_raw   = -iso.decision_function(X)[0]
    iso_norm  = (iso_raw - iso_min) / (iso_max - iso_min)
    iso_norm  = float(np.clip(iso_norm, 0, 1))

    score = 0.70 * rf_proba + 0.30 * iso_norm
    return float(round(min(score, 0.99), 3)), "scoring_v1", confidence, data_completeness

def compute_fallback_score(imei):
    """
    Score de secours basé sur règles métier (si modèle indisponible).
    """
    score = 0.0
    if not luhn_check(imei):          score += 0.40
    if len(imei) != 15:               score += 0.40
    if imei in TEST_IMEIS:            score += 0.50
    if imei and len(set(imei)) == 1:  score += 0.35
    return round(min(score, 0.99), 3)

# ────────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name":         "TraceIMEI-BJ API",
        "version":      "2.1.0",
        "engine":       "Random Forest 70% + Isolation Forest 30%",
        "data_origin":  "synthetic — terrain validation pending",
        "status":       "running",
        "model_loaded": MODEL is not None
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":               "ok",
        "model":                METRICS.get("model_version", "TraceIMEI-BJ v2.1-RF"),
        "model_loaded":         MODEL is not None,
        "model_status":         "calibration_terrain_en_cours",
        "data_origin":          METRICS.get("data_origin", "synthetic"),
        "scoring_mode":         "scoring_v1" if MODEL else "fallback_rules",

        # Métriques synthétiques — clairement labellisées
        "metrics_synthetic": {
            "auc_roc":   METRICS.get("metrics_synthetic", {}).get("auc_roc"),
            "cv_mean":   METRICS.get("metrics_synthetic", {}).get("cv_mean"),
            "cv_std":    METRICS.get("metrics_synthetic", {}).get("cv_std"),
            "precision": METRICS.get("metrics_synthetic", {}).get("precision"),
            "recall":    METRICS.get("metrics_synthetic", {}).get("recall"),
            "f1":        METRICS.get("metrics_synthetic", {}).get("f1"),
        },

        # Métriques terrain — null jusqu'à pilote réel
        "metrics_terrain": METRICS.get("metrics_terrain", {
            "auc_roc": None,
            "status":  "calibration_terrain_en_cours"
        }),

        "timestamp": time.time()
    })

@app.route("/api/check-imei", methods=["POST"])
def check_imei():
    start = time.time()
    data  = request.get_json()

    if not data or "imei" not in data:
        return jsonify({"error": "IMEI manquant"}), 400

    imei  = str(data.get("imei", "")).strip()
    extra = data.get("features", None)

    luhn                        = luhn_check(imei)
    manufacturer, series, tac_ok = get_manufacturer(imei)
    score, mode, confidence, data_completeness = compute_ml_score(imei, extra)

    if score >= 0.80:
        status = "vole"
    elif score >= 0.50:
        status = "suspect"
    else:
        status = "legitime"

    elapsed = round((time.time() - start) * 1000, 2)

    return jsonify({
        "imei":             imei,
        "score":            score,
        "status":           status,
        "manufacturer":     manufacturer,
        "model_series":     series,

        "features": {
            "luhn_valid":               luhn,
            "imei_length_valid":        len(imei) == 15,
            "tac_code":                 imei[:8] if len(imei) >= 8 else "",
            "tac_manufacturer_match":   tac_ok,
            "all_same_digits":          len(set(imei)) == 1 if imei else False,
            "known_test_imei":          imei in TEST_IMEIS,
        },

        # Transparence sur la qualité du scoring
        "scoring_mode":      mode,
        "confidence":        confidence,
        "data_completeness": data_completeness,
        "model_status":      "calibration_terrain_en_cours",

        "response_time_ms":  elapsed,
        "model_version":     METRICS.get("model_version", "TraceIMEI-BJ v2.1-RF")
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
        score, mode, confidence, data_completeness = compute_ml_score(imei)
        manufacturer, series, _ = get_manufacturer(imei)
        results.append({
            "imei":              imei,
            "score":             score,
            "status":            "vole"    if score >= 0.80 else
                                 "suspect" if score >= 0.50 else "legitime",
            "manufacturer":      manufacturer,
            "scoring_mode":      mode,
            "confidence":        confidence,
            "data_completeness": data_completeness,
        })

    return jsonify({
        "results":      results,
        "total":        len(results),
        "model_status": "calibration_terrain_en_cours",
        "model_version": METRICS.get("model_version", "TraceIMEI-BJ v2.1-RF")
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
