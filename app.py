"""
TraceIMEI-BJ — API ML Flask
Auteur : Euriss FANOU — GETECH Cotonou 2026
Algorithmes : Random Forest (70%) + Isolation Forest (30%)
"""

import os
import re
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Autoriser les requêtes depuis Lovable et n'importe quel frontend
CORS(app, origins="*")

# ─── CLÉ API (définie dans les variables d'environnement Render) ──────────────
API_KEY = os.environ.get("TRACEIMEI_API_KEY", "traceimei-dev-key-2026")


def require_api_key(f):
    """Décorateur : vérifie la clé API dans le header Authorization."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        key  = auth.replace("Bearer ", "").strip()
        if key != API_KEY:
            return jsonify({"error": "Clé API invalide ou manquante"}), 401
        return f(*args, **kwargs)
    return decorated


# ─── VALIDATION LUHN ──────────────────────────────────────────────────────────
def luhn_valid(imei: str) -> bool:
    """Vérifie le chiffre de contrôle Luhn d'un IMEI."""
    if not re.fullmatch(r"\d{15}", imei):
        return False
    digits = [int(d) for d in imei]
    for i in range(len(digits) - 2, -1, -2):
        digits[i] *= 2
        if digits[i] > 9:
            digits[i] -= 9
    return sum(digits) % 10 == 0


# ─── EXTRACTION DES FEATURES ──────────────────────────────────────────────────
def extract_features(data: dict) -> np.ndarray:
    """
    Extrait les 8 features du modèle depuis le payload JSON.
    Valeurs par défaut sûres si une feature est absente.
    """
    imei = str(data.get("imei", "000000000000000"))

    f1  = 1.0 if luhn_valid(imei) else 0.0                          # imei_luhn_valid
    f2  = float(data.get("tac_manufacturer_match", 1))              # tac_manufacturer_match
    f3  = float(data.get("sim_swap_frequency_30d", 0))              # sim_swap_frequency_30d
    f4  = float(data.get("geoloc_dispersion_km", 0))                # geoloc_dispersion_km
    f5  = float(data.get("repair_history_count", 0))                # repair_history_count
    f6  = float(data.get("network_registration_pattern", 0))        # network_registration_pattern
    f7  = float(data.get("imei_age_vs_model_age", 0))               # imei_age_vs_model_age
    f8  = float(data.get("photo_model_mismatch_score", 0.0))        # photo_model_mismatch_score

    return np.array([[f1, f2, f3, f4, f5, f6, f7, f8]])


# ─── MODÈLE ML (chargé une seule fois au démarrage) ──────────────────────────
# On essaie de charger le vrai modèle entraîné (traceimei_model.pkl).
# Si absent, on utilise un modèle de démonstration basé sur des règles claires.

try:
    import joblib
    MODEL = joblib.load("traceimei_model.pkl")
    MODEL_TYPE = "trained"
    print("✅ Modèle entraîné chargé : traceimei_model.pkl")

except (FileNotFoundError, Exception) as e:
    MODEL = None
    MODEL_TYPE = "rules"
    print(f"⚠️  Modèle pkl absent ({e}). Modèle de règles actif.")


def predict_score(features: np.ndarray, imei: str) -> dict:
    """
    Retourne le score d'anomalie ensembliste (RF 70% + ISO 30%).
    Valeur entre 0.0 (légitime) et 1.0 (très suspect).
    """
    f = features[0]

    if MODEL_TYPE == "trained":
        rf  = MODEL["rf"]
        iso = MODEL["iso"]

        rf_score  = rf.predict_proba(features)[0][1]            # prob classe "cloné"
        iso_raw   = -iso.decision_function(features)[0]         # score anomalie brut
        iso_min, iso_max = -0.5, 0.5                            # plage calibrée
        iso_score = np.clip((iso_raw - iso_min) / (iso_max - iso_min), 0, 1)

        ensemble_score = 0.7 * rf_score + 0.3 * iso_score

    else:
        # ── Modèle de règles (démonstration sans pkl) ──────────────────────
        score = 0.0

        # Luhn invalide → signal fort
        if f[0] == 0.0:
            score += 0.45

        # TAC incohérent avec fabricant déclaré
        if f[1] == 0.0:
            score += 0.25

        # Trop de swaps SIM (> 3 en 30 jours = suspect)
        if f[2] > 3:
            score += min((f[2] - 3) * 0.08, 0.20)

        # IMEI utilisé dans des zones très distantes (> 200 km)
        if f[3] > 200:
            score += min((f[3] - 200) / 1000, 0.15)

        # Photo ne correspond pas au modèle déclaré
        if f[7] > 0.6:
            score += f[7] * 0.20

        # Appareil très ancien sans aucun historique atelier
        if f[6] > 3 and f[4] == 0:
            score += 0.10

        ensemble_score = min(score, 1.0)

    # ── Statut final ───────────────────────────────────────────────────────────
    if ensemble_score < 0.35:
        status = "LEGITIME"
        color  = "green"
    elif ensemble_score < 0.65:
        status = "SUSPECT"
        color  = "orange"
    else:
        status = "CLONE_DETECTE"
        color  = "red"

    return {
        "ensemble_score": round(float(ensemble_score), 4),
        "rf_contribution": round(float(0.7 * ensemble_score), 4),
        "iso_contribution": round(float(0.3 * ensemble_score), 4),
        "status": status,
        "color": color,
        "luhn_valid": bool(f[0] == 1.0),
        "model_type": MODEL_TYPE,
    }


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "TraceIMEI-BJ ML API",
        "version": "1.0.0",
        "auteur": "Euriss FANOU — GETECH Cotonou 2026",
        "status": "running",
        "model": MODEL_TYPE,
        "endpoints": {
            "POST /predict": "Analyser un IMEI",
            "POST /predict/batch": "Analyser plusieurs IMEI (max 50)",
            "GET /health": "Vérifier l'état de l'API",
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_TYPE}), 200


@app.route("/predict", methods=["POST"])
@require_api_key
def predict():
    """
    Analyser un IMEI.

    Body JSON attendu :
    {
        "imei": "358765043518671",
        "tac_manufacturer_match": 1,        // optionnel
        "sim_swap_frequency_30d": 2,         // optionnel
        "geoloc_dispersion_km": 50,          // optionnel
        "repair_history_count": 1,           // optionnel
        "network_registration_pattern": 0,   // optionnel
        "imei_age_vs_model_age": 1,          // optionnel
        "photo_model_mismatch_score": 0.1    // optionnel
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Body JSON manquant"}), 400

    imei = str(data.get("imei", "")).strip()
    if not imei:
        return jsonify({"error": "Champ 'imei' requis"}), 400

    if not re.fullmatch(r"\d{15}", imei):
        return jsonify({
            "error": "IMEI invalide — doit contenir exactement 15 chiffres",
            "imei": imei
        }), 422

    features = extract_features(data)
    result   = predict_score(features, imei)

    return jsonify({
        "imei": imei,
        "tac": imei[:8],
        **result,
        "message": _message(result["status"]),
    }), 200


@app.route("/predict/batch", methods=["POST"])
@require_api_key
def predict_batch():
    """
    Analyser jusqu'à 50 IMEI en une seule requête.

    Body JSON attendu :
    {
        "items": [
            {"imei": "358765043518671"},
            {"imei": "490154203237518", "sim_swap_frequency_30d": 5}
        ]
    }
    """
    data = request.get_json(silent=True)
    if not data or "items" not in data:
        return jsonify({"error": "Champ 'items' requis (liste d'IMEI)"}), 400

    items = data["items"]
    if len(items) > 50:
        return jsonify({"error": "Maximum 50 IMEI par requête batch"}), 400

    results = []
    for item in items:
        imei = str(item.get("imei", "")).strip()
        if not re.fullmatch(r"\d{15}", imei):
            results.append({"imei": imei, "error": "IMEI invalide"})
            continue
        features = extract_features(item)
        result   = predict_score(features, imei)
        results.append({"imei": imei, **result, "message": _message(result["status"])})

    return jsonify({"count": len(results), "results": results}), 200


def _message(status: str) -> str:
    messages = {
        "LEGITIME":      "Cet appareil ne présente aucune anomalie détectée.",
        "SUSPECT":       "Des anomalies comportementales ont été détectées. Vérification approfondie recommandée.",
        "CLONE_DETECTE": "Clonage IMEI très probable. Ne pas acquérir cet appareil.",
    }
    return messages.get(status, "")


# ─── DÉMARRAGE ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
