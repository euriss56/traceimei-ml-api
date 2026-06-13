"""
TraceIMEI-BJ — API ML v3.2
Logique métier corrigée :
  - VOLÉ     → is_declared_stolen=True envoyé par le frontend (source : Supabase)
  - SUSPECT  → ML détecte anomalie comportementale (score >= 0.50)
  - LÉGITIME → Luhn valide + TAC connu + comportement normal

Le ML ne peut PAS décider seul qu'un IMEI est VOLÉ.
Le frontend consulte Supabase, puis passe is_declared_stolen à l'API.

Auteur : Euriss FANOU & Thierry MEHOUNOU — GETECH 2026
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import time
import os
import random
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ────────────────────────────────────────────────────────────
# CONSTANTES
# ────────────────────────────────────────────────────────────

KNOWN_TAC = [
    "35674108","35919004","35821804","35355810","35284608",
    "35328004","01326300","35299406","35469208","35607204",
    "35761904","35445610","35990410","35221710","35119710",
    "35856910","35991610","35120310","35284510",
    "35231910","35784510","35990610","35221810","35119610",
    "35842910","35284710","35119910","35990910",
    "86751904","86611102","86498904","86732204","86521604",
    "86498210","86739210","86521910","35284810","86739510",
    "35986710","86738904","86521810","35990110",
    "35124510","86739110","86521710","35990210",
    "35284910","86739310","35990710","35119510",
    "35354410","35119810","35990810",
    "86738710","35990310","35119410",
]

HIGH_VALUE_TAC = [
    "35328004","01326300","35299406","35469208","35607204",
    "35674108","35919004","35821804",
    "86498904","86732204",
]

TAC_DB = {
    "35674108":("Samsung","Galaxy Series"),   "35919004":("Samsung","Galaxy A"),
    "35821804":("Samsung","Galaxy S"),        "35328004":("Apple","iPhone"),
    "01326300":("Apple","iPhone 14"),         "35299406":("Apple","iPhone 13"),
    "35469208":("Apple","iPhone 15"),         "35607204":("Apple","iPhone 12"),
    "35761904":("Tecno","Spark"),             "35445610":("Tecno","Camon"),
    "35990410":("Tecno","Pop"),               "35856910":("Itel","A Series"),
    "35991610":("Itel","Vision"),             "35231910":("Infinix","Hot"),
    "35784510":("Infinix","Note"),            "35842910":("Nokia","G Series"),
    "86751904":("Huawei","Y Series"),         "86611102":("Huawei","Nova"),
    "86498904":("Huawei","P Series"),         "86732204":("Huawei","Mate"),
    "86498210":("Xiaomi","Redmi"),            "86739210":("Xiaomi","Mi"),
    "35986710":("Oppo","A Series"),           "35124510":("Vivo","Y Series"),
    "35284910":("Realme","C Series"),         "35354410":("Motorola","Moto G"),
    "86738710":("ZTE","Blade"),               "35990310":("Wiko","T Series"),
}

TEST_IMEIS = {
    "000000000000000","111111111111111",
    "123456789012345","999999999999999",
}

MODEL_PATH   = "traceimei_model.pkl"
METRICS_PATH = "model_metrics.json"

# ────────────────────────────────────────────────────────────
# UTILITAIRES
# ────────────────────────────────────────────────────────────

def luhn_check(imei: str) -> bool:
    if len(imei) != 15 or not imei.isdigit():
        return False
    digits = [int(d) for d in imei]
    total  = sum(digits[-1::-2])
    for d in digits[-2::-2]:
        total += sum(divmod(d * 2, 10))
    return total % 10 == 0


def get_manufacturer(imei: str):
    tac = imei[:8] if len(imei) >= 8 else ""
    for prefix, (brand, series) in TAC_DB.items():
        if tac.startswith(prefix[:6]):
            return brand, series, True
    return "Inconnu", "Modèle inconnu", False


def build_features(imei: str, ctx: dict = {}) -> list:
    luhn       = 1 if luhn_check(imei) else 0
    tac        = imei[:8] if len(imei) >= 8 else ""
    tac_known  = int(any(tac.startswith(k[:6]) for k in KNOWN_TAC))
    all_same   = 1 if imei and len(set(imei)) == 1 else 0
    is_test    = 1 if imei in TEST_IMEIS else 0
    length_ok  = 1 if len(imei) == 15 else 0
    tac_theft  = 0.7 if any(tac.startswith(t[:6]) for t in HIGH_VALUE_TAC) else 0.2

    if imei.isdigit() and len(imei) > 0:
        counts  = [imei.count(str(d)) / len(imei) for d in range(10)]
        entropy = -sum(p * np.log2(p + 1e-9) for p in counts if p > 0)
        digit_e = round(entropy / np.log2(10), 4)
    else:
        digit_e = 0.0

    check_freq = min(int(ctx.get("check_count", 1)), 15) / 15
    hour_norm  = int(ctx.get("hour", datetime.utcnow().hour)) / 23
    days_seen  = float(ctx.get("days_since_first_seen", 0)) / 365

    return [luhn, tac_known, all_same, is_test, length_ok,
            tac_theft, digit_e, check_freq, hour_norm, days_seen]

# ────────────────────────────────────────────────────────────
# AUTO-TRAINING
# ────────────────────────────────────────────────────────────

def train_and_save():
    """Entraîne RF + IF sur dataset synthétique et sauvegarde le modèle."""
    from sklearn.ensemble import RandomForestClassifier, IsolationForest

    print("🔧 Entraînement du modèle en cours...")
    random.seed(42)
    np.random.seed(42)

    def _gen_valid_imei(tac=None):
        t = tac or random.choice(KNOWN_TAC)
        snr = str(random.randint(0, 999999)).zfill(6)
        base = t + snr
        digits = [int(d) for d in base]
        total = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d *= 2
                if d > 9: d -= 9
            total += d
        return base + str((10 - total % 10) % 10)

    def _gen_fake_imei():
        mode = random.choice(["luhn_bad", "unknown_tac", "repeated"])
        if mode == "luhn_bad":
            tac = random.choice(KNOWN_TAC)
            snr = str(random.randint(0, 999999)).zfill(6)
            base = tac + snr
            digits = [int(d) for d in base]
            total = 0
            for i, d in enumerate(reversed(digits)):
                if i % 2 == 1:
                    d *= 2
                    if d > 9: d -= 9
                total += d
            good = (10 - total % 10) % 10
            bad  = (good + random.randint(1, 9)) % 10
            return base + str(bad)
        elif mode == "unknown_tac":
            fake_tac = str(random.randint(10000000, 99999999))
            snr = str(random.randint(0, 999999)).zfill(6)
            base = fake_tac + snr
            digits = [int(d) for d in base]
            total = 0
            for i, d in enumerate(reversed(digits)):
                if i % 2 == 1:
                    d *= 2
                    if d > 9: d -= 9
                total += d
            return base + str((10 - total % 10) % 10)
        else:
            return str(random.randint(0, 9)) * 15

    def _features(imei, label):
        luhn      = 1 if luhn_check(imei) else 0
        tac       = imei[:8] if len(imei) >= 8 else ""
        tac_known = int(any(tac.startswith(k[:6]) for k in KNOWN_TAC))
        all_same  = 1 if imei and len(set(imei)) == 1 else 0
        is_test   = 1 if imei in TEST_IMEIS else 0
        length_ok = 1 if len(imei) == 15 else 0
        tac_theft = 0.7 if any(tac.startswith(t[:6]) for t in HIGH_VALUE_TAC) else 0.2
        if imei.isdigit() and len(imei) > 0:
            counts  = [imei.count(str(d)) / len(imei) for d in range(10)]
            entropy = -sum(p * np.log2(p + 1e-9) for p in counts if p > 0)
            digit_e = round(entropy / np.log2(10), 4)
        else:
            digit_e = 0.0
        check_freq = (random.randint(3,15) if label==1 else random.randint(1,3)) / 15
        hour = random.choices(range(24),
            weights=[3,2,2,2,2,2,3,4,5,5,5,5,5,5,5,5,5,5,6,7,8,8,6,4])[0] if label==1 \
            else random.choices(range(24),
            weights=[1,1,1,1,1,1,2,5,8,9,9,9,9,9,9,9,8,7,6,5,4,3,2,1])[0]
        hour_norm = hour / 23
        days_seen = (random.randint(0,30) if label==1 else random.randint(30,730)) / 365
        return [luhn,tac_known,all_same,is_test,length_ok,
                tac_theft,digit_e,check_freq,hour_norm,days_seen]

    # Dataset : label 1 = comportement suspect (pas nécessairement volé)
    records = []
    for _ in range(1800):
        records.append(_features(_gen_valid_imei(), 0) + [0])
    for _ in range(900):
        records.append(_features(_gen_valid_imei(), 1) + [1])
    for _ in range(300):
        records.append(_features(_gen_fake_imei(), 1) + [1])

    random.shuffle(records)
    X = np.array([r[:-1] for r in records])
    y = np.array([r[-1]  for r in records])

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X, y)

    iso = IsolationForest(
        n_estimators=200, contamination=0.4,
        random_state=42, n_jobs=-1
    )
    iso.fit(X)

    iso_scores = -iso.decision_function(X)
    iso_min    = float(iso_scores.min())
    iso_max    = float(iso_scores.max())

    feature_names = ["luhn","tac_known","all_same","is_test","length_ok",
                     "tac_theft_rate","digit_entropy","check_freq","hour_norm","days_seen"]

    bundle = {
        "rf": rf, "iso": iso,
        "iso_min": iso_min, "iso_max": iso_max,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "version": "3.2"
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "model_version":      "TraceIMEI-BJ v3.2-RF+IF",
        "n_features":         len(feature_names),
        "feature_names":      feature_names,
        "dataset_size":       len(records),
        "data_origin":        "synthetic_behavioral_v3",
        "rf_estimators":      200,
        "iso_contamination":  0.4,
        "logic_note":         "VOLÉ=Supabase | SUSPECT=ML>=0.50 | LÉGITIME=ML<0.50"
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("✅ Modèle v3.2 entraîné et sauvegardé.")
    return bundle, metrics

# ────────────────────────────────────────────────────────────
# CHARGEMENT (ou entraînement) DU MODÈLE
# ────────────────────────────────────────────────────────────

MODEL   = None
METRICS = {}

def load_model():
    global MODEL, METRICS
    if os.path.exists(MODEL_PATH):
        try:
            MODEL = joblib.load(MODEL_PATH)
            print(f"✅ Modèle v{MODEL.get('version','?')} chargé.")
        except Exception as e:
            print(f"⚠️  Modèle corrompu ({e}), réentraînement...")
            MODEL, METRICS = train_and_save()
            return
    else:
        print("ℹ️  Aucun modèle trouvé, entraînement automatique...")
        MODEL, METRICS = train_and_save()
        return

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            METRICS = json.load(f)
    else:
        METRICS = {"model_version": "TraceIMEI-BJ v3.2-RF+IF"}

load_model()

# ────────────────────────────────────────────────────────────
# SCORING — LOGIQUE MÉTIER v3.2
#
# Règle fondamentale :
#   1. Si is_declared_stolen=True  → VOLÉ   (décision Supabase, ML enrichit)
#   2. Si ML score >= 0.50         → SUSPECT (ML détecte comportement anormal)
#   3. Sinon                       → LÉGITIME
#
# Le ML ne sort JAMAIS "volé" seul. Il détecte des anomalies comportementales.
# ────────────────────────────────────────────────────────────

def compute_ml_score(imei: str, ctx: dict = {}):
    """
    Retourne (score, mode, confidence).
    Score = probabilité d'anomalie comportementale (0.0 → 1.0).
    Ne détermine PAS si l'IMEI est volé — c'est le rôle de Supabase.
    """
    # IMEI de test connus → score maximum
    if imei in TEST_IMEIS:
        return 0.99, "blacklist", "high"
    # IMEI avec tous les chiffres identiques → clairement invalide
    if imei and len(imei) == 15 and len(set(imei)) == 1:
        return 0.99, "blacklist", "high"

    # Fallback si modèle absent
    if MODEL is None:
        score = 0.0
        if not luhn_check(imei): score += 0.40
        if len(imei) != 15:      score += 0.40
        return round(min(score, 0.99), 3), "fallback_rules", "low"

    X        = np.array([build_features(imei, ctx)])
    rf_proba = float(MODEL["rf"].predict_proba(X)[0][1])
    iso_raw  = float(-MODEL["iso"].decision_function(X)[0])
    iso_norm = float(np.clip(
        (iso_raw - MODEL["iso_min"]) / (MODEL["iso_max"] - MODEL["iso_min"] + 1e-9), 0, 1
    ))
    score = round(min(0.70 * rf_proba + 0.30 * iso_norm, 0.99), 3)
    return score, "scoring_v3", "high"


def resolve_status(score: float, is_declared_stolen: bool) -> str:
    """
    Logique métier officielle v3.2 :
      - VOLÉ     : déclaré volé dans Supabase (is_declared_stolen=True)
      - SUSPECT  : ML détecte anomalie comportementale (score >= 0.50)
      - LÉGITIME : comportement normal, non déclaré volé
    """
    if is_declared_stolen:
        return "vole"
    if score >= 0.50:
        return "suspect"
    return "legitime"

# ────────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name":         "TraceIMEI-BJ API",
        "version":      "3.2.0",
        "engine":       "Random Forest 70% + Isolation Forest 30%",
        "n_features":   MODEL["n_features"] if MODEL else 10,
        "status":       "running",
        "model_loaded": MODEL is not None,
        "logic":        "VOLE=Supabase | SUSPECT=ML>=0.50 | LEGITIME=ML<0.50",
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": MODEL is not None,
        "scoring_mode": "scoring_v3" if MODEL else "fallback_rules",
        "version":      "3.2.0",
        "timestamp":    time.time(),
    })


@app.route("/api/check-imei", methods=["POST"])
def check_imei():
    """
    Body attendu :
    {
        "imei": "358441080000000",
        "is_declared_stolen": false,   ← envoyé par le frontend depuis Supabase
        "context": {                   ← optionnel
            "check_count": 3,
            "hour": 22,
            "days_since_first_seen": 5
        }
    }
    """
    start = time.time()
    data  = request.get_json()
    if not data or "imei" not in data:
        return jsonify({"error": "IMEI manquant"}), 400

    imei               = str(data.get("imei", "")).strip()
    ctx                = data.get("context", {})
    # Clé principale : envoyée par le frontend après consultation Supabase
    is_declared_stolen = bool(data.get("is_declared_stolen", False))

    luhn                         = luhn_check(imei)
    manufacturer, series, tac_ok = get_manufacturer(imei)
    score, mode, confidence      = compute_ml_score(imei, ctx)

    # ← LOGIQUE MÉTIER CENTRALISÉE ICI
    status = resolve_status(score, is_declared_stolen)

    return jsonify({
        "imei":               imei,
        "score":              score,
        "status":             status,
        "is_declared_stolen": is_declared_stolen,
        "manufacturer":       manufacturer,
        "model_series":       series,
        "features": {
            "luhn_valid":             luhn,
            "imei_length_valid":      len(imei) == 15,
            "tac_code":               imei[:8] if len(imei) >= 8 else "",
            "tac_manufacturer_match": tac_ok,
            "all_same_digits":        len(set(imei)) == 1 if imei else False,
            "known_test_imei":        imei in TEST_IMEIS,
        },
        "scoring_mode":       mode,
        "confidence":         confidence,
        "response_time_ms":   round((time.time() - start) * 1000, 2),
        "model_version":      METRICS.get("model_version", "TraceIMEI-BJ v3.2-RF+IF"),
        "logic_version":      "v3.2",
    })


@app.route("/api/batch-check", methods=["POST"])
def batch_check():
    """
    Body attendu :
    {
        "imeis": ["imei1", "imei2", ...],
        "declared_stolen_list": ["imei1"],   ← liste des IMEI déclarés volés (depuis Supabase)
        "context": {}
    }
    """
    data = request.get_json()
    if not data or "imeis" not in data:
        return jsonify({"error": "Liste IMEI manquante"}), 400

    imeis               = data.get("imeis", [])[:50]
    ctx                 = data.get("context", {})
    # Le frontend passe la liste des IMEI confirmés volés dans Supabase
    declared_stolen_set = set(data.get("declared_stolen_list", []))
    results             = []

    for imei in imeis:
        imei               = str(imei).strip()
        is_declared_stolen = imei in declared_stolen_set
        score, mode, confidence      = compute_ml_score(imei, ctx)
        manufacturer, series, tac_ok = get_manufacturer(imei)
        status = resolve_status(score, is_declared_stolen)
        results.append({
            "imei":               imei,
            "score":              score,
            "status":             status,
            "is_declared_stolen": is_declared_stolen,
            "manufacturer":       manufacturer,
            "tac_match":          tac_ok,
            "scoring_mode":       mode,
            "confidence":         confidence,
        })

    return jsonify({
        "results":       results,
        "total":         len(results),
        "model_version": METRICS.get("model_version", "TraceIMEI-BJ v3.2-RF+IF"),
        "logic_version": "v3.2",
    })


@app.route("/api/retrain", methods=["POST"])
def retrain():
    """Endpoint pour forcer un réentraînement sans redéployer."""
    global MODEL, METRICS
    try:
        MODEL, METRICS = train_and_save()
        return jsonify({"status": "ok", "message": "Modèle réentraîné avec succès", "version": "3.2"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
