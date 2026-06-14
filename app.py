"""
TraceIMEI-BJ — API ML v3.7
Logique métier :
  - VOLÉ     → is_declared_stolen=True (source : Supabase)
  - SUSPECT  → Isolation Forest iso_score >= 0.50
               OU IMEI vérifié >= 5 fois dans la même journée
  - LÉGITIME → aucune des conditions ci-dessus

Changements v3.7 :
  - hour_norm et days_seen supprimés — trop bruités sur données synthétiques
  - RF réduit à 20% (score affichage), IF à 80%
  - 8 features purement structurelles sur l'IMEI

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
import requests
from datetime import datetime, timezone

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ────────────────────────────────────────────────────────────
# VARIABLES D'ENVIRONNEMENT
# ────────────────────────────────────────────────────────────

SUPABASE_URL              = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

SUPABASE_HEADERS = {
    "apikey":        SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=minimal",
}

# ────────────────────────────────────────────────────────────
# CONSTANTES
# ────────────────────────────────────────────────────────────

TEST_IMEIS = {
    "000000000000000", "111111111111111",
    "123456789012345", "999999999999999",
}

DAILY_CHECK_THRESHOLD = 5
ISO_ANOMALY_THRESHOLD = 0.50

MODEL_PATH   = "traceimei_model.pkl"
METRICS_PATH = "model_metrics.json"

# 8 features purement structurelles — aucune dépendance temporelle
FEATURE_NAMES = [
    "luhn",
    "all_same",
    "is_test",
    "length_ok",
    "digit_entropy",
    "consecutive",
    "unique_digit_ratio",
    "luhn_entropy_product",   # interaction : luhn × digit_entropy
]

# ────────────────────────────────────────────────────────────
# TAC LOOKUP (affichage uniquement)
# ────────────────────────────────────────────────────────────

def tac_lookup_supabase(tac: str):
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return None
    try:
        url    = f"{SUPABASE_URL}/rest/v1/enregistrements_imei"
        params = {"tac": f"eq.{tac}", "select": "tac,manufacturer,model_series", "limit": "1"}
        r = requests.get(url, headers=SUPABASE_HEADERS, params=params, timeout=3)
        if r.status_code == 200:
            data = r.json()
            if data:
                return data[0]["manufacturer"], data[0]["model_series"], True
    except Exception as e:
        print(f"⚠️  Supabase TAC lookup failed: {e}")
    return None


def tac_lookup_imeicheck(imei: str):
    try:
        url = f"https://alpha.imeicheck.com/api/modelBrandName?imei={imei}&format=json"
        r   = requests.get(url, timeout=5)
        if r.status_code == 200:
            data  = r.json()
            brand = data.get("brand") or data.get("manufacturer") or ""
            model = data.get("model") or data.get("name") or ""
            if brand:
                return brand.strip(), model.strip(), True
    except Exception as e:
        print(f"⚠️  imeicheck.com lookup failed: {e}")
    return None


def tac_save_supabase(tac: str, manufacturer: str, model_series: str):
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/enregistrements_imei",
            headers=SUPABASE_HEADERS,
            json={"tac": tac, "manufacturer": manufacturer, "model_series": model_series},
            timeout=3,
        )
    except Exception as e:
        print(f"⚠️  Supabase TAC save failed: {e}")


def get_manufacturer(imei: str):
    if len(imei) < 8:
        return "Inconnu", "Modèle inconnu", False
    tac    = imei[:8]
    result = tac_lookup_supabase(tac)
    if result:
        return result[0], result[1], True
    result = tac_lookup_imeicheck(imei)
    if result:
        tac_save_supabase(tac, result[0], result[1])
        return result[0], result[1], True
    return "Inconnu", "Modèle inconnu", False

# ────────────────────────────────────────────────────────────
# SUPABASE — COMPTAGE JOURNALIER
# ────────────────────────────────────────────────────────────

def get_daily_check_count(imei: str) -> int:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return 0
    try:
        today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        url    = f"{SUPABASE_URL}/rest/v1/verifications_imei"
        params = {"imei": f"eq.{imei}", "created_at": f"gte.{today}T00:00:00Z", "select": "imei"}
        headers = {**SUPABASE_HEADERS, "Prefer": "count=exact"}
        r = requests.get(url, headers=headers, params=params, timeout=3)
        if r.status_code in (200, 206):
            cr = r.headers.get("Content-Range", "")
            if "/" in cr:
                return int(cr.split("/")[1])
            return len(r.json())
    except Exception as e:
        print(f"⚠️  Supabase daily check count failed: {e}")
    return 0

# ────────────────────────────────────────────────────────────
# UTILITAIRES ML
# ────────────────────────────────────────────────────────────

def luhn_check(imei: str) -> bool:
    if len(imei) != 15 or not imei.isdigit():
        return False
    digits = [int(d) for d in imei]
    total  = sum(digits[-1::-2])
    for d in digits[-2::-2]:
        total += sum(divmod(d * 2, 10))
    return total % 10 == 0


def build_features(imei: str) -> list:
    """
    8 features purement structurelles — sans dépendance temporelle.

      0  luhn                — Luhn valide (0/1)
      1  all_same            — tous chiffres identiques (0/1)
      2  is_test             — IMEI liste noire test (0/1)
      3  length_ok           — longueur 15 (0/1)
      4  digit_entropy       — entropie normalisée (0.0–1.0)
      5  consecutive         — proportion transitions +1 (0.0–1.0)
      6  unique_digit_ratio  — ratio chiffres uniques / 10
      7  luhn_entropy_product— luhn × digit_entropy (interaction)
    """
    luhn      = 1 if luhn_check(imei) else 0
    all_same  = 1 if imei and len(set(imei)) == 1 else 0
    is_test   = 1 if imei in TEST_IMEIS else 0
    length_ok = 1 if len(imei) == 15 else 0

    if imei.isdigit() and len(imei) > 0:
        counts  = [imei.count(str(d)) / len(imei) for d in range(10)]
        entropy = -sum(p * np.log2(p + 1e-9) for p in counts if p > 0)
        digit_e = round(entropy / np.log2(10), 4)
    else:
        digit_e = 0.0

    if imei.isdigit() and len(imei) == 15:
        consecutive = sum(
            1 for i in range(len(imei) - 1)
            if int(imei[i + 1]) - int(imei[i]) == 1
        ) / 14
    else:
        consecutive = 0.0

    unique_digit_ratio   = len(set(imei)) / 10 if imei.isdigit() else 0.0
    luhn_entropy_product = round(luhn * digit_e, 4)

    return [luhn, all_same, is_test, length_ok,
            digit_e, consecutive, unique_digit_ratio, luhn_entropy_product]

# ────────────────────────────────────────────────────────────
# AUTO-TRAINING v3.7
# ────────────────────────────────────────────────────────────

KNOWN_TAC_TRAIN = [
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


def train_and_save():
    from sklearn.ensemble import RandomForestClassifier, IsolationForest

    print("🔧 Entraînement du modèle v3.7 en cours...")
    random.seed(42)
    np.random.seed(42)

    def _gen_valid_imei(tac=None):
        t      = tac or random.choice(KNOWN_TAC_TRAIN)
        snr    = str(random.randint(0, 999999)).zfill(6)
        base   = t + snr
        digits = [int(d) for d in base]
        total  = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d *= 2
                if d > 9: d -= 9
            total += d
        return base + str((10 - total % 10) % 10)

    def _gen_fake_imei():
        mode = random.choice(["luhn_bad", "unknown_tac", "repeated", "sequential"])
        if mode == "luhn_bad":
            tac  = random.choice(KNOWN_TAC_TRAIN)
            snr  = str(random.randint(0, 999999)).zfill(6)
            base = tac + snr
            digits = [int(d) for d in base]
            total  = 0
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
            snr      = str(random.randint(0, 999999)).zfill(6)
            base     = fake_tac + snr
            digits   = [int(d) for d in base]
            total    = 0
            for i, d in enumerate(reversed(digits)):
                if i % 2 == 1:
                    d *= 2
                    if d > 9: d -= 9
                total += d
            return base + str((10 - total % 10) % 10)
        elif mode == "sequential":
            base   = "".join([str(i % 10) for i in range(14)])
            digits = [int(d) for d in base]
            total  = 0
            for i, d in enumerate(reversed(digits)):
                if i % 2 == 1:
                    d *= 2
                    if d > 9: d -= 9
                total += d
            return base + str((10 - total % 10) % 10)
        else:
            return str(random.randint(0, 9)) * 15

    def _features(imei):
        return build_features(imei)

    records = []
    # Légitimes : IMEI valides générés depuis TAC connus
    for _ in range(2000):
        records.append(_features(_gen_valid_imei()) + [0])
    # Suspects : IMEI mal formés
    for _ in range(1000):
        records.append(_features(_gen_fake_imei()) + [1])

    random.shuffle(records)
    X = np.array([r[:-1] for r in records])
    y = np.array([r[-1]  for r in records])

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)

    # contamination=0.20 : on s'attend à 20% d'IMEI suspects dans la vraie vie
    iso = IsolationForest(
        n_estimators=200, contamination=0.20,
        random_state=42, n_jobs=-1,
    )
    iso.fit(X)

    iso_scores = -iso.decision_function(X)
    iso_min    = float(iso_scores.min())
    iso_max    = float(iso_scores.max())

    bundle = {
        "rf":            rf,
        "iso":           iso,
        "iso_min":       iso_min,
        "iso_max":       iso_max,
        "feature_names": FEATURE_NAMES,
        "n_features":    len(FEATURE_NAMES),
        "version":       "3.7",
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "model_version":         "TraceIMEI-BJ v3.7-RF+IF",
        "n_features":            len(FEATURE_NAMES),
        "feature_names":         FEATURE_NAMES,
        "dataset_size":          len(records),
        "rf_estimators":         200,
        "rf_max_depth":          6,
        "iso_contamination":     0.20,
        "iso_anomaly_threshold": ISO_ANOMALY_THRESHOLD,
        "daily_check_threshold": DAILY_CHECK_THRESHOLD,
        "rf_weight":             0.20,
        "iso_weight":            0.80,
        "logic_note": (
            "VOLÉ=Supabase | "
            "SUSPECT=IF>=0.50 OU verifications_imei>=5/jour | "
            "LÉGITIME=sinon"
        ),
        "changelog_v3.7": [
            "hour_norm supprimé — biais temporel synthétique",
            "days_seen supprimé — toujours 0 en production",
            "RF réduit à 20% (score display), IF à 80%",
            "contamination IF abaissée à 0.20 (plus réaliste)",
            "Ajout feature luhn_entropy_product (interaction)",
            "8 features purement structurelles",
        ],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ Modèle v3.7 entraîné ({len(FEATURE_NAMES)} features) et sauvegardé.")
    return bundle, metrics

# ────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ────────────────────────────────────────────────────────────

MODEL   = None
METRICS = {}
REQUIRED_BUNDLE_KEYS = ["rf", "iso", "iso_min", "iso_max", "n_features"]


def load_model():
    global MODEL, METRICS
    if os.path.exists(MODEL_PATH):
        try:
            loaded = joblib.load(MODEL_PATH)
            if not all(k in loaded for k in REQUIRED_BUNDLE_KEYS):
                print("⚠️  Bundle incomplet, réentraînement forcé...")
                MODEL, METRICS = train_and_save()
                return
            if loaded.get("n_features") != len(FEATURE_NAMES):
                print(f"⚠️  n_features mismatch ({loaded.get('n_features')} vs {len(FEATURE_NAMES)}), réentraînement...")
                MODEL, METRICS = train_and_save()
                return
            MODEL = loaded
            print(f"✅ Modèle v{MODEL.get('version','?')} chargé ({MODEL['n_features']} features).")
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
        METRICS = {"model_version": "TraceIMEI-BJ v3.7-RF+IF"}


load_model()

# ────────────────────────────────────────────────────────────
# SCORING
# ────────────────────────────────────────────────────────────

def compute_ml_score(imei: str):
    if imei in TEST_IMEIS or (imei and len(imei) == 15 and len(set(imei)) == 1):
        return 0.99, 0.99, 0.99, "blacklist", "high"

    if MODEL is None:
        score = 0.0
        if not luhn_check(imei): score += 0.40
        if len(imei) != 15:      score += 0.40
        return round(min(score, 0.99), 3), score, score, "fallback_rules", "low"

    X        = np.array([build_features(imei)])
    rf_proba = float(MODEL["rf"].predict_proba(X)[0][1])
    iso_raw  = float(-MODEL["iso"].decision_function(X)[0])
    iso_norm = float(np.clip(
        (iso_raw - MODEL["iso_min"]) / (MODEL["iso_max"] - MODEL["iso_min"] + 1e-9),
        0, 1,
    ))
    # RF 20% uniquement pour l'affichage — IF 80% décide
    score_display = round(min(0.20 * rf_proba + 0.80 * iso_norm, 0.99), 3)
    return score_display, round(iso_norm, 3), round(rf_proba, 3), "scoring_v3.7", "high"


def resolve_status(iso_score, is_declared_stolen, daily_count):
    if is_declared_stolen:
        return "vole", "declared_stolen"
    if iso_score >= ISO_ANOMALY_THRESHOLD:
        if daily_count >= DAILY_CHECK_THRESHOLD:
            return "suspect", "anomalie_ml_et_frequence_journaliere"
        return "suspect", "anomalie_ml"
    if daily_count >= DAILY_CHECK_THRESHOLD:
        return "suspect", "frequence_journaliere"
    return "legitime", "aucune_anomalie"

# ────────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name":                  "TraceIMEI-BJ API",
        "version":               "3.7.0",
        "engine":                "Isolation Forest 80% + Random Forest 20% (display)",
        "n_features":            len(FEATURE_NAMES),
        "feature_names":         FEATURE_NAMES,
        "iso_anomaly_threshold": ISO_ANOMALY_THRESHOLD,
        "daily_check_threshold": DAILY_CHECK_THRESHOLD,
        "status":                "running",
        "model_loaded":          MODEL is not None,
        "logic": "VOLE=Supabase | SUSPECT=IF>=0.50 OU verifications/jour>=5 | LEGITIME=sinon",
    })


@app.route("/api/health", methods=["GET"])
def health():
    supabase_ok = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)
    return jsonify({
        "status":                "ok",
        "model_loaded":          MODEL is not None,
        "model_version":         MODEL.get("version", "?") if MODEL else "none",
        "n_features":            MODEL.get("n_features", 0) if MODEL else 0,
        "scoring_mode":          "scoring_v3.7" if MODEL else "fallback_rules",
        "iso_anomaly_threshold": ISO_ANOMALY_THRESHOLD,
        "daily_check_threshold": DAILY_CHECK_THRESHOLD,
        "supabase":              "connected" if supabase_ok else "missing_credentials",
        "version":               "3.7.0",
        "timestamp":             time.time(),
    })


@app.route("/api/check-imei", methods=["POST"])
def check_imei():
    start = time.time()
    data  = request.get_json()
    if not data or "imei" not in data:
        return jsonify({"error": "IMEI manquant"}), 400

    imei               = str(data.get("imei", "")).strip()
    is_declared_stolen = bool(data.get("is_declared_stolen", False))

    luhn_ok                      = luhn_check(imei)
    manufacturer, series, tac_ok = get_manufacturer(imei)
    daily_count                  = get_daily_check_count(imei)

    score_display, iso_score, rf_proba, mode, confidence = compute_ml_score(imei)
    status, reason = resolve_status(iso_score, is_declared_stolen, daily_count)

    features_detail = dict(zip(FEATURE_NAMES, build_features(imei)))

    return jsonify({
        "imei":               imei,
        "score":              score_display,
        "iso_score":          iso_score,
        "rf_proba":           rf_proba,
        "status":             status,
        "suspect_reason":     reason,
        "is_declared_stolen": is_declared_stolen,
        "manufacturer":       manufacturer,
        "model_series":       series,
        "daily_check_count":  daily_count,
        "daily_threshold":    DAILY_CHECK_THRESHOLD,
        "iso_threshold":      ISO_ANOMALY_THRESHOLD,
        "features": {
            "luhn_valid":        luhn_ok,
            "imei_length_valid": len(imei) == 15,
            "tac_code":          imei[:8] if len(imei) >= 8 else "",
            "tac_found":         tac_ok,
            "all_same_digits":   len(set(imei)) == 1 if imei else False,
            "known_test_imei":   imei in TEST_IMEIS,
        },
        "features_ml":      features_detail,
        "scoring_mode":     mode,
        "confidence":       confidence,
        "response_time_ms": round((time.time() - start) * 1000, 2),
        "model_version":    METRICS.get("model_version", "TraceIMEI-BJ v3.7-RF+IF"),
        "logic_version":    "v3.7",
    })


@app.route("/api/batch-check", methods=["POST"])
def batch_check():
    data = request.get_json()
    if not data or "imeis" not in data:
        return jsonify({"error": "Liste IMEI manquante"}), 400

    imeis               = data.get("imeis", [])[:50]
    declared_stolen_set = set(data.get("declared_stolen_list", []))
    results             = []

    for imei in imeis:
        imei               = str(imei).strip()
        is_declared_stolen = imei in declared_stolen_set
        manufacturer, series, tac_ok = get_manufacturer(imei)
        daily_count = get_daily_check_count(imei)
        score_display, iso_score, rf_proba, mode, confidence = compute_ml_score(imei)
        status, reason = resolve_status(iso_score, is_declared_stolen, daily_count)
        results.append({
            "imei":               imei,
            "score":              score_display,
            "iso_score":          iso_score,
            "status":             status,
            "suspect_reason":     reason,
            "is_declared_stolen": is_declared_stolen,
            "manufacturer":       manufacturer,
            "model_series":       series,
            "tac_match":          tac_ok,
            "daily_check_count":  daily_count,
            "scoring_mode":       mode,
            "confidence":         confidence,
        })

    return jsonify({
        "results":       results,
        "total":         len(results),
        "model_version": METRICS.get("model_version", "TraceIMEI-BJ v3.7-RF+IF"),
        "logic_version": "v3.7",
    })


@app.route("/api/retrain", methods=["POST"])
def retrain():
    global MODEL, METRICS
    try:
        MODEL, METRICS = train_and_save()
        return jsonify({
            "status":        "ok",
            "message":       "Modèle v3.7 réentraîné avec succès",
            "version":       "3.7",
            "n_features":    MODEL["n_features"],
            "feature_names": MODEL["feature_names"],
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/features", methods=["GET"])
def get_features():
    descriptions = {
        "luhn":                 "Algorithme de Luhn valide (0/1)",
        "all_same":             "Tous les chiffres identiques, ex: 111... (0/1)",
        "is_test":              "IMEI dans la liste noire de test (0/1)",
        "length_ok":            "Longueur exactement 15 chiffres (0/1)",
        "digit_entropy":        "Entropie normalisée des chiffres (0.0–1.0)",
        "consecutive":          "Proportion de transitions consécutives +1",
        "unique_digit_ratio":   "Ratio de chiffres uniques sur 10 possibles",
        "luhn_entropy_product": "Interaction luhn × digit_entropy",
    }
    return jsonify({
        "n_features":            len(FEATURE_NAMES),
        "iso_anomaly_threshold": ISO_ANOMALY_THRESHOLD,
        "daily_check_threshold": DAILY_CHECK_THRESHOLD,
        "decision_logic":        "SUSPECT si iso_score>=0.50 OU daily_count>=5",
        "rf_weight":             "20% (score display uniquement)",
        "iso_weight":            "80% (décideur principal)",
        "features": [
            {"name": n, "description": descriptions.get(n, "")}
            for n in FEATURE_NAMES
        ],
        "model_version": METRICS.get("model_version", "TraceIMEI-BJ v3.7-RF+IF"),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
