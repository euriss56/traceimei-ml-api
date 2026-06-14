"""
TraceIMEI-BJ — API ML v3.5
Logique métier :
  - VOLÉ     → is_declared_stolen=True (source : Supabase declarations/stolen_phones)
  - SUSPECT  → Isolation Forest détecte une anomalie (iso_score >= 0.50)
               OU IMEI vérifié >= 5 fois dans la même journée (table verifications_imei)
  - LÉGITIME → aucune des conditions ci-dessus

Rôle du Random Forest :
  - Ne décide plus du statut
  - Contribue uniquement au score final pour affichage (pondération 40%)
  - Isolation Forest = décideur principal (pondération 60%)

TAC lookup :
  1. Cache local Supabase (table enregistrements_imei) → priorité
  2. imeicheck.com API gratuite → fallback si absent du cache
  3. Sauvegarde automatique dans Supabase après chaque lookup réussi

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
# CONSTANTES ML
# ────────────────────────────────────────────────────────────

HIGH_VALUE_PREFIXES = [
    "353280", "013263", "352994", "354692", "356072",
    "356741", "359190", "358218", "864989", "867322",
]

TEST_IMEIS = {
    "000000000000000", "111111111111111",
    "123456789012345", "999999999999999",
}

# Seuil de vérifications répétées dans la journée → SUSPECT
DAILY_CHECK_THRESHOLD = 5

# Seuil Isolation Forest normalisé → SUSPECT
ISO_ANOMALY_THRESHOLD = 0.50

MODEL_PATH   = "traceimei_model.pkl"
METRICS_PATH = "model_metrics.json"

FEATURE_NAMES = [
    "luhn",
    "tac_known_real",
    "all_same",
    "is_test",
    "length_ok",
    "tac_theft_rate",
    "digit_entropy",
    "hour_norm",
    "days_seen",
    "consecutive",
    "unique_digit_ratio",
    "is_night",
]

# ────────────────────────────────────────────────────────────
# TAC LOOKUP — SUPABASE CACHE + IMEICHECK.COM FALLBACK
# ────────────────────────────────────────────────────────────

def tac_lookup_supabase(tac: str):
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return None
    try:
        url = f"{SUPABASE_URL}/rest/v1/enregistrements_imei"
        params = {
            "tac":    f"eq.{tac}",
            "select": "tac,manufacturer,model_series",
            "limit":  "1",
        }
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
        r = requests.get(url, timeout=5)
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
        url     = f"{SUPABASE_URL}/rest/v1/enregistrements_imei"
        payload = {"tac": tac, "manufacturer": manufacturer, "model_series": model_series}
        requests.post(url, headers=SUPABASE_HEADERS, json=payload, timeout=3)
        print(f"✅ TAC {tac} sauvegardé dans Supabase.")
    except Exception as e:
        print(f"⚠️  Supabase TAC save failed: {e}")


def get_manufacturer(imei: str):
    if len(imei) < 8:
        return "Inconnu", "Modèle inconnu", False
    tac = imei[:8]
    result = tac_lookup_supabase(tac)
    if result:
        manufacturer, model_series, _ = result
        return manufacturer, model_series, True
    result = tac_lookup_imeicheck(imei)
    if result:
        manufacturer, model_series, _ = result
        tac_save_supabase(tac, manufacturer, model_series)
        return manufacturer, model_series, True
    return "Inconnu", "Modèle inconnu", False

# ────────────────────────────────────────────────────────────
# SUPABASE — COMPTAGE DES VÉRIFICATIONS JOURNALIÈRES
# ────────────────────────────────────────────────────────────

def get_daily_check_count(imei: str) -> int:
    """
    Interroge la table verifications_imei pour compter combien de fois
    cet IMEI a été vérifié aujourd'hui (UTC).
    Colonnes attendues : imei (text), created_at (timestamptz)
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return 0
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        url   = f"{SUPABASE_URL}/rest/v1/verifications_imei"
        params = {
            "imei":       f"eq.{imei}",
            "created_at": f"gte.{today}T00:00:00Z",
            "select":     "imei",
        }
        # On utilise prefer: count=exact pour récupérer le total
        headers = {
            **SUPABASE_HEADERS,
            "Prefer": "count=exact",
        }
        r = requests.get(url, headers=headers, params=params, timeout=3)
        if r.status_code in (200, 206):
            content_range = r.headers.get("Content-Range", "")
            # Format : "0-4/12" → total = 12
            if "/" in content_range:
                return int(content_range.split("/")[1])
            # Fallback : compter les lignes retournées
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


def build_features(imei: str, ctx: dict = {}) -> list:
    """
    Vecteur de 12 features pour RF + Isolation Forest.
    check_freq et repeat_check supprimés : la fréquence
    est désormais décidée via Supabase (get_daily_check_count).
    """
    luhn      = 1 if luhn_check(imei) else 0
    tac       = imei[:8] if len(imei) >= 8 else ""
    all_same  = 1 if imei and len(set(imei)) == 1 else 0
    is_test   = 1 if imei in TEST_IMEIS else 0
    length_ok = 1 if len(imei) == 15 else 0
    tac_theft = 0.7 if any(tac.startswith(p) for p in HIGH_VALUE_PREFIXES) else 0.2

    if imei.isdigit() and len(imei) > 0:
        counts  = [imei.count(str(d)) / len(imei) for d in range(10)]
        entropy = -sum(p * np.log2(p + 1e-9) for p in counts if p > 0)
        digit_e = round(entropy / np.log2(10), 4)
    else:
        digit_e = 0.0

    tac_known_real     = 1 if bool(ctx.get("tac_found", False)) else 0
    hour_val           = int(ctx.get("hour", datetime.utcnow().hour))
    hour_norm          = hour_val / 23
    days_seen          = float(ctx.get("days_since_first_seen", 0)) / 365

    # Séquences consécutives
    if imei.isdigit() and len(imei) == 15:
        consecutive = sum(
            1 for i in range(len(imei) - 1)
            if int(imei[i + 1]) - int(imei[i]) == 1
        ) / 14
    else:
        consecutive = 0.0

    unique_digit_ratio = len(set(imei)) / 10 if imei.isdigit() else 0.0
    is_night           = 1 if (hour_val >= 22 or hour_val <= 5) else 0

    return [
        luhn, tac_known_real, all_same, is_test, length_ok,
        tac_theft, digit_e, hour_norm, days_seen,
        consecutive, unique_digit_ratio, is_night,
    ]

# ────────────────────────────────────────────────────────────
# AUTO-TRAINING v3.5
# ────────────────────────────────────────────────────────────

KNOWN_TAC_TRAIN = [
    "35674108", "35919004", "35821804", "35355810", "35284608",
    "35328004", "01326300", "35299406", "35469208", "35607204",
    "35761904", "35445610", "35990410", "35221710", "35119710",
    "35856910", "35991610", "35120310", "35284510",
    "35231910", "35784510", "35990610", "35221810", "35119610",
    "35842910", "35284710", "35119910", "35990910",
    "86751904", "86611102", "86498904", "86732204", "86521604",
    "86498210", "86739210", "86521910", "35284810", "86739510",
    "35986710", "86738904", "86521810", "35990110",
    "35124510", "86739110", "86521710", "35990210",
    "35284910", "86739310", "35990710", "35119510",
    "35354410", "35119810", "35990810",
    "86738710", "35990310", "35119410",
]


def train_and_save():
    from sklearn.ensemble import RandomForestClassifier, IsolationForest

    print("🔧 Entraînement du modèle v3.5 en cours...")
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

    def _features(imei, label):
        luhn           = 1 if luhn_check(imei) else 0
        tac            = imei[:8] if len(imei) >= 8 else ""
        tac_known_real = int(any(tac.startswith(k[:6]) for k in KNOWN_TAC_TRAIN))
        all_same       = 1 if imei and len(set(imei)) == 1 else 0
        is_test        = 1 if imei in TEST_IMEIS else 0
        length_ok      = 1 if len(imei) == 15 else 0
        tac_theft      = 0.7 if any(tac.startswith(p) for p in HIGH_VALUE_PREFIXES) else 0.2

        if imei.isdigit() and len(imei) > 0:
            counts  = [imei.count(str(d)) / len(imei) for d in range(10)]
            entropy = -sum(p * np.log2(p + 1e-9) for p in counts if p > 0)
            digit_e = round(entropy / np.log2(10), 4)
        else:
            digit_e = 0.0

        hour = random.choices(
            range(24),
            weights=[3,2,2,2,2,2,3,4,5,5,5,5,5,5,5,5,5,5,6,7,8,8,6,4]
        )[0] if label == 1 else random.choices(
            range(24),
            weights=[1,1,1,1,1,1,2,5,8,9,9,9,9,9,9,9,8,7,6,5,4,3,2,1]
        )[0]
        hour_norm = hour / 23
        days_seen = (random.randint(0, 30) if label == 1 else random.randint(30, 730)) / 365

        if imei.isdigit() and len(imei) == 15:
            consecutive = sum(
                1 for i in range(len(imei) - 1)
                if int(imei[i + 1]) - int(imei[i]) == 1
            ) / 14
        else:
            consecutive = 0.0

        unique_digit_ratio = len(set(imei)) / 10 if imei.isdigit() else 0.0
        is_night           = 1 if (hour >= 22 or hour <= 5) else 0

        return [
            luhn, tac_known_real, all_same, is_test, length_ok,
            tac_theft, digit_e, hour_norm, days_seen,
            consecutive, unique_digit_ratio, is_night,
        ]

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
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)

    iso = IsolationForest(
        n_estimators=200, contamination=0.35,
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
        "version":       "3.5",
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "model_version":        "TraceIMEI-BJ v3.5-RF+IF",
        "n_features":           len(FEATURE_NAMES),
        "feature_names":        FEATURE_NAMES,
        "dataset_size":         len(records),
        "data_origin":          "synthetic_behavioral_v3.5",
        "rf_estimators":        200,
        "iso_contamination":    0.35,
        "iso_anomaly_threshold": ISO_ANOMALY_THRESHOLD,
        "daily_check_threshold": DAILY_CHECK_THRESHOLD,
        "logic_note": (
            "VOLÉ=Supabase | "
            "SUSPECT=IsolationForest>=0.50 OU verifications_imei>=5/jour | "
            "LÉGITIME=sinon"
        ),
        "rf_role":     "score_display_only (pondération 40%)",
        "iso_role":    "décideur principal anomalie (pondération 60%)",
        "tac_source":  "Supabase cache + imeicheck.com fallback",
        "changelog_v3.5": [
            "RF retiré de la décision, rôle réduit à pondération score affichage",
            "Isolation Forest = décideur principal anomalie (seuil 0.50)",
            "Nouveau critère : IMEI vérifié >= 5 fois/jour via verifications_imei",
            "check_freq et repeat_check supprimés des features ML (gérés par Supabase)",
            "12 features au lieu de 14",
        ],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ Modèle v3.5 entraîné ({len(FEATURE_NAMES)} features) et sauvegardé.")
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
        METRICS = {"model_version": "TraceIMEI-BJ v3.5-RF+IF"}


load_model()

# ────────────────────────────────────────────────────────────
# SCORING v3.5
# ────────────────────────────────────────────────────────────

def compute_ml_score(imei: str, ctx: dict = {}):
    """
    Retourne (score_display, iso_score, rf_proba, mode, confidence).
    - iso_score  : score Isolation Forest normalisé [0,1] → décision SUSPECT
    - rf_proba   : probabilité Random Forest [0,1] → contribution affichage seulement
    - score_display : 0.40*rf + 0.60*iso → score affiché dans l'UI
    """
    if imei in TEST_IMEIS or (imei and len(imei) == 15 and len(set(imei)) == 1):
        return 0.99, 0.99, 0.99, "blacklist", "high"

    if MODEL is None:
        score = 0.0
        if not luhn_check(imei): score += 0.40
        if len(imei) != 15:      score += 0.40
        return round(min(score, 0.99), 3), score, score, "fallback_rules", "low"

    X        = np.array([build_features(imei, ctx)])
    rf_proba = float(MODEL["rf"].predict_proba(X)[0][1])
    iso_raw  = float(-MODEL["iso"].decision_function(X)[0])
    iso_norm = float(np.clip(
        (iso_raw - MODEL["iso_min"]) / (MODEL["iso_max"] - MODEL["iso_min"] + 1e-9),
        0, 1,
    ))
    score_display = round(min(0.40 * rf_proba + 0.60 * iso_norm, 0.99), 3)
    return score_display, round(iso_norm, 3), round(rf_proba, 3), "scoring_v3.5", "high"


def resolve_status(
    score_display: float,
    iso_score: float,
    is_declared_stolen: bool,
    daily_count: int,
) -> tuple:
    """
    Logique métier v3.5 :
      VOLÉ     → is_declared_stolen=True
      SUSPECT  → iso_score >= ISO_ANOMALY_THRESHOLD
                 OU daily_count >= DAILY_CHECK_THRESHOLD
      LÉGITIME → sinon

    Retourne (status, reason) pour transparence.
    """
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
        "version":               "3.5.0",
        "engine":                "Isolation Forest (décision) + Random Forest (score display)",
        "n_features":            len(FEATURE_NAMES),
        "feature_names":         FEATURE_NAMES,
        "iso_anomaly_threshold": ISO_ANOMALY_THRESHOLD,
        "daily_check_threshold": DAILY_CHECK_THRESHOLD,
        "status":                "running",
        "model_loaded":          MODEL is not None,
        "logic": (
            "VOLE=Supabase | "
            "SUSPECT=IF>=0.50 OU verifications/jour>=5 | "
            "LEGITIME=sinon"
        ),
    })


@app.route("/api/health", methods=["GET"])
def health():
    supabase_ok = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)
    return jsonify({
        "status":                "ok",
        "model_loaded":          MODEL is not None,
        "model_version":         MODEL.get("version", "?") if MODEL else "none",
        "n_features":            MODEL.get("n_features", 0) if MODEL else 0,
        "scoring_mode":          "scoring_v3.5" if MODEL else "fallback_rules",
        "iso_anomaly_threshold": ISO_ANOMALY_THRESHOLD,
        "daily_check_threshold": DAILY_CHECK_THRESHOLD,
        "supabase":              "connected" if supabase_ok else "missing_credentials",
        "version":               "3.5.0",
        "timestamp":             time.time(),
    })


@app.route("/api/check-imei", methods=["POST"])
def check_imei():
    """
    Body attendu :
    {
        "imei": "358441080000000",
        "is_declared_stolen": false,
        "context": {
            "hour": 14,
            "days_since_first_seen": 0
        }
    }
    Note : check_count n'est plus nécessaire dans le contexte.
    La fréquence journalière est lue directement depuis Supabase.
    """
    start = time.time()
    data  = request.get_json()
    if not data or "imei" not in data:
        return jsonify({"error": "IMEI manquant"}), 400

    imei               = str(data.get("imei", "")).strip()
    ctx                = data.get("context", {})
    is_declared_stolen = bool(data.get("is_declared_stolen", False))

    luhn_ok                      = luhn_check(imei)
    manufacturer, series, tac_ok = get_manufacturer(imei)
    ctx["tac_found"]             = tac_ok

    # Comptage Supabase : combien de fois cet IMEI a été vérifié aujourd'hui
    daily_count = get_daily_check_count(imei)

    score_display, iso_score, rf_proba, mode, confidence = compute_ml_score(imei, ctx)
    status, reason = resolve_status(score_display, iso_score, is_declared_stolen, daily_count)

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
            "luhn_valid":             luhn_ok,
            "imei_length_valid":      len(imei) == 15,
            "tac_code":               imei[:8] if len(imei) >= 8 else "",
            "tac_manufacturer_match": tac_ok,
            "all_same_digits":        len(set(imei)) == 1 if imei else False,
            "known_test_imei":        imei in TEST_IMEIS,
        },
        "scoring_mode":     mode,
        "confidence":       confidence,
        "response_time_ms": round((time.time() - start) * 1000, 2),
        "model_version":    METRICS.get("model_version", "TraceIMEI-BJ v3.5-RF+IF"),
        "logic_version":    "v3.5",
    })


@app.route("/api/batch-check", methods=["POST"])
def batch_check():
    """
    Body attendu :
    {
        "imeis": ["imei1", "imei2", ...],
        "declared_stolen_list": ["imei1"],
        "context": {}
    }
    """
    data = request.get_json()
    if not data or "imeis" not in data:
        return jsonify({"error": "Liste IMEI manquante"}), 400

    imeis               = data.get("imeis", [])[:50]
    ctx                 = data.get("context", {})
    declared_stolen_set = set(data.get("declared_stolen_list", []))
    results             = []

    for imei in imeis:
        imei               = str(imei).strip()
        is_declared_stolen = imei in declared_stolen_set
        manufacturer, series, tac_ok = get_manufacturer(imei)

        imei_ctx             = {**ctx, "tac_found": tac_ok}
        daily_count          = get_daily_check_count(imei)
        score_display, iso_score, rf_proba, mode, confidence = compute_ml_score(imei, imei_ctx)
        status, reason       = resolve_status(score_display, iso_score, is_declared_stolen, daily_count)

        results.append({
            "imei":              imei,
            "score":             score_display,
            "iso_score":         iso_score,
            "status":            status,
            "suspect_reason":    reason,
            "is_declared_stolen":is_declared_stolen,
            "manufacturer":      manufacturer,
            "model_series":      series,
            "tac_match":         tac_ok,
            "daily_check_count": daily_count,
            "scoring_mode":      mode,
            "confidence":        confidence,
        })

    return jsonify({
        "results":       results,
        "total":         len(results),
        "model_version": METRICS.get("model_version", "TraceIMEI-BJ v3.5-RF+IF"),
        "logic_version": "v3.5",
    })


@app.route("/api/retrain", methods=["POST"])
def retrain():
    global MODEL, METRICS
    try:
        MODEL, METRICS = train_and_save()
        return jsonify({
            "status":        "ok",
            "message":       "Modèle v3.5 réentraîné avec succès",
            "version":       "3.5",
            "n_features":    MODEL["n_features"],
            "feature_names": MODEL["feature_names"],
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/features", methods=["GET"])
def get_features():
    descriptions = {
        "luhn":              "Algorithme de Luhn valide (0/1)",
        "tac_known_real":    "TAC reconnu via Supabase ou imeicheck.com (0/1)",
        "all_same":          "Tous les chiffres identiques, ex: 111... (0/1)",
        "is_test":           "IMEI dans la liste noire de test (0/1)",
        "length_ok":         "Longueur exactement 15 chiffres (0/1)",
        "tac_theft_rate":    "TAC appartient aux préfixes haute valeur (0.2/0.7)",
        "digit_entropy":     "Entropie normalisée des chiffres (0.0–1.0)",
        "hour_norm":         "Heure UTC normalisée (0.0–1.0)",
        "days_seen":         "Ancienneté depuis première vérification, normalisée 365j",
        "consecutive":       "Proportion de transitions consécutives +1 entre chiffres",
        "unique_digit_ratio":"Ratio de chiffres uniques sur 10 possibles",
        "is_night":          "Vérification entre 22h et 5h UTC (0/1)",
    }
    return jsonify({
        "n_features":            len(FEATURE_NAMES),
        "iso_anomaly_threshold": ISO_ANOMALY_THRESHOLD,
        "daily_check_threshold": DAILY_CHECK_THRESHOLD,
        "decision_logic":        "SUSPECT si iso_score>=0.50 OU daily_count>=5",
        "rf_role":               "score affichage uniquement (pondération 40%)",
        "features": [
            {"name": name, "description": descriptions.get(name, "")}
            for name in FEATURE_NAMES
        ],
        "model_version": METRICS.get("model_version", "TraceIMEI-BJ v3.5-RF+IF"),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
