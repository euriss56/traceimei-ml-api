"""
TraceIMEI-BJ — Script d'entraînement v3.0
Génère un dataset synthétique réaliste avec features comportementales
et réentraîne RF + Isolation Forest
"""

import numpy as np
import pandas as pd
import joblib
import json
import random
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

random.seed(42)
np.random.seed(42)

# ─── TAC connus au Bénin ───────────────────────────────────────────────
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

TEST_IMEIS = {"000000000000000","111111111111111","123456789012345","999999999999999"}

def luhn_check(imei):
    if len(imei) != 15 or not imei.isdigit():
        return False
    digits = [int(d) for d in imei]
    total = sum(digits[-1::-2])
    for d in digits[-2::-2]:
        total += sum(divmod(d * 2, 10))
    return total % 10 == 0

def generate_valid_imei(tac=None):
    if tac is None:
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
    check = (10 - (total % 10)) % 10
    return base + str(check)

def generate_fake_imei():
    """IMEI invalide ou suspect"""
    mode = random.choice(["luhn_invalid", "unknown_tac", "sequential", "repeated"])
    if mode == "luhn_invalid":
        tac = random.choice(KNOWN_TAC)
        snr = str(random.randint(0, 999999)).zfill(6)
        bad_check = random.randint(0, 9)
        base = tac + snr
        # Calcule le bon check puis choisit un autre
        digits = [int(d) for d in base]
        total = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d *= 2
                if d > 9: d -= 9
            total += d
        correct = (10 - (total % 10)) % 10
        bad_check = (correct + random.randint(1, 9)) % 10
        return base + str(bad_check)
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
        check = (10 - (total % 10)) % 10
        return base + str(check)
    elif mode == "sequential":
        start = random.randint(1, 8)
        return "".join([str((start + i) % 10) for i in range(15)])
    else:
        d = str(random.randint(0, 9))
        return d * 15

# ─── Feature engineering ──────────────────────────────────────────────

def extract_features(imei, label=None):
    """
    10 features réalistes et variées
    """
    # 1. luhn_valid
    luhn = 1 if luhn_check(imei) else 0

    # 2. tac_known
    tac = imei[:8] if len(imei) >= 8 else ""
    tac_known = 0
    for known in KNOWN_TAC:
        if tac.startswith(known[:6]):
            tac_known = 1
            break

    # 3. all_same_digits
    all_same = 1 if imei and len(set(imei)) == 1 else 0

    # 4. is_test_imei
    is_test = 1 if imei in TEST_IMEIS else 0

    # 5. length_ok
    length_ok = 1 if len(imei) == 15 else 0

    # 6. tac_theft_rate — fréquence de vol du TAC dans la base simulée
    # Certains TAC sont plus souvent volés (Samsung, iPhone = cibles)
    high_value_tac = ["35328004","01326300","35299406","35469208","35607204",  # Apple
                      "35674108","35919004","35821804",  # Samsung haut de gamme
                      "86498904","86732204"]             # Huawei P/Mate
    tac_theft_rate = 0.7 if any(tac.startswith(t[:6]) for t in high_value_tac) else 0.2

    # 7. digit_entropy — diversité des chiffres (IMEI séquentiels ont entropy faible)
    if imei.isdigit() and len(imei) > 0:
        counts = [imei.count(str(d)) / len(imei) for d in range(10)]
        entropy = -sum(p * np.log2(p + 1e-9) for p in counts if p > 0)
        digit_entropy = round(entropy / np.log2(10), 3)  # normalisé 0-1
    else:
        digit_entropy = 0.0

    # 8. check_frequency — combien de fois cet IMEI a été vérifié (simulé)
    # Les IMEI volés sont souvent vérifiés plusieurs fois
    if label == 1:  # volé
        check_freq = random.randint(3, 15) / 15
    elif label == 0:  # légitime
        check_freq = random.randint(1, 3) / 15
    else:
        check_freq = random.uniform(0, 1)

    # 9. verification_hour — heure de vérification normalisée (0-1)
    # Vols souvent signalés la nuit
    if label == 1:
        hour = random.choices(range(24), weights=[
            3,2,2,2,2,2,3,4,5,5,5,5,5,5,5,5,5,5,6,7,8,8,6,4
        ])[0]
    else:
        hour = random.choices(range(24), weights=[
            1,1,1,1,1,1,2,5,8,9,9,9,9,9,9,9,8,7,6,5,4,3,2,1
        ])[0]
    hour_norm = hour / 23

    # 10. days_since_first_seen — ancienneté dans le système (simulé)
    if label == 1:
        days = random.randint(0, 30) / 365  # récent = plus suspect
    else:
        days = random.randint(30, 730) / 365
    
    return [luhn, tac_known, all_same, is_test, length_ok,
            tac_theft_rate, digit_entropy, check_freq, hour_norm, days]

# ─── Génération dataset ────────────────────────────────────────────────

print("📊 Génération du dataset...")

records = []

# Légitimes (label=0) — 60%
for _ in range(1800):
    imei = generate_valid_imei()
    features = extract_features(imei, label=0)
    records.append(features + [0])

# Volés déclarés (label=1) — 30% : IMEI valides mais signalés
for _ in range(900):
    imei = generate_valid_imei()
    features = extract_features(imei, label=1)
    records.append(features + [1])

# Faux/suspects (label=1) — 10% : IMEI malformés
for _ in range(300):
    imei = generate_fake_imei()
    features = extract_features(imei, label=1)
    records.append(features + [1])

cols = ["luhn","tac_known","all_same","is_test","length_ok",
        "tac_theft_rate","digit_entropy","check_freq","hour_norm","days_seen","label"]

df = pd.DataFrame(records, columns=cols)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset: {len(df)} lignes | Légitimes: {(df.label==0).sum()} | Volés: {(df.label==1).sum()}")

X = df.drop("label", axis=1).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ─── Entraînement RF ──────────────────────────────────────────────────
print("\n🌲 Entraînement Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC RF: {auc:.4f}")
print(classification_report(y_test, y_pred, target_names=["légitime","volé"]))

# ─── Entraînement Isolation Forest ───────────────────────────────────
print("🔍 Entraînement Isolation Forest...")
iso = IsolationForest(
    n_estimators=200,
    contamination=0.4,
    random_state=42,
    n_jobs=-1
)
iso.fit(X_train)

# Calibration des bornes pour normalisation
iso_scores = -iso.decision_function(X_train)
iso_min = float(iso_scores.min())
iso_max = float(iso_scores.max())
print(f"ISO scores: min={iso_min:.3f} max={iso_max:.3f}")

# ─── Sauvegarde ────────────────────────────────────────────────────────
model_bundle = {
    "rf": rf,
    "iso": iso,
    "iso_min": iso_min,
    "iso_max": iso_max,
    "feature_names": cols[:-1],
    "n_features": len(cols) - 1,
    "version": "3.0"
}

joblib.dump(model_bundle, "/home/claude/traceimei_model_v3.pkl")
print("\n✅ Modèle sauvegardé : traceimei_model_v3.pkl")

# ─── Métriques ────────────────────────────────────────────────────────
metrics = {
    "model_version": "TraceIMEI-BJ v3.0-RF+IF",
    "auc_roc": round(auc, 4),
    "n_features": len(cols) - 1,
    "feature_names": cols[:-1],
    "dataset_size": len(df),
    "class_distribution": {"legitime": int((df.label==0).sum()), "vole": int((df.label==1).sum())},
    "data_origin": "synthetic_behavioral_v3",
    "rf_estimators": 200,
    "iso_estimators": 200,
    "iso_contamination": 0.4
}
with open("/home/claude/model_metrics_v3.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print("✅ Métriques sauvegardées : model_metrics_v3.json")

# ─── Test rapide sur quelques IMEI ────────────────────────────────────
print("\n🧪 Test sur IMEI types :")
test_cases = [
    ("352606064431669", "légitime (Luhn ✅, TAC connu)"),
    ("123456789012345", "test IMEI blacklist"),
    ("111111111111111", "all same digits"),
    ("999999999999998", "TAC inconnu"),
]
for imei, desc in test_cases:
    feats = np.array([extract_features(imei)])
    rf_score = rf.predict_proba(feats)[0][1]
    iso_raw = -iso.decision_function(feats)[0]
    iso_norm = float(np.clip((iso_raw - iso_min)/(iso_max - iso_min), 0, 1))
    score = round(0.70 * rf_score + 0.30 * iso_norm, 3)
    status = "VOLÉ" if score >= 0.80 else "SUSPECT" if score >= 0.50 else "LÉGITIME"
    print(f"  {imei} → {score:.3f} [{status}] — {desc}")
