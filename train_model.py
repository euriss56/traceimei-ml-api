"""
TraceIMEI-BJ — Script d'entraînement v2.2
Features : luhn, tac_match, all_same_digits, is_test_imei, length_ok
Moteur   : Random Forest 70% + Isolation Forest 30%
Auteur   : Euriss FANOU & Thierry MEHOUNOU — GETECH 2026
"""

import numpy as np
import joblib
import json
import random
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score
)

# ────────────────────────────────────────────────────────────
# SEED
# ────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ────────────────────────────────────────────────────────────
# GÉNÉRATION DES DONNÉES SYNTHÉTIQUES
# Features : luhn, tac_match, all_same, is_test, length_ok
# ────────────────────────────────────────────────────────────

def generate_dataset(n_legit=3000, n_stolen=1500, n_suspect=1000):
    """
    Génère un dataset synthétique réaliste pour 3 classes :
    - légitime  (label 0)
    - suspect   (label 1, score 0.50-0.79)
    - volé      (label 1, score >= 0.80)
    """
    X, y = [], []

    # ── IMEI LÉGITIMES ──────────────────────────────────────
    for _ in range(n_legit):
        X.append([
            1,                        # luhn_valid       : toujours OK
            random.choice([1, 1, 0]), # tac_match        : 67% reconnu
            0,                        # all_same_digits  : jamais
            0,                        # is_test_imei     : jamais
            1,                        # length_ok        : toujours 15 chiffres
        ])
        y.append(0)

    # ── IMEI VOLÉS ──────────────────────────────────────────
    for _ in range(n_stolen):
        X.append([
            random.choice([1, 0, 0]), # luhn_valid       : souvent invalide
            random.choice([0, 0, 1]), # tac_match        : souvent inconnu
            random.choice([0, 1]),    # all_same_digits  : parfois
            random.choice([0, 1]),    # is_test_imei     : parfois
            random.choice([1, 0]),    # length_ok        : parfois incorrect
        ])
        y.append(1)

    # ── IMEI SUSPECTS ───────────────────────────────────────
    for _ in range(n_suspect):
        X.append([
            1,                        # luhn_valid       : valide
            random.choice([0, 1]),    # tac_match        : incertain
            0,                        # all_same_digits  : non
            0,                        # is_test_imei     : non
            1,                        # length_ok        : OK
        ])
        y.append(random.choice([0, 1]))

    return np.array(X), np.array(y)


print("📊 Génération du dataset synthétique...")
X, y = generate_dataset()
print(f"   Total : {len(X)} échantillons")
print(f"   Légitimes : {sum(y==0)} | Suspects/Volés : {sum(y==1)}")

# ────────────────────────────────────────────────────────────
# ENTRAÎNEMENT RANDOM FOREST
# ────────────────────────────────────────────────────────────

print("\n🌲 Entraînement Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)
rf.fit(X, y)

# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
print(f"   CV AUC-ROC : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Métriques sur tout le dataset
y_pred  = rf.predict(X)
y_proba = rf.predict_proba(X)[:, 1]

auc       = roc_auc_score(y, y_proba)
precision = precision_score(y, y_pred)
recall    = recall_score(y, y_pred)
f1        = f1_score(y, y_pred)

print(f"   AUC-ROC   : {auc:.3f}")
print(f"   Precision : {precision:.3f}")
print(f"   Recall    : {recall:.3f}")
print(f"   F1-Score  : {f1:.3f}")

# Importance des features
feature_names = ['luhn', 'tac_match', 'all_same', 'is_test', 'length_ok']
importances   = rf.feature_importances_
print("\n📌 Importance des features :")
for name, imp in zip(feature_names, importances):
    bar = "█" * int(imp * 40)
    print(f"   {name:<15} {imp:.3f}  {bar}")

# ────────────────────────────────────────────────────────────
# ENTRAÎNEMENT ISOLATION FOREST
# ────────────────────────────────────────────────────────────

print("\n🔍 Entraînement Isolation Forest...")
iso = IsolationForest(
    n_estimators=200,
    contamination=0.25,
    random_state=42
)
iso.fit(X)

# Calibration min/max pour normalisation
iso_scores = -iso.decision_function(X)
iso_min    = float(iso_scores.min())
iso_max    = float(iso_scores.max())
print(f"   Score min : {iso_min:.4f}")
print(f"   Score max : {iso_max:.4f}")

# ────────────────────────────────────────────────────────────
# TEST DU SCORE ENSEMBLISTE
# ────────────────────────────────────────────────────────────

print("\n🧪 Test du score ensembliste RF 70% + IF 30% ...")

test_cases = [
    # (description, features)
    ("IMEI légitime Samsung",   [1, 1, 0, 0, 1]),
    ("IMEI TAC inconnu",        [1, 0, 0, 0, 1]),
    ("IMEI Luhn invalide",      [0, 0, 0, 0, 1]),
    ("IMEI test connu",         [1, 0, 0, 1, 1]),
    ("IMEI tous mêmes chiffres",[1, 0, 1, 0, 1]),
    ("IMEI longueur incorrecte",[0, 0, 0, 0, 0]),
]

for desc, feats in test_cases:
    X_test    = np.array([feats])
    rf_proba  = rf.predict_proba(X_test)[0][1]
    iso_raw   = -iso.decision_function(X_test)[0]
    iso_norm  = float(np.clip((iso_raw - iso_min) / (iso_max - iso_min), 0, 1))
    score     = round(0.70 * rf_proba + 0.30 * iso_norm, 3)
    status    = "🔴 VOLÉ" if score >= 0.80 else "🟡 SUSPECT" if score >= 0.50 else "🟢 LÉGITIME"
    print(f"   {desc:<35} score={score:.3f}  {status}")

# ────────────────────────────────────────────────────────────
# SAUVEGARDE DU MODÈLE
# ────────────────────────────────────────────────────────────

model_bundle = {
    'rf':      rf,
    'iso':     iso,
    'iso_min': iso_min,
    'iso_max': iso_max,
    'features': feature_names,
    'n_features': 5,
    'version': '2.2'
}

joblib.dump(model_bundle, 'traceimei_model.pkl')
print("\n✅ Modèle sauvegardé : traceimei_model.pkl")

# ────────────────────────────────────────────────────────────
# SAUVEGARDE DES MÉTRIQUES
# ────────────────────────────────────────────────────────────

metrics = {
    "model_version": "TraceIMEI-BJ v2.2-RF",
    "data_origin":   "synthetic",
    "n_features":    5,
    "features":      feature_names,
    "metrics_synthetic": {
        "auc_roc":   round(auc, 3),
        "cv_mean":   round(float(cv_scores.mean()), 3),
        "cv_std":    round(float(cv_scores.std()), 3),
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
    },
    "metrics_terrain": {
        "auc_roc": None,
        "status":  "calibration_terrain_en_cours"
    }
}

with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("✅ Métriques sauvegardées : model_metrics.json")
print("\n🚀 Entraînement terminé — v2.2 prête à déployer !")
