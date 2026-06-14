{
  "model_version": "TraceIMEI-BJ v3.0",
  "data_origin": "synthetic",
  "n_features": 6,
  "features": [
    "luhn_valid",
    "tac_known",
    "tac_match",
    "all_same",
    "is_test",
    "length_ok"
  ],
  "seuils": {
    "vole": 0.8,
    "suspect": 0.6,
    "grey_low": 0.55,
    "grey_high": 0.7
  },
  "metrics_synthetic": {
    "auc_roc": 0.913,
    "cv_mean": 0.887,
    "cv_std": 0.167,
    "precision": 0.909,
    "recall": 0.774,
    "f1": 0.836,
    "faux_positifs_rate": 0.045,
    "WARNING": "Ces métriques sont sur données synthétiques. Ne pas utiliser pour valider la performance réelle."
  },
  "metrics_terrain": {
    "auc_roc": null,
    "n_samples": 0,
    "status": "collecte_en_cours",
    "instruction": "Chaque cas low_confidence doit être enregistré dans la table imei_terrain_labels (Supabase) avec le label validé manuellement."
  },
  "fixes_v3": [
    "contamination: 0.25 → 0.05",
    "tac séparé: tac_known + tac_match (6 features)",
    "seuil suspect: 0.50 → 0.60",
    "zone grise low_confidence: 0.55–0.70"
  ]
}
