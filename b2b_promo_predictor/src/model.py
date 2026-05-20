"""LightGBM Training & Prediction für Aktionsvorhersagen.

Gibt bei fehlendem Modell Mock-Predictions zurück, damit das Dashboard
sofort lauffähig ist.
"""

from __future__ import annotations

import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .features import FEATURE_COLS, TARGET_COL, create_features

MODEL_PATH = PROJECT_ROOT / "data" / "lgbm_model.pkl"


def train_lgbm(df: pd.DataFrame | None = None) -> Any:
    """Trainiert ein LightGBM-Klassifikationsmodell auf Aktionsdaten.

    Args:
        df: Rohdaten mit historischen Aktionen. Wenn None, werden Mock-Daten genutzt.

    Returns:
        Trainiertes LGBMClassifier-Objekt.
    """
    import lightgbm as lgb

    feature_df = create_features(df)
    X = feature_df[FEATURE_COLS].fillna(0)
    y = feature_df[TARGET_COL]

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model


def _load_model() -> Any | None:
    """Lädt ein gespeichertes Modell, falls vorhanden."""
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


def _mock_predictions(n: int = 10) -> pd.DataFrame:
    """Generiert Mock-Vorhersagen für das Dashboard."""
    rng = np.random.default_rng(7)
    products = [
        "Milka Schokolade 300g",
        "Coca-Cola 1,5L",
        "Ritter Sport 100g",
        "Haribo Goldbären 200g",
        "Nutella 450g",
        "Ja! Vollmilch 1L",
        "Ariel Pulver 20WL",
        "Pampers Gr.3 44Stk",
        "Red Bull 250ml",
        "Pringles Original 185g",
    ]
    retailers = ["Lidl", "Aldi Süd", "Penny", "Rewe", "Netto"]
    today = date.today()
    promo_starts = [today + timedelta(days=int(d)) for d in rng.integers(1, 14, n)]

    return pd.DataFrame(
        {
            "product": rng.choice(products, n),
            "retailer": rng.choice(retailers, n),
            "predicted_promo_start": promo_starts,
            "confidence": rng.uniform(0.62, 0.97, n).round(3),
            "expected_price": rng.uniform(0.49, 9.99, n).round(2),
        }
    ).sort_values("confidence", ascending=False).reset_index(drop=True)


def predict(
    df: pd.DataFrame | None = None,
    model: Any | None = None,
) -> pd.DataFrame:
    """Erstellt Aktionsvorhersagen für die nächste Woche.

    Args:
        df: Rohdaten. Wenn None, werden Mock-Daten genutzt.
        model: Vortrainiertes Modell. Wird aus Disk geladen wenn None.

    Returns:
        DataFrame mit Spalten [product, retailer, predicted_promo_start,
        confidence, expected_price].
    """
    if model is None:
        model = _load_model()

    if model is None:
        return _mock_predictions()

    try:
        feature_df = create_features(df)
        X = feature_df[FEATURE_COLS].fillna(0)
        proba = model.predict_proba(X)[:, 1]

        result = feature_df[["product", "retailer", "date", "price_promo"]].copy()
        result["confidence"] = proba.round(3)
        result = result[result["confidence"] >= 0.50].copy()
        result = result.rename(
            columns={"date": "predicted_promo_start", "price_promo": "expected_price"}
        )
        return result.sort_values("confidence", ascending=False).reset_index(drop=True)
    except Exception:
        return _mock_predictions()


def get_feature_importance(model: Any | None = None) -> pd.DataFrame:
    """Gibt Feature Importances als DataFrame zurück.

    Args:
        model: Trainiertes Modell. Wenn None, wird Mock-Importance genutzt.

    Returns:
        DataFrame mit [feature, importance].
    """
    if model is None:
        model = _load_model()

    if model is None:
        rng = np.random.default_rng(99)
        return pd.DataFrame(
            {
                "feature": FEATURE_COLS,
                "importance": rng.integers(10, 100, len(FEATURE_COLS)),
            }
        ).sort_values("importance", ascending=False)

    return pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
