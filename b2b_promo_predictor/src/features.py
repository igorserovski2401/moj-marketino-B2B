"""Feature Engineering für das ML-Vorhersagemodell.

Transformiert rohe Aktionsdaten in ML-ready Features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_mock_history(n_rows: int = 300) -> pd.DataFrame:
    """Generiert synthetische historische Aktionsdaten für Demo/Tests."""
    rng = np.random.default_rng(42)
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
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_rows, freq="3D")

    return pd.DataFrame(
        {
            "date": rng.choice(dates, n_rows),
            "product": rng.choice(products, n_rows),
            "retailer": rng.choice(retailers, n_rows),
            "price_promo": rng.uniform(0.49, 12.99, n_rows).round(2),
            "price_regular": rng.uniform(1.0, 15.99, n_rows).round(2),
            "is_on_promo": rng.integers(0, 2, n_rows),
        }
    ).sort_values("date").reset_index(drop=True)


def create_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Erzeugt ML-Features aus historischen Aktionsdaten.

    Features:
        - month: Monat (1–12)
        - week_of_year: Kalenderwoche
        - day_of_week: Wochentag (0=Mo)
        - days_since_last_promo: Tage seit letzter Aktion pro Produkt/Retailer
        - promo_count_last_4w: Aktionsanzahl der letzten 4 Wochen
        - discount_depth: Relativer Rabatt (1 - promo/regular)
        - is_on_promo_next_week: Zielvariable (0/1)

    Args:
        df: DataFrame mit Spalten [date, product, retailer, price_promo,
            price_regular, is_on_promo]. Wenn None, werden Mock-Daten genutzt.

    Returns:
        DataFrame mit Original-Spalten + neuen Feature-Spalten + Zielvariable.
    """
    if df is None:
        df = _make_mock_history()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["product", "retailer", "date"])

    # Kalender-Features
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek

    # Rabatttiefe
    df["discount_depth"] = (
        1 - df["price_promo"] / df["price_regular"].replace(0, np.nan)
    ).clip(0, 1).round(4)

    # Days since last promo (per product+retailer)
    group_key = ["product", "retailer"]
    promo_dates = df[df["is_on_promo"] == 1][["date"] + group_key]

    def _days_since_last(sub: pd.DataFrame) -> pd.Series:
        sub = sub.sort_values("date")
        sub["last_promo"] = sub["date"].where(sub["is_on_promo"] == 1).ffill()
        return (sub["date"] - sub["last_promo"]).dt.days.fillna(999)

    df["days_since_last_promo"] = (
        df.groupby(group_key, group_keys=False).apply(_days_since_last)
    )

    # Aktionsanzahl letzte 4 Wochen (rolling, pro Gruppe)
    df_indexed = df.set_index("date")

    def _rolling_promo_count(sub: pd.DataFrame) -> pd.Series:
        return (
            sub["is_on_promo"]
            .rolling("28D", min_periods=1)
            .sum()
            .rename("promo_count_last_4w")
        )

    df["promo_count_last_4w"] = (
        df_indexed.groupby(group_key, group_keys=False)
        .apply(_rolling_promo_count)
        .values
    )

    # Zielvariable: War das Produkt in der folgenden Woche auf Aktion?
    df["is_on_promo_next_week"] = (
        df.groupby(group_key)["is_on_promo"]
        .shift(-1)
        .fillna(0)
        .astype(int)
    )

    return df.reset_index(drop=True)


FEATURE_COLS = [
    "month",
    "week_of_year",
    "day_of_week",
    "discount_depth",
    "days_since_last_promo",
    "promo_count_last_4w",
]
TARGET_COL = "is_on_promo_next_week"
