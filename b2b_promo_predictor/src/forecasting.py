"""KAM-taugliche Promo-Vorhersage auf Basis historischer Aktionsdaten.

Regelbasierte Heuristik als stabile Basis. Eine Vorhersage darf NUR angezeigt
werden, wenn ausreichend Historie vorhanden ist (≥3 Aktionen, ≥90 Tage Spanne)
und der vorhergesagte Start in der Zukunft liegt. Andernfalls Status-Tags
machen den Mangel sichtbar – statt unbegründbar hohe Wahrscheinlichkeiten anzuzeigen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

# ── Konstanten ────────────────────────────────────────────────────────────────

MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER: int = 3
MIN_HISTORY_DAYS: int = 90
MIN_OBSERVATIONS: int = 12
MAX_OUTLIER_FACTOR: float = 3.0
PROMO_TREND_THRESHOLD_PCT: float = 5.0

PriceTag = Literal[
    "valid_price",
    "outlier_high",
    "outlier_low",
    "invalid_price",
    "mixed_currency",
    "mixed_product",
    "mixed_retailer",
]

PredictionStatus = Literal[
    "ok",
    "insufficient_history",
    "invalid_past_prediction",
    "no_data",
]

PriceTrend = Literal["steigend", "fallend", "stabil", "unbekannt"]
Signal = Literal[
    "Hoch relevant", "Beobachten", "Normal", "Nicht belastbar", "Ungültig"
]
Priority = Literal["Hoch", "Mittel", "Niedrig"]


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class PromoForecast:
    """Vollständige KAM-Prognose für eine Produkt-Händler-Kombination."""

    product: str
    brand: str
    retailer: str
    country: str
    category: str

    expected_start: date | None = None
    expected_end: date | None = None
    probability: float = 0.0

    expected_price: float | None = None
    last_promo_price: float | None = None
    avg_promo_price_90d: float | None = None
    avg_promo_price_180d: float | None = None
    min_promo_price_12m: float | None = None
    max_promo_price_12m: float | None = None

    price_trend: PriceTrend = "unbekannt"
    price_trend_pct: float = 0.0
    price_change_vs_last_eur: float | None = None
    price_change_vs_last_pct: float | None = None

    avg_cycle_days: int | None = None
    median_cycle_days: int | None = None
    days_since_last_promo: int | None = None
    typical_duration_days: int | None = None
    typical_discount_pct_min: float | None = None
    typical_discount_pct_max: float | None = None

    historical_count: int = 0
    last_promos: list[dict] = field(default_factory=list)

    prediction_status: PredictionStatus = "no_data"
    signal: Signal = "Nicht belastbar"
    priority: Priority = "Niedrig"
    justification: str = ""


# ── Pure helpers ──────────────────────────────────────────────────────────────

def calculate_price_trend(prices: list[float]) -> tuple[PriceTrend, float]:
    """Erkennt Preistrend aus Promo-Preisen (letzte 3 vs vorherige 3).

    Returns:
        (Trend-Label, Änderung in Prozent)
    """
    if len(prices) < 2:
        return "unbekannt", 0.0

    if len(prices) >= 6:
        recent = float(np.mean(prices[-3:]))
        older = float(np.mean(prices[-6:-3]))
    elif len(prices) >= 4:
        recent = float(np.mean(prices[-2:]))
        older = float(np.mean(prices[:2]))
    else:
        recent = float(prices[-1])
        older = float(prices[0])

    if older <= 0:
        return "unbekannt", 0.0

    pct = (recent - older) / older * 100
    if pct > PROMO_TREND_THRESHOLD_PCT:
        return "steigend", round(pct, 1)
    if pct < -PROMO_TREND_THRESHOLD_PCT:
        return "fallend", round(pct, 1)
    return "stabil", round(pct, 1)


def compute_cycle_days(starts: list[date]) -> tuple[int | None, int | None]:
    """Berechnet Ø und Median des Aktionszyklus in Tagen."""
    if len(starts) < 2:
        return None, None
    sorted_starts = sorted(starts)
    diffs = [
        (sorted_starts[i + 1] - sorted_starts[i]).days
        for i in range(len(sorted_starts) - 1)
    ]
    if not diffs:
        return None, None
    return int(round(np.mean(diffs))), int(round(np.median(diffs)))


def compute_signal(
    status: PredictionStatus,
    probability: float,
    expected_start: date | None,
    avg_cycle: int | None,
    days_since_last: int | None,
    price_trend: PriceTrend,
) -> Signal:
    """KAM-Signal aus den Forecast-Eigenschaften ableiten."""
    if status in ("insufficient_history", "no_data"):
        return "Nicht belastbar"

    today = date.today()
    if expected_start is not None and expected_start < today:
        return "Ungültig"

    is_overdue = bool(
        avg_cycle is not None
        and days_since_last is not None
        and days_since_last > avg_cycle * 1.2
    )

    if probability >= 0.80 and is_overdue:
        return "Hoch relevant"
    if probability >= 0.65:
        return "Beobachten"
    return "Normal"


def compute_priority(signal: Signal) -> Priority:
    """Mappt Signal → Priorität für die Sortierung."""
    if signal == "Hoch relevant":
        return "Hoch"
    if signal == "Beobachten":
        return "Mittel"
    return "Niedrig"


def build_justification(
    avg_cycle: int | None,
    days_since_last: int | None,
    historical_count: int,
    price_trend: PriceTrend,
    price_trend_pct: float,
) -> str:
    """Menschenlesbare Begründung für die Vorhersage."""
    parts: list[str] = []

    if avg_cycle and days_since_last is not None:
        if days_since_last > avg_cycle * 1.2:
            parts.append(
                f"Letzte Aktion vor {days_since_last} Tagen – "
                f"über dem Ø-Zyklus von {avg_cycle} Tagen (überfällig)."
            )
        elif days_since_last < avg_cycle * 0.5:
            parts.append(
                f"Letzte Aktion vor {days_since_last} Tagen – "
                f"unter dem Ø-Zyklus von {avg_cycle} Tagen."
            )
        else:
            parts.append(
                f"Letzte Aktion vor {days_since_last} Tagen, "
                f"Ø-Zyklus {avg_cycle} Tage."
            )

    if historical_count > 0:
        parts.append(f"Basis: {historical_count} historische Aktionen.")

    if price_trend != "unbekannt" and abs(price_trend_pct) > 0.5:
        if price_trend == "steigend":
            parts.append(f"Promo-Preise zuletzt steigend ({price_trend_pct:+.1f} %).")
        elif price_trend == "fallend":
            parts.append(f"Promo-Preise zuletzt fallend ({price_trend_pct:+.1f} %).")
        else:
            parts.append(f"Promo-Preise stabil ({price_trend_pct:+.1f} %).")

    if not parts:
        return "Keine belastbare Begründung verfügbar."
    return " ".join(parts)


def compute_probability(
    historical_count: int,
    avg_cycle_days: int | None,
    days_since_last: int | None,
) -> float:
    """Heuristische Wahrscheinlichkeit aus Historien-Tiefe + Zyklus-Position."""
    if historical_count < MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER:
        return 0.0

    base = min(0.45 + (historical_count - 3) * 0.05, 0.75)

    if avg_cycle_days and days_since_last is not None and avg_cycle_days > 0:
        ratio = days_since_last / avg_cycle_days
        if 0.8 <= ratio <= 1.3:
            base += 0.15  # Sweet spot
        elif ratio > 1.3:
            base += 0.10  # Überfällig
        elif ratio < 0.5:
            base -= 0.20  # Zu früh

    return round(max(0.0, min(0.95, base)), 2)


# ── Forecast-Engine ───────────────────────────────────────────────────────────

def build_forecast_from_history(
    product: str,
    brand: str,
    retailer: str,
    country: str,
    category: str,
    history: pd.DataFrame,
) -> PromoForecast:
    """Erstellt eine PromoForecast aus historischen Aktionsdaten.

    Args:
        history: Aktionen für genau diese Produkt-Händler-Kombi, beliebige Sortierung.
                 Erwartet Spalten [valid_from, valid_until, price_eur, original_price_eur,
                 discount_pct].
    """
    fc = PromoForecast(
        product=product,
        brand=brand,
        retailer=retailer,
        country=country,
        category=category,
    )

    if history is None or history.empty:
        fc.prediction_status = "no_data"
        fc.justification = "Keine historischen Daten verfügbar."
        return fc

    # Spalten finden / fallback
    p_col = "price_eur" if "price_eur" in history.columns else "price"

    valid = history.dropna(subset=["valid_from", p_col]).copy()
    if valid.empty:
        fc.prediction_status = "no_data"
        fc.justification = "Keine Aktionen mit gültigem Datum + Preis."
        return fc

    valid["valid_from"] = pd.to_datetime(valid["valid_from"], errors="coerce")
    valid = valid.dropna(subset=["valid_from"]).sort_values("valid_from")
    fc.historical_count = len(valid)

    # ── Mindest-Anforderungen ────────────────────────────────────────────────
    if fc.historical_count < MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER:
        fc.prediction_status = "insufficient_history"
        fc.justification = (
            f"Nur {fc.historical_count} historische Aktion(en) – "
            f"min. {MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER} erforderlich."
        )
        return fc

    history_span = (valid["valid_from"].iloc[-1] - valid["valid_from"].iloc[0]).days
    if history_span < MIN_HISTORY_DAYS:
        fc.prediction_status = "insufficient_history"
        fc.justification = (
            f"Historie umfasst nur {history_span} Tage – "
            f"min. {MIN_HISTORY_DAYS} Tage erforderlich."
        )
        return fc

    # ── Zyklus-Statistik ─────────────────────────────────────────────────────
    starts = valid["valid_from"].dt.date.tolist()
    fc.avg_cycle_days, fc.median_cycle_days = compute_cycle_days(starts)

    last_start = valid["valid_from"].iloc[-1].date()
    today = date.today()
    fc.days_since_last_promo = (today - last_start).days

    # Typische Aktionsdauer
    if "valid_until" in valid.columns:
        durations = (
            pd.to_datetime(valid["valid_until"], errors="coerce") - valid["valid_from"]
        ).dropna()
        if len(durations) > 0:
            fc.typical_duration_days = int(round(durations.dt.days.median()))

    # ── Erwarteter Start (Zyklus + Guard gegen Vergangenheits-Prognose) ──────
    cycle = fc.median_cycle_days or fc.avg_cycle_days or 30
    expected_start_dt = last_start + timedelta(days=cycle)

    if expected_start_dt < today:
        # Überfällig → Verschiebung auf "demnächst" mit kleinem Buffer
        delay_window = max(1, min(cycle // 4, 14))
        expected_start_dt = today + timedelta(days=delay_window)

    fc.expected_start = expected_start_dt
    if fc.typical_duration_days:
        fc.expected_end = expected_start_dt + timedelta(days=fc.typical_duration_days)

    # ── Preis-Statistik ──────────────────────────────────────────────────────
    prices = valid[p_col].dropna().astype(float).tolist()
    if prices:
        fc.last_promo_price = round(prices[-1], 2)

        cutoff_90  = pd.Timestamp(today - timedelta(days=90))
        cutoff_180 = pd.Timestamp(today - timedelta(days=180))
        cutoff_365 = pd.Timestamp(today - timedelta(days=365))

        prices_90  = valid[valid["valid_from"] >= cutoff_90][p_col].dropna()
        prices_180 = valid[valid["valid_from"] >= cutoff_180][p_col].dropna()
        prices_365 = valid[valid["valid_from"] >= cutoff_365][p_col].dropna()

        if len(prices_90) > 0:
            fc.avg_promo_price_90d = round(float(prices_90.mean()), 2)
        if len(prices_180) > 0:
            fc.avg_promo_price_180d = round(float(prices_180.mean()), 2)
        if len(prices_365) > 0:
            fc.min_promo_price_12m = round(float(prices_365.min()), 2)
            fc.max_promo_price_12m = round(float(prices_365.max()), 2)

        # Trend
        fc.price_trend, fc.price_trend_pct = calculate_price_trend(prices)

        # Erwarteter Preis = trendangepasste 90d-Basis
        base = fc.avg_promo_price_90d or fc.last_promo_price or float(np.mean(prices))
        adj  = (fc.price_trend_pct / 100.0) / 2.0
        fc.expected_price = round(base * (1 + adj), 2)

        if fc.last_promo_price is not None and fc.expected_price is not None:
            fc.price_change_vs_last_eur = round(
                fc.expected_price - fc.last_promo_price, 2
            )
            if fc.last_promo_price > 0:
                fc.price_change_vs_last_pct = round(
                    (fc.expected_price - fc.last_promo_price) / fc.last_promo_price * 100,
                    1,
                )

    # ── Rabatt-Statistik ─────────────────────────────────────────────────────
    if "discount_pct" in valid.columns:
        discounts = valid["discount_pct"].dropna()
        if len(discounts) > 0:
            fc.typical_discount_pct_min = round(float(discounts.quantile(0.25)), 1)
            fc.typical_discount_pct_max = round(float(discounts.quantile(0.75)), 1)

    # ── Historie für UI (letzte 5 Aktionen, neueste zuerst) ─────────────────
    last_n = valid.tail(5).iloc[::-1]
    for _, row in last_n.iterrows():
        fc.last_promos.append({
            "start": row["valid_from"].date(),
            "end": (
                pd.to_datetime(row["valid_until"], errors="coerce").date()
                if "valid_until" in row.index and pd.notna(row["valid_until"])
                else None
            ),
            "price": round(float(row[p_col]), 2) if pd.notna(row[p_col]) else None,
            "discount_pct": (
                round(float(row["discount_pct"]), 1)
                if "discount_pct" in row.index and pd.notna(row["discount_pct"])
                else None
            ),
        })

    # ── Status, Signal, Priorität, Begründung ────────────────────────────────
    fc.prediction_status = "ok"
    fc.probability = compute_probability(
        fc.historical_count, fc.avg_cycle_days, fc.days_since_last_promo
    )
    fc.signal = compute_signal(
        fc.prediction_status,
        fc.probability,
        fc.expected_start,
        fc.avg_cycle_days,
        fc.days_since_last_promo,
        fc.price_trend,
    )
    fc.priority = compute_priority(fc.signal)
    fc.justification = build_justification(
        fc.avg_cycle_days,
        fc.days_since_last_promo,
        fc.historical_count,
        fc.price_trend,
        fc.price_trend_pct,
    )
    return fc


def build_forecasts_from_promo_history(
    promos_df: pd.DataFrame,
    min_historical: int = MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER,
) -> list[PromoForecast]:
    """Erstellt Forecasts für alle (Produkt, Händler)-Kombinationen.

    Filtert Kombinationen unter `min_historical` Aktionen automatisch aus.
    """
    if promos_df is None or promos_df.empty:
        return []

    name_col = "name" if "name" in promos_df.columns else "product_name"
    retailer_col = "store_name" if "store_name" in promos_df.columns else "retailer"

    if name_col not in promos_df.columns or retailer_col not in promos_df.columns:
        return []

    df = promos_df.dropna(subset=[name_col, retailer_col])

    forecasts: list[PromoForecast] = []
    for (product, retailer), group in df.groupby([name_col, retailer_col], dropna=False):
        if len(group) < min_historical:
            continue

        country = ""
        if "country_code" in group.columns:
            mode = group["country_code"].dropna().mode()
            if not mode.empty:
                country = str(mode.iloc[0])

        category = ""
        for cat_col in ("category_de", "category_l1"):
            if cat_col in group.columns:
                mode = group[cat_col].dropna().mode()
                if not mode.empty:
                    category = str(mode.iloc[0])
                    break

        brand = ""
        for brand_col in ("brand", "manufacturer"):
            if brand_col in group.columns:
                mode = group[brand_col].dropna().mode()
                if not mode.empty:
                    brand = str(mode.iloc[0])
                    break

        fc = build_forecast_from_history(
            product=str(product),
            brand=brand,
            retailer=str(retailer),
            country=country,
            category=category,
            history=group,
        )
        forecasts.append(fc)

    return forecasts


def forecasts_to_dataframe(forecasts: list[PromoForecast]) -> pd.DataFrame:
    """Plattet die Forecasts in einen DataFrame für die UI."""
    if not forecasts:
        return pd.DataFrame()

    rows = []
    for fc in forecasts:
        rows.append({
            "priority": fc.priority,
            "product": fc.product,
            "brand": fc.brand,
            "retailer": fc.retailer,
            "country": fc.country,
            "category": fc.category,
            "expected_start": fc.expected_start,
            "expected_end": fc.expected_end,
            "probability": fc.probability,
            "expected_price": fc.expected_price,
            "last_promo_price": fc.last_promo_price,
            "avg_promo_price_90d": fc.avg_promo_price_90d,
            "avg_promo_price_180d": fc.avg_promo_price_180d,
            "min_promo_price_12m": fc.min_promo_price_12m,
            "max_promo_price_12m": fc.max_promo_price_12m,
            "price_trend": fc.price_trend,
            "price_trend_pct": fc.price_trend_pct,
            "price_change_vs_last_eur": fc.price_change_vs_last_eur,
            "price_change_vs_last_pct": fc.price_change_vs_last_pct,
            "avg_cycle_days": fc.avg_cycle_days,
            "median_cycle_days": fc.median_cycle_days,
            "days_since_last_promo": fc.days_since_last_promo,
            "typical_duration_days": fc.typical_duration_days,
            "typical_discount_pct_min": fc.typical_discount_pct_min,
            "typical_discount_pct_max": fc.typical_discount_pct_max,
            "historical_count": fc.historical_count,
            "signal": fc.signal,
            "prediction_status": fc.prediction_status,
            "justification": fc.justification,
            "last_promos": fc.last_promos,
        })
    df = pd.DataFrame(rows)

    # Priorität-Sortierung: Hoch > Mittel > Niedrig, dann Wahrscheinlichkeit
    prio_order = {"Hoch": 0, "Mittel": 1, "Niedrig": 2}
    df["_prio_sort"] = df["priority"].map(prio_order).fillna(3)
    df = df.sort_values(
        by=["_prio_sort", "probability"],
        ascending=[True, False],
    ).drop(columns=["_prio_sort"]).reset_index(drop=True)
    return df


def validate_price_series(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each row with a price quality label.

    Returns the input DataFrame with an added ``price_tag`` column.
    Tags: valid_price / outlier_high / outlier_low / invalid_price /
          mixed_currency / mixed_product / mixed_retailer.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    result = df.copy()
    p_col = "price_eur" if "price_eur" in result.columns else "price"

    # Detect structural issues first
    tags: pd.Series = pd.Series("valid_price", index=result.index, dtype=object)

    if "currency" in result.columns and result["currency"].nunique() > 1:
        tags[:] = "mixed_currency"
        result["price_tag"] = tags
        return result

    name_col = next((c for c in ("name", "product_name") if c in result.columns), None)
    if name_col and result[name_col].nunique() > 1:
        tags[:] = "mixed_product"
        result["price_tag"] = tags
        return result

    retailer_col = next((c for c in ("store_name", "retailer") if c in result.columns), None)
    if retailer_col and result[retailer_col].nunique() > 1:
        tags[:] = "mixed_retailer"
        result["price_tag"] = tags
        return result

    if p_col not in result.columns:
        result["price_tag"] = tags
        return result

    prices = result[p_col].astype(float)
    invalid_mask = prices.isna() | (prices <= 0)
    tags[invalid_mask] = "invalid_price"

    valid_prices = prices[~invalid_mask]
    if len(valid_prices) >= 3:
        median = float(valid_prices.median())
        if median > 0:
            high_mask = (~invalid_mask) & (prices > median * MAX_OUTLIER_FACTOR)
            low_mask  = (~invalid_mask) & (prices < median / MAX_OUTLIER_FACTOR)
            tags[high_mask] = "outlier_high"
            tags[low_mask]  = "outlier_low"

    result["price_tag"] = tags
    return result


def compute_data_quality_score(fc: "PromoForecast") -> float:
    """Return a 0–100 data quality score for a single forecast.

    Scoring components:
    - History depth  (≥12 → 30 pts, linear below)
    - History span   (≥180 days → 20 pts, linear)
    - Price series   (all prices present → 20 pts)
    - Cycle regularity (low CoV → 20 pts)
    - Discount info  (both min/max → 10 pts)
    """
    score = 0.0

    # History depth
    depth_pts = min(30.0, (fc.historical_count / MIN_OBSERVATIONS) * 30.0)
    score += depth_pts

    # History span (use avg_cycle * count as proxy if direct span unavailable)
    if fc.avg_cycle_days and fc.historical_count > 1:
        span = fc.avg_cycle_days * (fc.historical_count - 1)
        score += min(20.0, (span / 180.0) * 20.0)

    # Price series completeness
    if fc.last_promo_price is not None:
        score += 10.0
    if fc.avg_promo_price_90d is not None:
        score += 5.0
    if fc.avg_promo_price_180d is not None:
        score += 5.0

    # Cycle regularity: avg ≈ median → low variance
    if fc.avg_cycle_days and fc.median_cycle_days:
        ratio = abs(fc.avg_cycle_days - fc.median_cycle_days) / max(fc.avg_cycle_days, 1)
        reg_pts = max(0.0, 20.0 * (1.0 - ratio * 2))
        score += reg_pts

    # Discount info
    if fc.typical_discount_pct_min is not None and fc.typical_discount_pct_max is not None:
        score += 10.0

    return round(min(100.0, max(0.0, score)), 1)


def check_forecast_eligibility(
    historical_count: int,
    history_span_days: int,
    outlier_ratio: float = 0.0,
) -> tuple[bool, str]:
    """Gate that a product-retailer combo must pass before a forecast is shown.

    Returns (is_eligible, reason_if_not).
    Uses MIN_OBSERVATIONS (12) as the strict bar for a "reliable" forecast.
    Falls back to MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER (3) for a
    "limited" forecast (shown with a warning but not blocked).
    """
    if historical_count < MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER:
        return False, (
            f"Only {historical_count} historical promotion(s) available – "
            f"minimum {MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER} required."
        )
    if history_span_days < MIN_HISTORY_DAYS:
        return False, (
            f"History spans only {history_span_days} days – "
            f"minimum {MIN_HISTORY_DAYS} days required."
        )
    if outlier_ratio > 0.5:
        return False, (
            f"{outlier_ratio*100:.0f}% of price observations are outliers – "
            "price series too noisy for a reliable forecast."
        )
    return True, ""


def apply_forecast_filters(
    fc_df: pd.DataFrame,
    retailer: str | None = None,
    product_query: str = "",
    min_probability: float = 0.0,
    signal: str | None = None,
    price_trend: str | None = None,
    only_future: bool = True,
    prediction_window_days: int | None = None,
) -> pd.DataFrame:
    """Filterpipeline auf bereits berechneten Forecasts."""
    if fc_df.empty:
        return fc_df

    df = fc_df.copy()

    if only_future:
        today = pd.Timestamp.today().normalize().date()
        df = df[df["expected_start"].apply(lambda d: d is not None and d >= today)]

    if prediction_window_days is not None and prediction_window_days > 0:
        cutoff = date.today() + timedelta(days=prediction_window_days)
        df = df[df["expected_start"].apply(lambda d: d is not None and d <= cutoff)]

    if retailer:
        df = df[df["retailer"] == retailer]

    if product_query:
        q = product_query.strip().lower()
        df = df[
            df["product"].fillna("").str.lower().str.contains(q, regex=False)
            | df["brand"].fillna("").str.lower().str.contains(q, regex=False)
        ]

    if min_probability > 0:
        df = df[df["probability"] >= min_probability]

    if signal and signal != "Alle":
        df = df[df["signal"] == signal]

    if price_trend and price_trend != "Alle":
        df = df[df["price_trend"] == price_trend]

    return df.reset_index(drop=True)
