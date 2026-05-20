"""LightGBM Training & Prediction sowie Prophet Preistrend-Prognose.

Gibt bei fehlendem Modell / fehlenden Abhängigkeiten Mock-Daten zurück,
damit das Dashboard sofort lauffähig ist.
"""

from __future__ import annotations

import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
    retailers = ["Konzum", "Kaufland", "Lidl", "Spar", "Bingo", "Tinex", "Maxi", "Voli"]
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


# ── Prophet Preistrend-Prognose ───────────────────────────────────────────────

def forecast_price_trend(
    df: pd.DataFrame,
    product_id: str | None = None,
    periods: int = 30,
) -> tuple[go.Figure, pd.DataFrame]:
    """Trainiert ein Prophet-Modell auf historischen Preisen und prognostiziert den Trend.

    Erkennt automatisch erwartete Preissturz-Punkte und annotiert sie im Chart.
    Fällt auf synthetische Mock-Daten zurück wenn zu wenig Datenpunkte oder
    Prophet nicht installiert ist.

    Args:
        df: Preishistorie-DataFrame. Erwartet Spalten [recorded_at|date, price|price_promo]
            und optional [product_name] für die Filterung.
        product_id: Produktname zum Filtern. None = alle Zeilen verwenden.
        periods: Anzahl Tage für die Zukunftsprognose (Standard: 30).

    Returns:
        Tuple aus (Plotly Figure, forecast-DataFrame mit [ds, yhat, yhat_lower, yhat_upper]).
    """
    work = df.copy()

    # Filtern nach Produkt
    if product_id and "product_name" in work.columns:
        mask = work["product_name"].str.contains(product_id, case=False, na=False)
        work = work[mask]

    # Spalten normieren
    date_col  = next((c for c in ["recorded_at", "date", "valid_from"] if c in work.columns), None)
    price_col = next((c for c in ["price", "price_promo"] if c in work.columns), None)

    if date_col is None or price_col is None:
        return _mock_forecast_figure(product_id or "Produkt", periods)

    work["ds"] = pd.to_datetime(work[date_col], errors="coerce")
    work["y"]  = pd.to_numeric(work[price_col], errors="coerce")

    # Pro Tag aggregieren (Mittelwert über alle Händler)
    prophet_df = (
        work.dropna(subset=["ds", "y"])
        .groupby("ds")["y"]
        .mean()
        .reset_index()
        .sort_values("ds")
    )

    if len(prophet_df) < 5:
        return _mock_forecast_figure(product_id or "Produkt", periods)

    try:
        from prophet import Prophet  # lazy import – optional dependency

        m = Prophet(
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.15,
            interval_width=0.80,
        )
        # Zusätzliche Monatssaisonalität für Aktionszyklen
        m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        m.fit(prophet_df)

        future   = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)

        fig = _build_forecast_figure(prophet_df, forecast, product_id or "Produkt", periods)
        future_fc = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).copy()
        return fig, future_fc

    except Exception:
        return _mock_forecast_figure(product_id or "Produkt", periods)


def _build_forecast_figure(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    title: str,
    periods: int,
) -> go.Figure:
    """Baut einen Plotly-Figure mit historischen Preisen, Prognose und Konfidenzband."""
    today = pd.Timestamp.today().normalize()
    hist_last_price = historical["y"].iloc[-1] if not historical.empty else None

    # Konfidenzband (Fläche zwischen yhat_lower und yhat_upper)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_upper"],
        mode="lines", line=dict(width=0),
        name="Oberes KI", showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"],
        mode="lines", line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(26, 86, 219, 0.12)",
        name="80 % Konfidenzband",
        hoverinfo="skip",
    ))

    # Prognoselinie
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"].round(4),
        mode="lines",
        line=dict(color="#1A56DB", width=2.5, dash="dot"),
        name="Prophet-Prognose",
        hovertemplate="%{x|%d.%m.%Y}<br>Prognose: <b>%{y:.2f} €</b><extra></extra>",
    ))

    # Historische Preise
    fig.add_trace(go.Scatter(
        x=historical["ds"], y=historical["y"].round(4),
        mode="lines+markers",
        line=dict(color="#111827", width=2),
        marker=dict(size=6, color="#111827"),
        name="Historischer Preis",
        hovertemplate="%{x|%d.%m.%Y}<br>Preis: <b>%{y:.2f} €</b><extra></extra>",
    ))

    # "Heute"-Linie
    fig.add_vline(
        x=today.timestamp() * 1000,
        line_dash="dash", line_color="#6B7280", line_width=1.5,
        annotation_text="Heute", annotation_position="top right",
        annotation_font=dict(color="#6B7280", size=11),
    )

    # Erwarteten Preissturz annotieren
    future_only = forecast[forecast["ds"] > today]
    if not future_only.empty and hist_last_price is not None:
        min_idx   = future_only["yhat"].idxmin()
        min_row   = future_only.loc[min_idx]
        drop_pct  = (hist_last_price - min_row["yhat"]) / hist_last_price * 100

        if drop_pct > 3:  # Nur annotieren wenn Sturz > 3%
            fig.add_annotation(
                x=min_row["ds"], y=min_row["yhat"],
                text=f"⬇ Tiefpunkt<br>{min_row['yhat']:.2f} €<br>({drop_pct:+.1f} %)",
                showarrow=True, arrowhead=2,
                arrowcolor="#F05252", font=dict(color="#F05252", size=11),
                bgcolor="rgba(254,226,226,0.9)", bordercolor="#F05252",
                ax=40, ay=-50,
            )
            fig.add_vline(
                x=min_row["ds"].timestamp() * 1000,
                line_dash="dot", line_color="#F05252", line_width=1,
            )

    fig.update_layout(
        title=dict(text=f"🔮 Preisprognose: {title} (nächste {periods} Tage)", font=dict(size=15)),
        xaxis=dict(title="Datum", showgrid=True, gridcolor="#F3F4F6"),
        yaxis=dict(title="Preis (€)", showgrid=True, gridcolor="#F3F4F6"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=70, b=30, l=10, r=10),
    )
    return fig


def _mock_forecast_figure(product_name: str, periods: int) -> tuple[go.Figure, pd.DataFrame]:
    """Generiert einen realistisch aussehenden Mock-Forecast ohne echte Daten."""
    rng   = np.random.default_rng(seed=abs(hash(product_name)) % (2**32))
    today = pd.Timestamp.today().normalize()

    # 60 Tage Vergangenheit simulieren
    past_dates  = pd.date_range(end=today, periods=60, freq="D")
    base_price  = rng.uniform(1.5, 8.0)
    trend       = np.linspace(0, rng.uniform(-0.3, 0.1), 60)
    seasonality = 0.08 * np.sin(np.linspace(0, 4 * np.pi, 60))
    noise       = rng.normal(0, 0.04, 60)
    hist_prices = (base_price + trend + seasonality * base_price + noise * base_price).clip(0.3)

    # Zukunft simulieren: leichter Abfall mit Aktions-Delle
    future_dates  = pd.date_range(start=today + pd.Timedelta(days=1), periods=periods, freq="D")
    future_trend  = np.linspace(0, rng.uniform(-0.25, -0.05), periods)
    future_season = 0.10 * np.sin(np.linspace(0, 2 * np.pi, periods))
    yhat          = (hist_prices[-1] + future_trend + future_season * base_price).clip(0.3)
    yhat_lower    = (yhat - 0.08 * base_price).clip(0.1)
    yhat_upper    = (yhat + 0.08 * base_price)

    historical = pd.DataFrame({"ds": past_dates, "y": hist_prices})
    all_ds     = pd.concat([
        pd.Series(past_dates),
        pd.Series(future_dates),
    ]).reset_index(drop=True)
    all_yhat = np.concatenate([hist_prices, yhat])
    all_lower = np.concatenate([hist_prices - 0.06 * base_price, yhat_lower])
    all_upper = np.concatenate([hist_prices + 0.06 * base_price, yhat_upper])

    full_fc = pd.DataFrame({
        "ds": pd.concat([pd.Series(past_dates), pd.Series(future_dates)]).values,
        "yhat": all_yhat, "yhat_lower": all_lower, "yhat_upper": all_upper,
    })

    fig = _build_forecast_figure(historical, full_fc, f"{product_name} (Demo)", periods)
    future_fc = pd.DataFrame({
        "ds": future_dates, "yhat": yhat.round(4),
        "yhat_lower": yhat_lower.round(4), "yhat_upper": yhat_upper.round(4),
    })
    return fig, future_fc
