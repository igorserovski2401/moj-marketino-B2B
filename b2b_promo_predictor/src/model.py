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
    allow_mock: bool = False,
) -> pd.DataFrame:
    """Erstellt Aktionsvorhersagen für die nächste Woche.

    In production mode (allow_mock=False) returns an empty DataFrame when no
    model or no data is available — never silently shows demo retailers.
    """
    if model is None:
        model = _load_model()

    if model is None:
        if allow_mock:
            return _mock_predictions()
        return pd.DataFrame(columns=[
            "product", "retailer", "predicted_promo_start", "confidence", "expected_price",
        ])

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
        if allow_mock:
            return _mock_predictions()
        return pd.DataFrame(columns=[
            "product", "retailer", "predicted_promo_start", "confidence", "expected_price",
        ])


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
    allow_mock: bool = False,
    currency: str = "EUR",
    lang: str = "EN",
    title_prefix: str = "",
) -> tuple[go.Figure, pd.DataFrame]:
    """Prophet-based price trend forecast, with hard production isolation.

    In production mode (allow_mock=False) returns an empty figure when there
    is insufficient data — never falls back to synthetic demo data.
    """
    from .i18n import t

    work = df.copy()

    if product_id and "product_name" in work.columns:
        mask = work["product_name"].str.contains(product_id, case=False, na=False)
        work = work[mask]

    date_col  = next((c for c in ["recorded_at", "date", "valid_from"] if c in work.columns), None)
    price_col = next((c for c in ["price_eur", "price", "price_promo"] if c in work.columns), None)

    if date_col is None or price_col is None:
        if allow_mock:
            return _mock_forecast_figure(product_id or "Product", periods, currency=currency, lang=lang)
        return _empty_forecast(lang, t("forecast.no_data_columns", lang)), pd.DataFrame()

    work["ds"] = pd.to_datetime(work[date_col], errors="coerce")
    work["y"]  = pd.to_numeric(work[price_col], errors="coerce")

    prophet_df = (
        work.dropna(subset=["ds", "y"])
        .groupby("ds")["y"]
        .mean()
        .reset_index()
        .sort_values("ds")
    )

    if len(prophet_df) < 5:
        if allow_mock:
            return _mock_forecast_figure(product_id or "Product", periods, currency=currency, lang=lang)
        return _empty_forecast(lang, t("forecast.too_few_points", lang, n=len(prophet_df))), pd.DataFrame()

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
        m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        m.fit(prophet_df)

        future   = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)

        title = title_prefix or (product_id or "Product")
        fig = _build_forecast_figure(prophet_df, forecast, title, periods, currency=currency, lang=lang)
        future_fc = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).copy()
        return fig, future_fc

    except Exception:
        if allow_mock:
            return _mock_forecast_figure(product_id or "Product", periods, currency=currency, lang=lang)
        return _empty_forecast(lang, t("forecast.prophet_failed", lang)), pd.DataFrame()


def _empty_forecast(lang: str, reason: str) -> go.Figure:
    """Empty placeholder figure shown when production forecast is blocked."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"<b>{reason}</b>",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#6B7280"),
    )
    fig.update_layout(
        margin=dict(t=30, b=30, l=30, r=30),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=300,
    )
    return fig


def compute_backtest_mape(prophet_df: pd.DataFrame, periods: int = 14) -> tuple[float, float, float]:
    """Run an 80/20 backtest on the Prophet model.

    Returns (mape, mae, bias) where mape is in [0, ∞) (e.g. 0.15 = 15% error).
    Returns (NaN, NaN, NaN) on failure or insufficient data.
    """
    nan = float("nan")
    try:
        from prophet import Prophet
    except ImportError:
        return nan, nan, nan

    if len(prophet_df) < 12:
        return nan, nan, nan

    split = int(len(prophet_df) * 0.8)
    train, test = prophet_df.iloc[:split], prophet_df.iloc[split:]
    if len(test) < 1:
        return nan, nan, nan

    try:
        m = Prophet(
            seasonality_mode="multiplicative",
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.15,
            interval_width=0.80,
        )
        m.fit(train)
        future = pd.DataFrame({"ds": test["ds"].values})
        pred = m.predict(future)["yhat"].values
        actual = test["y"].values
        mask = actual > 0
        if mask.sum() < 1:
            return nan, nan, nan
        errs = (pred[mask] - actual[mask]) / actual[mask]
        mape = float(abs(errs).mean())
        mae = float(abs(pred[mask] - actual[mask]).mean())
        bias = float(errs.mean())
        return mape, mae, bias
    except Exception:
        return nan, nan, nan


def _build_forecast_figure(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    title: str,
    periods: int,
    currency: str = "EUR",
    lang: str = "EN",
) -> go.Figure:
    """Build a Plotly figure with historical prices, forecast and confidence band."""
    from .i18n import t, CURRENCY_SYMBOLS

    cur_sym = CURRENCY_SYMBOLS.get(currency, currency)
    today = pd.Timestamp.today().normalize()
    hist_last_price = historical["y"].iloc[-1] if not historical.empty else None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_upper"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(26, 86, 219, 0.12)",
        name=t("forecast.confidence_band", lang),
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"].round(4),
        mode="lines",
        line=dict(color="#1A56DB", width=2.5, dash="dot"),
        name=t("forecast.prophet_forecast", lang),
        hovertemplate=f"%{{x|%d.%m.%Y}}<br>{t('forecast.forecast_label', lang)}: <b>%{{y:.2f}} {cur_sym}</b><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=historical["ds"], y=historical["y"].round(4),
        mode="lines+markers",
        line=dict(color="#111827", width=2),
        marker=dict(size=6, color="#111827"),
        name=t("forecast.historical_price", lang),
        hovertemplate=f"%{{x|%d.%m.%Y}}<br>{t('forecast.price_label', lang)}: <b>%{{y:.2f}} {cur_sym}</b><extra></extra>",
    ))

    fig.add_vline(
        x=today.timestamp() * 1000,
        line_dash="dash", line_color="#6B7280", line_width=1.5,
        annotation_text=t("forecast.today", lang), annotation_position="top right",
        annotation_font=dict(color="#6B7280", size=11),
    )

    # Annotate min point with clear semantics — never a misleading positive "drop"
    future_only = forecast[forecast["ds"] > today]
    if not future_only.empty and hist_last_price is not None and hist_last_price > 0:
        min_idx = future_only["yhat"].idxmin()
        min_row = future_only.loc[min_idx]
        min_price = float(min_row["yhat"])

        if min_price < hist_last_price:
            drop_pct = (hist_last_price - min_price) / hist_last_price * 100
            label = t("forecast.lowest_drop", lang, val=min_price, sym=cur_sym, pct=drop_pct)
            color = "#F05252"
        else:
            rise_pct = (min_price - hist_last_price) / hist_last_price * 100
            label = t("forecast.no_drop_expected", lang, val=min_price, sym=cur_sym, pct=rise_pct)
            color = "#059669"

        fig.add_annotation(
            x=min_row["ds"], y=min_price,
            text=label,
            showarrow=True, arrowhead=2,
            arrowcolor=color, font=dict(color=color, size=11),
            bgcolor="rgba(255,255,255,0.92)", bordercolor=color,
            ax=40, ay=-50,
        )
        fig.add_vline(
            x=min_row["ds"].timestamp() * 1000,
            line_dash="dot", line_color=color, line_width=1,
        )

    fig.update_layout(
        title=dict(
            text=f"🔮 {t('forecast.chart_title', lang, product=title, days=periods)}",
            font=dict(size=15),
        ),
        xaxis=dict(title=t("forecast.axis_date", lang), showgrid=True, gridcolor="#F3F4F6"),
        yaxis=dict(
            title=t("forecast.axis_price", lang, currency=currency),
            showgrid=True, gridcolor="#F3F4F6",
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=70, b=30, l=10, r=10),
    )
    return fig


def _mock_forecast_figure(
    product_name: str,
    periods: int,
    currency: str = "EUR",
    lang: str = "EN",
) -> tuple[go.Figure, pd.DataFrame]:
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

    fig = _build_forecast_figure(
        historical, full_fc, f"{product_name} [DEMO]", periods,
        currency=currency, lang=lang,
    )
    future_fc = pd.DataFrame({
        "ds": future_dates, "yhat": yhat.round(4),
        "yhat_lower": yhat_lower.round(4), "yhat_upper": yhat_upper.round(4),
    })
    return fig, future_fc
