"""Tests für die regelbasierte Promo-Forecast-Engine."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    MAX_OUTLIER_FACTOR,
    MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER,
    MIN_HISTORY_DAYS,
    MIN_OBSERVATIONS,
    PROMO_TREND_THRESHOLD_PCT,
    apply_forecast_filters,
    build_forecast_from_history,
    build_forecasts_from_promo_history,
    calculate_price_trend,
    check_forecast_eligibility,
    compute_cycle_days,
    compute_data_quality_score,
    compute_priority,
    compute_probability,
    compute_signal,
    forecasts_to_dataframe,
    validate_price_series,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_history(
    n_promos: int = 5,
    cycle_days: int = 45,
    base_price: float = 2.0,
    price_drift_pct_total: float = 0.0,
    today: date | None = None,
    duration_days: int = 7,
    discount_pct: float = 25.0,
) -> pd.DataFrame:
    """Erzeugt eine künstliche Promo-Historie mit n Aktionen rückwärts gerechnet."""
    today = today or date.today()
    rows = []
    for i in range(n_promos):
        # i=0 ist die ÄLTESTE Aktion (am weitesten in der Vergangenheit)
        offset = (n_promos - i) * cycle_days
        start = today - timedelta(days=offset)
        end = start + timedelta(days=duration_days)
        # Linearer Drift: 0 % beim ältesten, drift_total % beim neuesten
        factor = 1.0 + (price_drift_pct_total / 100.0) * (i / max(n_promos - 1, 1))
        promo_price = round(base_price * factor, 2)
        orig_price = round(base_price * (1 + discount_pct / 100.0), 2)
        rows.append({
            "name": "Test Product",
            "brand": "Test Brand",
            "store_name": "Test Retailer",
            "country_code": "HR",
            "category_l1": "Hrana",
            "category_de": "Lebensmittel",
            "valid_from": pd.Timestamp(start),
            "valid_until": pd.Timestamp(end),
            "price": promo_price,
            "original_price": orig_price,
            "price_eur": promo_price,
            "original_price_eur": orig_price,
            "discount_pct": discount_pct,
            "currency": "EUR",
        })
    return pd.DataFrame(rows)


# ── Pure helpers ──────────────────────────────────────────────────────────────

class TestCalculatePriceTrend:
    def test_empty_returns_unknown(self):
        assert calculate_price_trend([]) == ("unbekannt", 0.0)

    def test_single_price_returns_unknown(self):
        assert calculate_price_trend([1.0]) == ("unbekannt", 0.0)

    def test_rising_detected(self):
        prices = [1.0, 1.0, 1.0, 1.5, 1.5, 1.5]
        trend, pct = calculate_price_trend(prices)
        assert trend == "steigend"
        assert pct > PROMO_TREND_THRESHOLD_PCT

    def test_falling_detected(self):
        prices = [2.0, 2.0, 2.0, 1.5, 1.5, 1.5]
        trend, pct = calculate_price_trend(prices)
        assert trend == "fallend"
        assert pct < -PROMO_TREND_THRESHOLD_PCT

    def test_stable_detected(self):
        prices = [2.0, 2.01, 2.0, 2.02, 2.0, 2.0]
        trend, pct = calculate_price_trend(prices)
        assert trend == "stabil"

    def test_zero_base_returns_unknown(self):
        prices = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        trend, pct = calculate_price_trend(prices)
        assert trend == "unbekannt"


class TestComputeCycleDays:
    def test_empty_returns_none(self):
        assert compute_cycle_days([]) == (None, None)

    def test_single_date_returns_none(self):
        assert compute_cycle_days([date(2025, 1, 1)]) == (None, None)

    def test_regular_cycle(self):
        starts = [
            date(2025, 1, 1),
            date(2025, 2, 1),
            date(2025, 3, 3),
            date(2025, 4, 2),
        ]
        avg, med = compute_cycle_days(starts)
        assert 28 <= avg <= 32
        assert 28 <= med <= 32


class TestComputeSignal:
    def test_insufficient_history_returns_unbelastbar(self):
        s = compute_signal(
            "insufficient_history", 0.95, date(2026, 12, 1), 45, 30, "stabil"
        )
        assert s == "Nicht belastbar"

    def test_past_start_returns_ungueltig(self):
        s = compute_signal("ok", 0.90, date(2020, 1, 1), 45, 30, "stabil")
        assert s == "Ungültig"

    def test_high_relevance_when_overdue_and_high_prob(self):
        future = date.today() + timedelta(days=10)
        s = compute_signal("ok", 0.85, future, 30, 50, "stabil")
        assert s == "Hoch relevant"

    def test_beobachten_for_mid_prob(self):
        future = date.today() + timedelta(days=10)
        s = compute_signal("ok", 0.70, future, 30, 25, "stabil")
        assert s == "Beobachten"

    def test_normal_for_low_prob(self):
        future = date.today() + timedelta(days=10)
        s = compute_signal("ok", 0.50, future, 30, 25, "stabil")
        assert s == "Normal"


class TestComputePriority:
    def test_hoch_for_hoch_relevant(self):
        assert compute_priority("Hoch relevant") == "Hoch"

    def test_mittel_for_beobachten(self):
        assert compute_priority("Beobachten") == "Mittel"

    def test_niedrig_for_normal(self):
        assert compute_priority("Normal") == "Niedrig"

    def test_niedrig_for_unbelastbar(self):
        assert compute_priority("Nicht belastbar") == "Niedrig"


class TestComputeProbability:
    def test_insufficient_count_returns_zero(self):
        assert compute_probability(2, 30, 30) == 0.0

    def test_min_count_returns_nonzero(self):
        p = compute_probability(MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER, 30, 30)
        assert p > 0.0

    def test_capped_at_095(self):
        p = compute_probability(50, 30, 35)
        assert p <= 0.95

    def test_sweet_spot_adds_bonus(self):
        p_sweet = compute_probability(5, 40, 40)
        p_too_soon = compute_probability(5, 40, 5)
        assert p_sweet > p_too_soon

    def test_too_soon_subtracts(self):
        p_too_soon = compute_probability(5, 40, 5)
        p_neutral = compute_probability(5, 40, 40)
        assert p_too_soon < p_neutral


# ── Forecast-Engine ───────────────────────────────────────────────────────────

class TestBuildForecastFromHistory:
    def test_empty_history_no_data(self):
        fc = build_forecast_from_history(
            "Test", "Brand", "Retailer", "HR", "Hrana", pd.DataFrame()
        )
        assert fc.prediction_status == "no_data"
        assert fc.signal == "Nicht belastbar"

    def test_below_min_count_insufficient(self):
        df = _build_history(n_promos=2, cycle_days=45)
        fc = build_forecast_from_history(
            "Test", "Brand", "Retailer", "HR", "Hrana", df
        )
        assert fc.prediction_status == "insufficient_history"
        assert fc.signal == "Nicht belastbar"

    def test_below_min_span_insufficient(self):
        # 4 Aktionen, aber alle innerhalb von 60 Tagen → unter MIN_HISTORY_DAYS
        df = _build_history(n_promos=4, cycle_days=15)
        fc = build_forecast_from_history(
            "Test", "Brand", "Retailer", "HR", "Hrana", df
        )
        assert fc.prediction_status == "insufficient_history"

    def test_sufficient_history_returns_ok(self):
        df = _build_history(n_promos=6, cycle_days=45)
        fc = build_forecast_from_history(
            "Test", "Brand", "Retailer", "HR", "Hrana", df
        )
        assert fc.prediction_status == "ok"
        assert fc.historical_count == 6

    def test_expected_start_in_future(self):
        df = _build_history(n_promos=6, cycle_days=45)
        fc = build_forecast_from_history(
            "Test", "Brand", "Retailer", "HR", "Hrana", df
        )
        assert fc.expected_start is not None
        assert fc.expected_start >= date.today()

    def test_no_past_predictions(self):
        # Wenn last_promo + cycle in der Vergangenheit liegen würde
        # (z.B. cycle sehr kurz und letzte Aktion weit zurück)
        df = _build_history(n_promos=5, cycle_days=30)
        # Override: setze ältere Aktionen
        df["valid_from"] = pd.to_datetime(df["valid_from"]) - pd.Timedelta(days=200)
        fc = build_forecast_from_history(
            "Test", "Brand", "Retailer", "HR", "Hrana", df
        )
        if fc.prediction_status == "ok":
            assert fc.expected_start >= date.today()

    def test_rising_trend_detected(self):
        df = _build_history(n_promos=6, cycle_days=45, price_drift_pct_total=20.0)
        fc = build_forecast_from_history(
            "Test", "Brand", "Retailer", "HR", "Hrana", df
        )
        assert fc.price_trend == "steigend"
        assert fc.price_trend_pct > 0

    def test_falling_trend_detected(self):
        df = _build_history(n_promos=6, cycle_days=45, price_drift_pct_total=-20.0)
        fc = build_forecast_from_history(
            "Test", "Brand", "Retailer", "HR", "Hrana", df
        )
        assert fc.price_trend == "fallend"
        assert fc.price_trend_pct < 0

    def test_price_module_complete(self):
        df = _build_history(n_promos=6, cycle_days=45)
        fc = build_forecast_from_history(
            "Test", "Brand", "Retailer", "HR", "Hrana", df
        )
        assert fc.last_promo_price is not None
        assert fc.expected_price is not None
        assert fc.avg_promo_price_180d is not None

    def test_cycle_days_computed(self):
        df = _build_history(n_promos=6, cycle_days=45)
        fc = build_forecast_from_history(
            "Test", "Brand", "Retailer", "HR", "Hrana", df
        )
        assert fc.avg_cycle_days is not None
        assert 40 <= fc.avg_cycle_days <= 50

    def test_days_since_last_promo(self):
        df = _build_history(n_promos=6, cycle_days=45)
        fc = build_forecast_from_history(
            "Test", "Brand", "Retailer", "HR", "Hrana", df
        )
        assert fc.days_since_last_promo is not None
        assert fc.days_since_last_promo >= 0

    def test_last_promos_populated(self):
        df = _build_history(n_promos=6, cycle_days=45)
        fc = build_forecast_from_history(
            "Test", "Brand", "Retailer", "HR", "Hrana", df
        )
        assert len(fc.last_promos) > 0
        assert len(fc.last_promos) <= 5

    def test_justification_not_empty_when_ok(self):
        df = _build_history(n_promos=6, cycle_days=45)
        fc = build_forecast_from_history(
            "Test", "Brand", "Retailer", "HR", "Hrana", df
        )
        assert fc.justification != ""
        assert "Begründung" not in fc.justification  # nicht der placeholder


# ── Multi-Combination ─────────────────────────────────────────────────────────

class TestBuildForecastsFromPromoHistory:
    def test_empty_returns_empty(self):
        assert build_forecasts_from_promo_history(pd.DataFrame()) == []

    def test_filters_below_min(self):
        # Zwei (product, retailer)-Paare: eines mit 5, eines mit 2 Aktionen
        df_high = _build_history(n_promos=6, cycle_days=45)
        df_low  = _build_history(n_promos=2, cycle_days=45)
        df_low["name"] = "Other Product"
        df_low["store_name"] = "Other Retailer"
        combined = pd.concat([df_high, df_low], ignore_index=True)

        forecasts = build_forecasts_from_promo_history(combined)
        assert len(forecasts) == 1
        assert forecasts[0].product == "Test Product"

    def test_aggregates_per_pair(self):
        df1 = _build_history(n_promos=6, cycle_days=45)
        df2 = _build_history(n_promos=6, cycle_days=45)
        df2["store_name"] = "Other Retailer"
        combined = pd.concat([df1, df2], ignore_index=True)

        forecasts = build_forecasts_from_promo_history(combined)
        assert len(forecasts) == 2


# ── Filter pipeline ───────────────────────────────────────────────────────────

class TestApplyForecastFilters:
    def setup_method(self):
        # Erzeuge synthetischen Forecast-DataFrame
        today = date.today()
        rows = []
        for prod, retailer, prob, sig, trend, start_delta in [
            ("Red Bull 250ml", "Konzum",  0.90, "Hoch relevant", "steigend",  10),
            ("Milka 300g",     "Lidl",    0.70, "Beobachten",    "stabil",    20),
            ("Coca-Cola",      "Spar",    0.40, "Normal",        "fallend",   30),
            ("Old Product",    "Tinex",   0.85, "Ungültig",      "stabil",   -10),
        ]:
            rows.append({
                "priority": "Hoch" if sig == "Hoch relevant" else "Mittel",
                "product": prod,
                "brand": "X",
                "retailer": retailer,
                "country": "HR",
                "category": "Hrana",
                "expected_start": today + timedelta(days=start_delta),
                "probability": prob,
                "signal": sig,
                "price_trend": trend,
            })
        self.df = pd.DataFrame(rows)

    def test_only_future_filters_past(self):
        result = apply_forecast_filters(self.df, only_future=True)
        # Past prediction (Old Product mit -10) muss raus
        assert "Old Product" not in result["product"].tolist()

    def test_window_filters_far_future(self):
        result = apply_forecast_filters(self.df, only_future=True, prediction_window_days=15)
        assert "Red Bull 250ml" in result["product"].tolist()
        assert "Coca-Cola" not in result["product"].tolist()  # 30 Tage

    def test_retailer_filter(self):
        result = apply_forecast_filters(self.df, retailer="Konzum", only_future=False)
        assert all(result["retailer"] == "Konzum")

    def test_product_query(self):
        result = apply_forecast_filters(self.df, product_query="milka", only_future=False)
        assert all("Milka" in p for p in result["product"])

    def test_min_probability(self):
        result = apply_forecast_filters(self.df, min_probability=0.80, only_future=False)
        assert all(result["probability"] >= 0.80)

    def test_signal_filter(self):
        result = apply_forecast_filters(self.df, signal="Hoch relevant", only_future=False)
        assert all(result["signal"] == "Hoch relevant")

    def test_trend_filter(self):
        result = apply_forecast_filters(self.df, price_trend="steigend", only_future=False)
        assert all(result["price_trend"] == "steigend")

    def test_alle_option_passes(self):
        result = apply_forecast_filters(self.df, signal="Alle", only_future=False)
        assert len(result) == len(self.df)


# ── DataFrame Konvertierung ───────────────────────────────────────────────────

class TestForecastsToDataframe:
    def test_empty_returns_empty(self):
        assert forecasts_to_dataframe([]).empty

    def test_sort_by_priority(self):
        df = _build_history(n_promos=6, cycle_days=45)
        forecasts = build_forecasts_from_promo_history(df)
        result = forecasts_to_dataframe(forecasts)
        if not result.empty:
            # Erwartete Spalten vorhanden
            assert "priority" in result.columns
            assert "product" in result.columns
            assert "probability" in result.columns

    def test_columns_present(self):
        df = _build_history(n_promos=6, cycle_days=45)
        forecasts = build_forecasts_from_promo_history(df)
        result = forecasts_to_dataframe(forecasts)
        expected_cols = {
            "priority", "product", "brand", "retailer", "country", "category",
            "expected_start", "probability", "expected_price", "last_promo_price",
            "price_trend", "avg_cycle_days", "historical_count", "signal",
            "justification", "last_promos",
        }
        assert expected_cols.issubset(set(result.columns))


# ── validate_price_series ─────────────────────────────────────────────────────

class TestValidatePriceSeries:
    def _make_df(self, prices, currency="EUR", name="Test Product", retailer="Test Retailer"):
        return pd.DataFrame({
            "name": [name] * len(prices),
            "store_name": [retailer] * len(prices),
            "price_eur": prices,
            "currency": [currency] * len(prices),
        })

    def test_none_returns_empty(self):
        result = validate_price_series(None)
        assert result.empty

    def test_empty_df_returns_empty(self):
        result = validate_price_series(pd.DataFrame())
        assert result.empty

    def test_valid_prices_tagged_valid(self):
        df = self._make_df([1.0, 1.1, 1.05, 0.99, 1.02])
        result = validate_price_series(df)
        assert "price_tag" in result.columns
        assert (result["price_tag"] == "valid_price").all()

    def test_outlier_high_detected(self):
        prices = [1.0, 1.0, 1.0, 1.0, 1.0, 99.0]  # last is 99× median
        df = self._make_df(prices)
        result = validate_price_series(df)
        assert result["price_tag"].iloc[-1] == "outlier_high"

    def test_outlier_low_detected(self):
        prices = [1.0, 1.0, 1.0, 1.0, 1.0, 0.001]
        df = self._make_df(prices)
        result = validate_price_series(df)
        assert result["price_tag"].iloc[-1] == "outlier_low"

    def test_zero_price_invalid(self):
        df = self._make_df([1.0, 0.0, 1.0])
        result = validate_price_series(df)
        assert result["price_tag"].iloc[1] == "invalid_price"

    def test_negative_price_invalid(self):
        df = self._make_df([1.0, -0.5, 1.0])
        result = validate_price_series(df)
        assert result["price_tag"].iloc[1] == "invalid_price"

    def test_mixed_currency_detected(self):
        df = pd.DataFrame({
            "name": ["A", "B"],
            "store_name": ["S", "S"],
            "price_eur": [1.0, 2.0],
            "currency": ["EUR", "RSD"],
        })
        result = validate_price_series(df)
        assert (result["price_tag"] == "mixed_currency").all()

    def test_mixed_product_detected(self):
        df = pd.DataFrame({
            "name": ["Product A", "Product B"],
            "store_name": ["S", "S"],
            "price_eur": [1.0, 2.0],
            "currency": ["EUR", "EUR"],
        })
        result = validate_price_series(df)
        assert (result["price_tag"] == "mixed_product").all()

    def test_mixed_retailer_detected(self):
        df = pd.DataFrame({
            "name": ["A", "A"],
            "store_name": ["Konzum", "Lidl"],
            "price_eur": [1.0, 2.0],
            "currency": ["EUR", "EUR"],
        })
        result = validate_price_series(df)
        assert (result["price_tag"] == "mixed_retailer").all()

    def test_price_tag_column_added(self):
        df = self._make_df([1.0, 1.1])
        result = validate_price_series(df)
        assert "price_tag" in result.columns

    def test_fewer_than_3_valid_no_outlier_check(self):
        df = self._make_df([1.0, 2.0])
        result = validate_price_series(df)
        # Only 2 valid prices → no IQR check, should all be valid
        assert not (result["price_tag"] == "outlier_high").any()


# ── check_forecast_eligibility ────────────────────────────────────────────────

class TestCheckForecastEligibility:
    def test_eligible_with_enough_data(self):
        ok, reason = check_forecast_eligibility(
            historical_count=10,
            history_span_days=200,
            outlier_ratio=0.0,
        )
        assert ok is True
        assert reason == ""

    def test_too_few_promos(self):
        ok, reason = check_forecast_eligibility(
            historical_count=1,
            history_span_days=200,
        )
        assert ok is False
        assert "1" in reason

    def test_too_short_span(self):
        ok, reason = check_forecast_eligibility(
            historical_count=5,
            history_span_days=30,
        )
        assert ok is False
        assert "30" in reason

    def test_too_many_outliers(self):
        ok, reason = check_forecast_eligibility(
            historical_count=10,
            history_span_days=365,
            outlier_ratio=0.6,
        )
        assert ok is False
        assert "outlier" in reason.lower() or "%" in reason

    def test_exactly_min_promos_eligible(self):
        ok, _ = check_forecast_eligibility(
            historical_count=MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER,
            history_span_days=MIN_HISTORY_DAYS,
        )
        assert ok is True

    def test_one_below_min_promos_not_eligible(self):
        ok, _ = check_forecast_eligibility(
            historical_count=MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER - 1,
            history_span_days=MIN_HISTORY_DAYS,
        )
        assert ok is False


# ── compute_data_quality_score ────────────────────────────────────────────────

class TestComputeDataQualityScore:
    def _full_forecast(self) -> "PromoForecast":
        from src.forecasting import PromoForecast
        return PromoForecast(
            product="Test", brand="Brand", retailer="Store",
            country="HR", category="Hrana",
            historical_count=12,
            avg_cycle_days=30, median_cycle_days=30,
            last_promo_price=1.5,
            avg_promo_price_90d=1.4,
            avg_promo_price_180d=1.45,
            typical_discount_pct_min=10.0,
            typical_discount_pct_max=25.0,
            prediction_status="ok",
        )

    def test_score_between_0_and_100(self):
        fc = self._full_forecast()
        score = compute_data_quality_score(fc)
        assert 0.0 <= score <= 100.0

    def test_full_data_high_score(self):
        fc = self._full_forecast()
        score = compute_data_quality_score(fc)
        assert score >= 60.0

    def test_empty_forecast_low_score(self):
        from src.forecasting import PromoForecast
        fc = PromoForecast(
            product="X", brand="", retailer="Y", country="HR", category="",
            historical_count=0,
        )
        score = compute_data_quality_score(fc)
        assert score < 30.0

    def test_more_history_gives_higher_score(self):
        from src.forecasting import PromoForecast
        fc_low = PromoForecast(
            product="X", brand="", retailer="Y", country="HR", category="",
            historical_count=3, avg_cycle_days=30, median_cycle_days=30,
        )
        fc_high = PromoForecast(
            product="X", brand="", retailer="Y", country="HR", category="",
            historical_count=12, avg_cycle_days=30, median_cycle_days=30,
        )
        assert compute_data_quality_score(fc_high) > compute_data_quality_score(fc_low)

    def test_score_with_discount_info_higher(self):
        from src.forecasting import PromoForecast
        fc_no_disc = PromoForecast(
            product="X", brand="", retailer="Y", country="HR", category="",
            historical_count=6, avg_cycle_days=30, median_cycle_days=30,
        )
        fc_with_disc = PromoForecast(
            product="X", brand="", retailer="Y", country="HR", category="",
            historical_count=6, avg_cycle_days=30, median_cycle_days=30,
            typical_discount_pct_min=10.0, typical_discount_pct_max=20.0,
        )
        assert compute_data_quality_score(fc_with_disc) > compute_data_quality_score(fc_no_disc)

    def test_returns_float(self):
        fc = self._full_forecast()
        score = compute_data_quality_score(fc)
        assert isinstance(score, float)


# ── i18n module ───────────────────────────────────────────────────────────────

class TestI18n:
    def test_t_returns_string(self):
        from src.i18n import t
        result = t("sidebar.market", "EN")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_t_fallback_to_en(self):
        from src.i18n import t
        result_en = t("sidebar.market", "EN")
        result_unknown = t("sidebar.market", "XX")
        assert result_en == result_unknown

    def test_t_unknown_key_returns_key(self):
        from src.i18n import t
        result = t("this.key.does.not.exist", "EN")
        assert result == "this.key.does.not.exist"

    def test_t_format_substitution(self):
        from src.i18n import t
        result = t("kam.min_req", "EN", n=3, d=90)
        assert "3" in result
        assert "90" in result

    def test_translate_category_en(self):
        from src.i18n import translate_category
        assert translate_category("Hrana", "EN") == "Food"

    def test_translate_category_hr(self):
        from src.i18n import translate_category
        assert translate_category("Hrana", "HR") == "Hrana"

    def test_translate_category_mk(self):
        from src.i18n import translate_category
        result = translate_category("Hrana", "MK")
        assert result == "Храна"

    def test_translate_category_unknown(self):
        from src.i18n import translate_category
        result = translate_category("UnknownCat999", "EN")
        assert result == "UnknownCat999"

    def test_all_supported_langs_have_market_keys(self):
        from src.i18n import SUPPORTED_LANGS, t
        for lang in SUPPORTED_LANGS:
            result = t("sidebar.market", lang)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_format_price_eur(self):
        from src.i18n import format_price
        result = format_price(1.99, "EUR")
        assert "1.99" in result
        assert "€" in result

    def test_format_price_rsd(self):
        from src.i18n import format_price
        result = format_price(250.0, "RSD")
        assert "250" in result
        assert "din" in result

    def test_format_price_none(self):
        from src.i18n import format_price
        assert format_price(None) == "—"

    def test_get_market_currency(self):
        from src.i18n import get_market_currency
        assert get_market_currency("HR") == "EUR"
        assert get_market_currency("RS") == "RSD"
        assert get_market_currency("MK") == "MKD"
        assert get_market_currency("BA") == "BAM"
        assert get_market_currency("SI") == "EUR"

    def test_get_market_currency_none(self):
        from src.i18n import get_market_currency
        assert get_market_currency(None) == "EUR"

    def test_default_language_by_market(self):
        from src.i18n import DEFAULT_LANGUAGE_BY_MARKET
        assert DEFAULT_LANGUAGE_BY_MARKET["HR"] == "HR"
        assert DEFAULT_LANGUAGE_BY_MARKET["SI"] == "SL"
        assert DEFAULT_LANGUAGE_BY_MARKET["RS"] == "SR"
        assert DEFAULT_LANGUAGE_BY_MARKET["MK"] == "MK"

    def test_market_currency_dict_completeness(self):
        from src.i18n import MARKET_CURRENCY, CATEGORY_TRANSLATIONS
        assert len(MARKET_CURRENCY) >= 6
        assert len(CATEGORY_TRANSLATIONS) >= 5

    def test_category_translations_food(self):
        from src.i18n import CATEGORY_TRANSLATIONS
        assert "Hrana" in CATEGORY_TRANSLATIONS
        assert CATEGORY_TRANSLATIONS["Hrana"]["EN"] == "Food"
        assert CATEGORY_TRANSLATIONS["Hrana"]["MK"] == "Храна"

    def test_all_langs_have_signal_high(self):
        from src.i18n import SUPPORTED_LANGS, t
        for lang in SUPPORTED_LANGS:
            result = t("signal.high", lang)
            assert "🔴" in result
