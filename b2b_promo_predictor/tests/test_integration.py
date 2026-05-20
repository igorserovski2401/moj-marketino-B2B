"""Integration tests for central filter pipeline, production isolation, and forecast guards."""

from __future__ import annotations

import pandas as pd
import pytest


# ── Brand/product search normalization ────────────────────────────────────────

class TestNormalizeSearchText:
    def test_lowercase(self):
        from app import normalize_search_text
        assert normalize_search_text("WUDY") == "wudy"

    def test_strips_whitespace(self):
        from app import normalize_search_text
        assert normalize_search_text("  wudy  ") == "wudy"

    def test_collapses_whitespace(self):
        from app import normalize_search_text
        assert normalize_search_text("wudy\t\thot dog") == "wudy hot dog"

    def test_strips_diacritics(self):
        from app import normalize_search_text
        assert normalize_search_text("Čokolada") == "cokolada"
        assert normalize_search_text("Müsli") == "musli"

    def test_none_returns_empty(self):
        from app import normalize_search_text
        assert normalize_search_text(None) == ""

    def test_number_to_string(self):
        from app import normalize_search_text
        assert normalize_search_text(123) == "123"


class TestApplyBrandProductSearch:
    def _df(self):
        return pd.DataFrame({
            "name": ["Wudy hot dog 200g", "Milka Schokolade", "Coca-Cola 1.5L"],
            "brand": ["Wudy", "Milka", "Coca-Cola"],
            "store_name": ["Konzum", "Konzum", "Konzum"],
        })

    def test_finds_wudy(self):
        from app import apply_brand_product_search
        result = apply_brand_product_search(self._df(), "wudy")
        assert len(result) == 1
        assert "Wudy" in result["name"].iloc[0]

    def test_case_insensitive(self):
        from app import apply_brand_product_search
        result = apply_brand_product_search(self._df(), "WUDY")
        assert len(result) == 1

    def test_no_results_returns_empty(self):
        from app import apply_brand_product_search
        result = apply_brand_product_search(self._df(), "xyz_nope")
        assert result.empty

    def test_empty_query_passes_through(self):
        from app import apply_brand_product_search
        result = apply_brand_product_search(self._df(), "")
        assert len(result) == 3

    def test_searches_across_brand_and_name(self):
        from app import apply_brand_product_search
        df = pd.DataFrame({
            "name": ["Some random product"],
            "brand": ["Wudy"],
        })
        result = apply_brand_product_search(df, "wudy")
        assert len(result) == 1


class TestBuildFilteredView:
    def _raw(self):
        return pd.DataFrame({
            "name": ["Wudy 200g", "Milka 300g", "Coca-Cola"] * 2,
            "brand": ["Wudy", "Milka", "Coca-Cola"] * 2,
            "store_name": ["Konzum"] * 3 + ["Lidl"] * 3,
            "country_code": ["HR"] * 3 + ["SI"] * 3,
            "category_l1": ["Hrana", "Slatkiši", "Piće"] * 2,
            "price": [1.0, 2.0, 1.5] * 2,
            "original_price": [1.5, 3.0, 2.0] * 2,
            "price_eur": [1.0, 2.0, 1.5] * 2,
            "original_price_eur": [1.5, 3.0, 2.0] * 2,
            "currency": ["EUR"] * 6,
        })

    def test_audit_contains_row_counts(self):
        from app import build_filtered_view
        _, audit = build_filtered_view(self._raw())
        assert audit["raw_rows"] == 6
        assert audit["final"] >= 0

    def test_country_filter_applies(self):
        from app import build_filtered_view
        df, audit = build_filtered_view(self._raw(), country="HR")
        assert (df["country_code"] == "HR").all()
        assert audit["after_market"] <= audit["after_quality"]

    def test_brand_filter_applies(self):
        from app import build_filtered_view
        df, audit = build_filtered_view(self._raw(), brand_query="wudy")
        assert all("Wudy" in n for n in df["name"])
        assert audit["after_brand"] <= audit["after_category"]

    def test_no_match_returns_empty(self):
        from app import build_filtered_view
        df, audit = build_filtered_view(self._raw(), brand_query="xyz_nope_nothing")
        assert df.empty
        assert audit["final"] == 0
        assert audit["after_brand"] == 0

    def test_empty_input_returns_empty(self):
        from app import build_filtered_view
        df, audit = build_filtered_view(pd.DataFrame())
        assert df.empty
        assert audit["raw_rows"] == 0

    def test_compound_filter(self):
        from app import build_filtered_view
        df, _ = build_filtered_view(self._raw(), country="HR", brand_query="wudy")
        assert len(df) >= 0
        for _, row in df.iterrows():
            assert "Wudy" in row["name"]
            assert row["country_code"] == "HR"


# ── Forecast selection validation ─────────────────────────────────────────────

class TestValidateForecastSelection:
    def test_ok_when_all_provided(self):
        from app import validate_forecast_selection
        assert validate_forecast_selection("Coca-Cola", "Konzum", "HR", "EUR") == "ok"

    def test_missing_product(self):
        from app import validate_forecast_selection
        assert validate_forecast_selection("", "Konzum", "HR", "EUR") == "missing_product"
        assert validate_forecast_selection(None, "Konzum", "HR", "EUR") == "missing_product"

    def test_missing_retailer(self):
        from app import validate_forecast_selection
        assert validate_forecast_selection("X", None, "HR", "EUR") == "missing_retailer"
        assert validate_forecast_selection("X", "__all__", "HR", "EUR") == "missing_retailer"

    def test_missing_market(self):
        from app import validate_forecast_selection
        assert validate_forecast_selection("X", "Konzum", None, "EUR") == "missing_market"

    def test_missing_currency(self):
        from app import validate_forecast_selection
        assert validate_forecast_selection("X", "Konzum", "HR", None) == "missing_currency"
        assert validate_forecast_selection("X", "Konzum", "HR", "") == "missing_currency"


# ── Trust level math ──────────────────────────────────────────────────────────

class TestComputeTrustLevel:
    def test_belastbar_when_low_mape_and_enough_data(self):
        from app import compute_trust_level
        assert compute_trust_level(mape=0.15, observations=20, history_days=180) == "belastbar"

    def test_eingeschr_when_medium_mape(self):
        from app import compute_trust_level
        assert compute_trust_level(mape=0.30, observations=20, history_days=180) == "eingeschr"

    def test_nicht_belastbar_when_too_few_observations(self):
        from app import compute_trust_level
        assert compute_trust_level(mape=0.10, observations=5, history_days=180) == "nicht_belastbar"

    def test_nicht_belastbar_when_short_history(self):
        from app import compute_trust_level
        assert compute_trust_level(mape=0.10, observations=20, history_days=30) == "nicht_belastbar"

    def test_eingeschr_when_mape_is_nan_but_data_ok(self):
        from app import compute_trust_level
        nan = float("nan")
        assert compute_trust_level(mape=nan, observations=20, history_days=180) == "eingeschr"

    def test_nicht_belastbar_when_high_mape(self):
        from app import compute_trust_level
        assert compute_trust_level(mape=0.55, observations=20, history_days=180) == "nicht_belastbar"


# ── Production isolation: predict() never returns demo retailers ──────────────

class TestPredictProductionIsolation:
    def test_predict_returns_empty_without_model_in_production(self):
        from src.model import predict
        result = predict(allow_mock=False)
        assert result.empty
        # No Aldi/Rewe/Penny/Netto leaking in
        for row in result.itertuples():
            assert "Aldi" not in str(row)
            assert "Rewe" not in str(row)
            assert "Penny" not in str(row)
            assert "Netto" not in str(row)

    def test_predict_allows_mock_when_enabled(self):
        from src.model import predict
        result = predict(allow_mock=True)
        # When mock is explicit, mock data is returned — but only Balkan retailers
        if not result.empty:
            for retailer in result["retailer"].unique():
                assert retailer not in ("Aldi Süd", "Rewe", "Penny", "Netto")


class TestForecastPriceTrendProductionIsolation:
    def test_returns_empty_in_production_with_no_data(self):
        from src.model import forecast_price_trend
        fig, df = forecast_price_trend(
            df=pd.DataFrame(),
            product_id=None,
            periods=30,
            allow_mock=False,
        )
        assert df.empty

    def test_no_demo_title_in_production(self):
        from src.model import forecast_price_trend
        fig, _ = forecast_price_trend(
            df=pd.DataFrame(),
            product_id="Test Product",
            periods=30,
            allow_mock=False,
        )
        title_text = ""
        if hasattr(fig, "layout") and fig.layout.title and fig.layout.title.text:
            title_text = str(fig.layout.title.text)
        for ann in (fig.layout.annotations or []):
            title_text += str(ann.text or "")
        assert "(Demo)" not in title_text
        assert "[DEMO]" not in title_text


# ── Mock data: no German retailers anywhere ───────────────────────────────────

class TestNoGermanRetailers:
    GERMAN_RETAILERS = {"Aldi", "Aldi Süd", "Aldi Nord", "Rewe", "Penny", "Netto", "Edeka", "Real"}

    def test_features_mock_uses_balkan_retailers(self):
        from src.features import _make_mock_history
        df = _make_mock_history(n_rows=50)
        retailers = set(df["retailer"].unique())
        for r in retailers:
            assert r not in self.GERMAN_RETAILERS, f"German retailer leaked: {r}"

    def test_extraction_mock_uses_balkan_retailers(self):
        from src.extraction import _generate_mock_data
        promos = _generate_mock_data(n=5)
        for p in promos:
            assert p.retailer not in self.GERMAN_RETAILERS
            assert p.country in ("HR", "SI", "BA", "RS", "MK", "ME", "DE"), p.country

    def test_model_mock_predictions_uses_balkan(self):
        from src.model import _mock_predictions
        df = _mock_predictions(n=20)
        retailers = set(df["retailer"].unique())
        for r in retailers:
            assert r not in self.GERMAN_RETAILERS


# ── Treemap guard ─────────────────────────────────────────────────────────────

class TestTreemapGuard:
    def test_min_treemap_rows_constant_defined(self):
        from app import MIN_TREEMAP_ROWS
        assert MIN_TREEMAP_ROWS >= 5


# ── App-version exists and is visible ─────────────────────────────────────────

class TestAppVersion:
    def test_app_version_is_string(self):
        from app import APP_VERSION
        assert isinstance(APP_VERSION, str)
        assert len(APP_VERSION) > 0

    def test_cache_version_set(self):
        from app import CACHE_VERSION
        assert isinstance(CACHE_VERSION, str)
        assert len(CACHE_VERSION) > 0


# ── i18n integrity: forbidden German strings in non-DE languages ──────────────

class TestNoGermanInBalkanLangs:
    FORBIDDEN_GERMAN = [
        "Bevorstehende Aktionen",
        "Marke / Hersteller",
        "Alle Kategorien",
        "Erweiterte Einstellungen",
        "Beteiligte Händler",
        "Aktionsverteilung deaktiviert",
        "KI-Aktionsvorhersage",
        "Modell neu trainieren",
        "Wahrscheinlichkeit:",
        "Datenqualität & Pipeline",
        "Preisanalyse & Wettbewerb",
        "Aktionen gefunden",
    ]

    def test_no_german_in_hr_translations(self):
        from src.i18n import TRANSLATIONS
        for key, entry in TRANSLATIONS.items():
            for lang in ("HR", "SL", "BS", "SR", "MK", "ME", "EN"):
                val = entry.get(lang, "")
                for ger in self.FORBIDDEN_GERMAN:
                    assert ger not in val, f"German leaked into {key}[{lang}]: {val!r}"

    def test_market_currency_complete(self):
        from src.i18n import MARKET_CURRENCY
        for m in ("HR", "SI", "BA", "RS", "MK", "ME"):
            assert m in MARKET_CURRENCY

    def test_mk_uses_mkd(self):
        from src.i18n import get_market_currency
        assert get_market_currency("MK") == "MKD"

    def test_rs_uses_rsd(self):
        from src.i18n import get_market_currency
        assert get_market_currency("RS") == "RSD"

    def test_ba_uses_bam(self):
        from src.i18n import get_market_currency
        assert get_market_currency("BA") == "BAM"
