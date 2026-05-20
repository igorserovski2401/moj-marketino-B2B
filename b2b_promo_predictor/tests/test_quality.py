"""Tests für die Data Quality Pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.quality import (
    QualityReport,
    _fix_brands,
    _fix_categories,
    _validate_prices,
    run_quality_pipeline,
    validate_promo_price,
)


def _make_df(**overrides) -> pd.DataFrame:
    """Minimaler Testdatensatz mit sinnvollen Defaults."""
    data = {
        "name": ["Test Produkt"],
        "brand": [""],
        "category_l1": ["Other"],
        "price": [1.5],
        "original_price": [2.0],
        "price_eur": [1.5],
        "original_price_eur": [2.0],
        "currency": ["EUR"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


# ── Brand-Fixing ──────────────────────────────────────────────────────────────

class TestFixBrands:
    def test_red_bull_brand_assigned(self):
        df = _make_df(name=["Red Bull 250ml"], brand=[""])
        result, n = _fix_brands(df)
        assert "Red Bull" in result["brand"].iloc[0]
        assert n >= 1

    def test_milka_brand_assigned(self):
        df = _make_df(name=["Milka Schokolade 300g"], brand=[""])
        result, n = _fix_brands(df)
        assert "Milka" in result["brand"].iloc[0]

    def test_haribo_brand_assigned(self):
        df = _make_df(name=["Haribo Goldbären 200g"], brand=[""])
        result, n = _fix_brands(df)
        assert "Haribo" in result["brand"].iloc[0]

    def test_ariel_brand_pg(self):
        df = _make_df(name=["Ariel Pods 35 Kom."], brand=[""])
        result, n = _fix_brands(df)
        assert "P&G" in result["brand"].iloc[0]

    def test_pampers_brand_pg(self):
        df = _make_df(name=["Pampers Baby-Dry Gr.3"], brand=[""])
        result, n = _fix_brands(df)
        assert "P&G" in result["brand"].iloc[0]

    def test_vegeta_brand_podravka(self):
        df = _make_df(name=["Vegeta Začin 500g"], brand=[""])
        result, n = _fix_brands(df)
        assert "Podravka" in result["brand"].iloc[0]

    def test_nutella_brand_ferrero(self):
        df = _make_df(name=["Nutella 450g"], brand=[""])
        result, n = _fix_brands(df)
        assert "Ferrero" in result["brand"].iloc[0]

    def test_already_correct_brand_not_counted(self):
        df = _make_df(name=["Red Bull 250ml"], brand=["Red Bull"])
        result, n = _fix_brands(df)
        assert n == 0

    def test_no_name_column_returns_unchanged(self):
        df = pd.DataFrame({"brand": ["TestBrand"]})
        result, n = _fix_brands(df)
        assert n == 0
        assert result["brand"].iloc[0] == "TestBrand"

    def test_empty_df_returns_empty(self):
        result, n = _fix_brands(pd.DataFrame())
        assert result.empty
        assert n == 0


# ── Kategorie-Fixing ──────────────────────────────────────────────────────────

class TestFixCategories:
    def test_energy_drink_category(self):
        df = _make_df(name=["Red Bull 250ml Energy"], category_l1=["Piće"])
        result, n = _fix_categories(df)
        assert "Energy" in result["category_de"].iloc[0] or "Getränke" in result["category_de"].iloc[0]

    def test_ariel_gets_waschmittel_category(self):
        df = _make_df(name=["Ariel Waschmittel 20 WL"], category_l1=["Other"])
        result, n = _fix_categories(df)
        assert result["category_de"].iloc[0] not in ("Other", "—", None)

    def test_haribo_not_drinks(self):
        df = _make_df(name=["Haribo Goldbären 200g"], category_l1=["Piće"])
        result, n = _fix_categories(df)
        cat = result["category_de"].iloc[0]
        assert "Getränke" not in cat or "Süßwaren" in cat

    def test_category_de_column_created(self):
        df = _make_df()
        result, _ = _fix_categories(df)
        assert "category_de" in result.columns

    def test_fallback_for_unknown_category(self):
        df = _make_df(name=["Unbekanntes Produkt XYZ"], category_l1=["unknown_cat"])
        result, _ = _fix_categories(df)
        assert "category_de" in result.columns
        assert result["category_de"].iloc[0] is not None

    def test_n_fixed_increments(self):
        df = pd.DataFrame({
            "name": ["Red Bull 250ml", "Ariel Pods"],
            "brand": ["", ""],
            "category_l1": ["Piće", "Other"],
            "price": [1.5, 1.0],
            "original_price": [2.0, 2.0],
            "price_eur": [1.5, 1.0],
            "original_price_eur": [2.0, 2.0],
            "currency": ["EUR", "EUR"],
        })
        result, n = _fix_categories(df)
        assert n >= 2


# ── Preis-Validierung ─────────────────────────────────────────────────────────

class TestValidatePrices:
    def test_inverted_prices_swapped(self):
        df = pd.DataFrame({
            "name": ["Test"],
            "brand": ["Test"],
            "category_l1": ["Food"],
            "price": [5.0],
            "original_price": [2.0],
            "price_eur": [5.0],
            "original_price_eur": [2.0],
            "currency": ["EUR"],
        })
        result, n_swap, n_excl = _validate_prices(df)
        assert n_swap >= 1
        assert result["price_eur"].iloc[0] <= result["original_price_eur"].iloc[0]

    def test_valid_prices_not_swapped(self):
        df = _make_df(price=[1.5], original_price=[2.0], price_eur=[1.5], original_price_eur=[2.0])
        result, n_swap, n_excl = _validate_prices(df)
        assert n_swap == 0
        assert result["price_eur"].iloc[0] == pytest.approx(1.5)

    def test_impossible_discount_excluded(self):
        df = pd.DataFrame({
            "name": ["Test"],
            "brand": ["Test"],
            "category_l1": ["Food"],
            "price": [0.01],
            "original_price": [10.0],
            "price_eur": [0.01],
            "original_price_eur": [10.0],
            "currency": ["EUR"],
        })
        result, n_swap, n_excl = _validate_prices(df)
        if n_excl > 0:
            assert len(result) < 1
        else:
            assert result["price_eur"].iloc[0] < result["original_price_eur"].iloc[0]

    def test_discount_pct_computed(self):
        df = _make_df(price=[1.0], original_price=[2.0], price_eur=[1.0], original_price_eur=[2.0])
        result, _, _ = _validate_prices(df)
        assert "discount_pct" in result.columns
        assert result["discount_pct"].iloc[0] == pytest.approx(50.0, abs=1.0)

    def test_no_price_column_returns_unchanged(self):
        df = pd.DataFrame({"name": ["Test"], "brand": ["Test"]})
        result, n_swap, n_excl = _validate_prices(df)
        assert n_swap == 0
        assert n_excl == 0


# ── validate_promo_price (Einzelzeile) ────────────────────────────────────────

class TestValidatePromoPrice:
    def test_valid_discount(self):
        row = pd.Series({"price_eur": 1.0, "original_price_eur": 2.0})
        assert validate_promo_price(row) == "valid_discount"

    def test_no_real_discount(self):
        row = pd.Series({"price_eur": 2.0, "original_price_eur": 2.0})
        assert validate_promo_price(row) == "no_real_discount"

    def test_missing_price(self):
        row = pd.Series({"price_eur": float("nan"), "original_price_eur": 2.0})
        assert validate_promo_price(row) == "missing_price"

    def test_missing_original_price(self):
        row = pd.Series({"price_eur": 1.5, "original_price_eur": float("nan")})
        assert validate_promo_price(row) == "missing_price"

    def test_invalid_zero_price(self):
        row = pd.Series({"price_eur": 0.0, "original_price_eur": 2.0})
        assert validate_promo_price(row) == "invalid_price"

    def test_falls_back_to_price_column(self):
        row = pd.Series({"price": 1.0, "original_price": 2.0})
        assert validate_promo_price(row) == "valid_discount"


# ── run_quality_pipeline ──────────────────────────────────────────────────────

class TestQualityPipeline:
    def test_returns_tuple(self):
        df = _make_df()
        result = run_quality_pipeline(df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_quality_report(self):
        df = _make_df()
        _, report = run_quality_pipeline(df)
        assert isinstance(report, QualityReport)

    def test_empty_df_returns_empty(self):
        df, report = run_quality_pipeline(pd.DataFrame())
        assert df.empty
        assert report.n_total == 0

    def test_report_fields_exist(self):
        _, report = run_quality_pipeline(_make_df())
        assert hasattr(report, "n_total")
        assert hasattr(report, "n_brand_fixed")
        assert hasattr(report, "n_cat_fixed")
        assert hasattr(report, "n_price_swapped")
        assert hasattr(report, "n_excluded")
        assert hasattr(report, "n_clean")

    def test_n_clean_consistency(self):
        df = _make_df()
        result, report = run_quality_pipeline(df)
        assert report.n_clean == report.n_total - report.n_excluded
        assert report.n_clean >= 0
        assert report.n_clean == len(result)

    def test_brands_fixed_in_pipeline(self):
        df = pd.DataFrame({
            "name": ["Red Bull 250ml", "Milka Schokolade", "Haribo Goldbären"],
            "brand": ["", "", ""],
            "category_l1": ["Piće", "Other", "Piće"],
            "price": [1.5, 2.0, 1.0],
            "original_price": [2.0, 3.0, 1.5],
            "price_eur": [1.5, 2.0, 1.0],
            "original_price_eur": [2.0, 3.0, 1.5],
            "currency": ["EUR", "EUR", "EUR"],
        })
        result, report = run_quality_pipeline(df)
        assert report.n_brand_fixed >= 3
        assert not result.empty

    def test_category_de_added(self):
        df = _make_df()
        result, _ = run_quality_pipeline(df)
        assert "category_de" in result.columns

    def test_asdict_works(self):
        _, report = run_quality_pipeline(_make_df())
        d = report._asdict()
        assert isinstance(d, dict)
        assert "n_total" in d
