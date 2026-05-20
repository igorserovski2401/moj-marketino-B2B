"""Tests für ProductMatcher – Entity Resolution Pipeline."""

import pytest

from src.matching import (
    BLOCKED_CATEGORY_PAIRS,
    KEYWORD_MASTER_RULES,
    MIN_MATCH_SCORE,
    UNKNOWN_MASTER_PRODUCT,
    MatchResult,
    ProductMatcher,
    _check_category_conflict,
    _keyword_match,
    _normalize_name,
)

MASTER_LIST = [
    "Red Bull Energy Drink 250ml",
    "Milka Schokolade 300g",
    "Ariel Waschmittel 20 WL",
    "Pampers Baby-Dry Gr.3 44 Stk",
    "Haribo Goldbären 200g",
    "Coca-Cola 1,5L PET",
    "Nutella Nuss-Nougat-Creme 450g",
]


# ── Konstanten ────────────────────────────────────────────────────────────────

class TestConstants:
    def test_min_score_is_085(self):
        assert MIN_MATCH_SCORE == 0.85

    def test_unknown_product_not_empty(self):
        assert len(UNKNOWN_MASTER_PRODUCT) > 0

    def test_keyword_rules_not_empty(self):
        assert len(KEYWORD_MASTER_RULES) >= 10

    def test_blocked_pairs_not_empty(self):
        assert len(BLOCKED_CATEGORY_PAIRS) >= 4


# ── Keyword-Matching ──────────────────────────────────────────────────────────

class TestKeywordMatch:
    def test_red_bull_matched(self):
        result = _keyword_match("Red Bull 250ml Dosen")
        assert result is not None
        assert result.master_product == "Red Bull Energy Drink 250ml"
        assert result.match_score == 1.0
        assert result.match_status == "keyword_exact"
        assert result.match_method == "keyword"

    def test_red_bull_case_insensitive(self):
        assert _keyword_match("RED BULL 355ML") is not None
        assert _keyword_match("red bull energy") is not None

    def test_milka_matched(self):
        result = _keyword_match("MILKA SCHOKOLADE 300g")
        assert result is not None
        assert "Milka" in result.master_product

    def test_ariel_matched(self):
        result = _keyword_match("Ariel Pods 35 Kom.")
        assert result is not None
        assert "Ariel" in result.master_product

    def test_pampers_matched(self):
        result = _keyword_match("Pampers Windeln Größe 4")
        assert result is not None

    def test_haribo_matched(self):
        result = _keyword_match("Haribo Goldbären 200g")
        assert result is not None

    def test_kokosja_juha_matched(self):
        result = _keyword_match("Kokošja juha 500ml")
        assert result is not None
        assert result.match_status == "keyword_exact"

    def test_majoneza_matched(self):
        result = _keyword_match("Majoneza 250g")
        assert result is not None

    def test_caj_matched(self):
        result = _keyword_match("Čaj od kamilice 20 vrećica")
        assert result is not None

    def test_no_keyword_match_returns_none(self):
        assert _keyword_match("Unbekanntes Produkt XYZ 999") is None

    def test_result_has_all_fields(self):
        result = _keyword_match("Red Bull 250ml")
        assert result is not None
        assert hasattr(result, "original_product_name")
        assert hasattr(result, "normalized_product_name")
        assert hasattr(result, "match_status")
        assert hasattr(result, "match_method")
        assert hasattr(result, "raw_candidate")
        assert result.original_product_name == "Red Bull 250ml"


# ── Kategorie-Konflikt ────────────────────────────────────────────────────────

class TestCategoryConflict:
    def test_food_household_blocked(self):
        assert _check_category_conflict("food", "household") is True

    def test_household_food_blocked(self):
        assert _check_category_conflict("household", "food") is True

    def test_drinks_cosmetics_blocked(self):
        assert _check_category_conflict("drinks", "cosmetics") is True

    def test_cosmetics_drinks_blocked(self):
        assert _check_category_conflict("cosmetics", "drinks") is True

    def test_hrana_kozmetika_blocked(self):
        assert _check_category_conflict("hrana", "kozmetika") is True

    def test_kozmetika_hrana_blocked(self):
        assert _check_category_conflict("kozmetika", "hrana") is True

    def test_food_food_not_blocked(self):
        assert _check_category_conflict("food", "food") is False

    def test_empty_cats_not_blocked(self):
        assert _check_category_conflict("", "") is False

    def test_unknown_pair_not_blocked(self):
        assert _check_category_conflict("electronics", "household") is False


# ── ProductMatcher ────────────────────────────────────────────────────────────

class TestProductMatcherInit:
    def test_threshold_enforced_to_min(self):
        m = ProductMatcher(threshold=0.50)
        assert m.threshold == MIN_MATCH_SCORE

    def test_threshold_set_above_min(self):
        m = ProductMatcher(threshold=0.90)
        assert m.threshold == 0.90

    def test_threshold_at_min(self):
        m = ProductMatcher(threshold=MIN_MATCH_SCORE)
        assert m.threshold == MIN_MATCH_SCORE


class TestMatchProduct:
    def setup_method(self):
        self.matcher = ProductMatcher(threshold=MIN_MATCH_SCORE)

    def test_keyword_match_wins(self):
        result = self.matcher.match_product("Red Bull 250ml", MASTER_LIST)
        assert result.master_product == "Red Bull Energy Drink 250ml"
        assert result.match_status == "keyword_exact"
        assert result.match_method == "keyword"
        assert result.match_score == 1.0

    def test_keyword_is_confident(self):
        result = self.matcher.match_product("Milka Schokolade", MASTER_LIST)
        assert result.is_confident is True

    def test_empty_master_list_returns_unmatched_empty(self):
        result = self.matcher.match_product("Red Bull 250ml", [])
        assert result.match_status == "unmatched_empty"
        assert result.master_product == UNKNOWN_MASTER_PRODUCT

    def test_score_property_alias(self):
        result = self.matcher.match_product("Red Bull 250ml", MASTER_LIST)
        assert result.score == result.match_score

    def test_is_confident_true_for_keyword(self):
        result = self.matcher.match_product("Red Bull 250ml", MASTER_LIST)
        assert result.is_confident is True

    def test_raw_candidate_populated(self):
        result = self.matcher.match_product("Red Bull 250ml", MASTER_LIST)
        assert result.raw_candidate != ""


class TestBatchMatch:
    def setup_method(self):
        self.matcher = ProductMatcher(threshold=MIN_MATCH_SCORE)

    def test_empty_names_returns_empty(self):
        assert self.matcher.batch_match([], MASTER_LIST) == []

    def test_empty_master_returns_empty(self):
        assert self.matcher.batch_match(["Red Bull"], []) == []

    def test_returns_same_count(self):
        names = ["Red Bull 250ml", "Milka", "Haribo", "Ariel", "Pampers"]
        results = self.matcher.batch_match(names, MASTER_LIST)
        assert len(results) == len(names)

    def test_all_keyword_products_confident(self):
        names = ["Red Bull 250ml", "Milka Schokolade", "Ariel Pods"]
        results = self.matcher.batch_match(names, MASTER_LIST)
        for r in results:
            assert r.match_status == "keyword_exact"
            assert r.is_confident is True

    def test_results_have_all_fields(self):
        results = self.matcher.batch_match(["Red Bull 250ml"], MASTER_LIST)
        r = results[0]
        assert hasattr(r, "original_product_name")
        assert hasattr(r, "normalized_product_name")
        assert hasattr(r, "match_status")
        assert hasattr(r, "match_method")
        assert hasattr(r, "raw_candidate")
        assert r.original_product_name == "Red Bull 250ml"

    def test_unknown_product_is_unmatched(self):
        results = self.matcher.batch_match(["XYZXYZ_UNKNOWN_PRODUCT_12345"], MASTER_LIST)
        r = results[0]
        # Either low confidence or it happens to match something – check it's NOT keyword
        assert r.match_method != "keyword"
        if r.match_status == "unmatched_low_confidence":
            assert r.master_product == UNKNOWN_MASTER_PRODUCT

    def test_order_preserved(self):
        names = ["Red Bull 250ml", "Milka", "Haribo"]
        results = self.matcher.batch_match(names, MASTER_LIST)
        assert results[0].original_product_name == "Red Bull 250ml"
        assert results[1].original_product_name == "Milka"
        assert results[2].original_product_name == "Haribo"

    def test_raw_categories_passed(self):
        names = ["Red Bull 250ml"]
        cats = ["Piće"]
        results = self.matcher.batch_match(names, MASTER_LIST, raw_categories=cats)
        assert results[0].category == "Piće"
