"""Supabase Data Access Layer für MarketinoDATABASE.

Alle Funktionen geben pandas DataFrames zurück und fallen auf
Mock-Daten zurück wenn keine Supabase-Verbindung konfiguriert ist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

from .config import settings
from .quality import run_quality_pipeline

# Spalten-Select für die products-Tabelle
_PRODUCT_COLS = (
    "id,name,brand,price,original_price,store_name,country_code,"
    "valid_from,valid_until,category_l1,category_l2,unit,amount,"
    "discount_label,promotion_type,approval_status,image_url,created_at,currency"
)

# Wechselkurse → EUR (approximative Festwerte)
_FX_TO_EUR: dict[str, float] = {
    "EUR": 1.0,
    "MKD": 1 / 61.5,   # 1 EUR ≈ 61.5 MKD
    "RSD": 1 / 117.0,  # 1 EUR ≈ 117 RSD
    "BAM": 1 / 1.956,  # 1 EUR ≈ 1.956 BAM (fester Kurs)
    "HRK": 1 / 7.534,  # historisch, Kroatien ist seit 2023 EUR
}

# ── Retailer-Normalisierung ───────────────────────────────────────────────────

_RETAILER_ALIASES: dict[str, str] = {
    "BIPA": "BIPA",
    "DM": "dm",
    "DM DROGERIE": "dm",
    "KONZUM": "Konzum",
    "KONZUM SUPER": "Konzum",
    "KAUFLAND": "Kaufland",
    "LIDL": "Lidl",
    "LIDL HRVATSKA": "Lidl",
    "SPAR": "Spar",
    "SPAR SUPERMARKET": "Spar",
    "INTERSPAR": "Interspar",
    "EUROSPAR": "Eurospar",
    "MAXI": "Maxi",
    "RODA": "Roda",
    "IDEA": "Idea",
    "SUPERKITGO": "Superkitgo",
    "TINEX": "Tinex",
    "VERO": "Vero",
    "STOKOMAK": "Stokomak",
    "BINGO": "Bingo",
    "VOLI": "Voli",
    "PLODINE": "Plodine",
    "MERCATOR": "Mercator",
    "TUŠ": "Tus",
    "TUS": "Tus",
    "JAGER": "Jager",
    "E.LECLERC": "E.Leclerc",
    "E. LECLERC": "E.Leclerc",
    "DIS": "Dis",
}


def normalize_retailer_name(name: str) -> str:
    """Normiert Händlernamen: strip + Alias-Auflösung für Duplikat-Dedup.

    Beispiel: 'BIPA' und 'bipa' → 'BIPA'; 'Lidl Hrvatska' → 'Lidl'.
    """
    if not name:
        return ""
    key = name.strip().upper()
    return _RETAILER_ALIASES.get(key, name.strip())


# Länder-Namen-Map
COUNTRY_NAMES: dict[str, str] = {
    "HR": "Kroatien 🇭🇷",
    "MK": "Nordmazedonien 🇲🇰",
    "SI": "Slowenien 🇸🇮",
    "RS": "Serbien 🇷🇸",
    "BA": "Bosnien 🇧🇦",
    "ME": "Montenegro 🇲🇪",
}


@lru_cache(maxsize=1)
def get_client() -> Any | None:
    """Erstellt und cached den Supabase-Client."""
    if not settings.has_supabase:
        return None
    try:
        from supabase import create_client
        return create_client(settings.supabase_url, settings.supabase_anon_key)
    except Exception:
        return None


def _safe_query(client: Any, table: str, select: str) -> Any:
    """Wrapper um client.table(...).select(...)."""
    return client.table(table).select(select)


# ── Produkte ──────────────────────────────────────────────────────────────────

def load_products(
    country_code: str | None = None,
    store_name: str | None = None,
    category_l1: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 500,
    approved_only: bool = False,
    allow_mock: bool = True,
) -> pd.DataFrame:
    """Load products from Supabase with optional filters.

    When allow_mock=False (production mode), returns empty DataFrame instead
    of synthetic fallback data on empty or error results.
    """
    client = get_client()
    if client is None:
        return _mock_products(limit) if allow_mock else pd.DataFrame()

    try:
        q = _safe_query(client, "products", _PRODUCT_COLS)
        if country_code:
            q = q.eq("country_code", country_code)
        if store_name:
            q = q.eq("store_name", store_name)
        if category_l1:
            q = q.eq("category_l1", category_l1)
        if date_from:
            q = q.gte("valid_from", date_from)
        if date_to:
            q = q.lte("valid_until", date_to)
        if approved_only:
            q = q.eq("approval_status", "approved")

        resp = q.order("created_at", desc=True).limit(limit).execute()
        if not resp.data:
            return _mock_products(limit) if allow_mock else pd.DataFrame()
        df = _normalize(pd.DataFrame(resp.data))
        df, _ = run_quality_pipeline(df)
        return df
    except Exception:
        return _mock_products(limit) if allow_mock else pd.DataFrame()


def load_upcoming_promos(
    country_code: str | None = None,
    days_ahead: int = 14,
    limit: int = 200,
    allow_mock: bool = True,
) -> pd.DataFrame:
    """Produkte deren Aktion in den nächsten `days_ahead` Tagen beginnt."""
    today = date.today().isoformat()
    future = (date.today() + timedelta(days=days_ahead)).isoformat()
    return load_products(
        country_code=country_code,
        date_from=today,
        date_to=future,
        limit=limit,
        allow_mock=allow_mock,
    )


def load_promo_history_for_forecast(
    country_code: str | None = None,
    category_l1: str | None = None,
    days_back: int = 365,
    limit: int = 5000,
) -> pd.DataFrame:
    """Lädt historische + aktuelle Aktionen für die Forecast-Berechnung.

    Anders als `load_products` wird kein Sortier-/Anzeige-Limit auf wenige
    aktuelle Datensätze gelegt – die ganze 12-Monats-Historie ist nötig
    damit Aktionszyklen und Preistrends berechnet werden können.
    """
    client = get_client()
    if client is None:
        return _mock_promo_history()

    since = (date.today() - timedelta(days=days_back)).isoformat()
    try:
        q = _safe_query(client, "products", _PRODUCT_COLS)
        if country_code:
            q = q.eq("country_code", country_code)
        if category_l1:
            q = q.eq("category_l1", category_l1)
        q = q.gte("valid_from", since).not_.is_("valid_from", "null")
        resp = q.order("valid_from", desc=False).limit(limit).execute()
        if not resp.data:
            return pd.DataFrame()
        df = _normalize(pd.DataFrame(resp.data))
        df, _ = run_quality_pipeline(df)
        return df
    except Exception:
        return pd.DataFrame()


def load_active_promos(
    country_code: str | None = None,
    limit: int = 300,
    allow_mock: bool = True,
) -> pd.DataFrame:
    """Products active today (valid_from <= today <= valid_until)."""
    client = get_client()
    if client is None:
        return _mock_products(50) if allow_mock else pd.DataFrame()

    today = date.today().isoformat()
    try:
        q = _safe_query(client, "products", _PRODUCT_COLS)
        q = q.lte("valid_from", today).gte("valid_until", today)
        if country_code:
            q = q.eq("country_code", country_code)
        resp = q.order("price", desc=False).limit(limit).execute()
        if not resp.data:
            return _mock_products(50) if allow_mock else pd.DataFrame()
        df = _normalize(pd.DataFrame(resp.data))
        df, _ = run_quality_pipeline(df)
        return df
    except Exception:
        return _mock_products(50) if allow_mock else pd.DataFrame()


def load_price_history(
    product_name: str | None = None,
    retailer: str | None = None,
    country_code: str | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Lädt Preishistorie aus product_price_history."""
    client = get_client()
    if client is None:
        return _mock_price_history()

    try:
        q = _safe_query(
            client, "product_price_history",
            "product_name,retailer,country_code,price,recorded_at"
        )
        if product_name:
            q = q.ilike("product_name", f"%{product_name}%")
        if retailer:
            q = q.eq("retailer", retailer)
        if country_code:
            q = q.eq("country_code", country_code)

        resp = q.order("recorded_at", desc=False).limit(limit).execute()
        if not resp.data:
            return _mock_price_history(product_name or "Milka")

        df = pd.DataFrame(resp.data)
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        return df.dropna(subset=["price", "recorded_at"])
    except Exception:
        return _mock_price_history()


# ── Aggregationen ─────────────────────────────────────────────────────────────

def get_retailer_distribution(country_code: str | None = None) -> pd.DataFrame:
    """Produkt-Anzahl pro Händler."""
    client = get_client()
    if client is None:
        return pd.DataFrame({
            "store_name": ["Konzum", "Kaufland", "Lidl", "Spar", "Bingo", "Tinex", "Maxi", "Voli", "Plodine", "Mercator"],
            "cnt": [182430, 154210, 112800, 98340, 76120, 61450, 54380, 43200, 38760, 31200],
        })

    try:
        q = _safe_query(client, "products", "store_name,country_code")
        if country_code:
            q = q.eq("country_code", country_code)
        resp = q.limit(10000).execute()
        if not resp.data:
            return pd.DataFrame()
        df = pd.DataFrame(resp.data)
        return (
            df.groupby("store_name")
            .size()
            .reset_index(name="cnt")
            .sort_values("cnt", ascending=False)
        )
    except Exception:
        return pd.DataFrame()


def get_category_distribution(country_code: str | None = None) -> pd.DataFrame:
    """Produkt-Anzahl pro Kategorie L1."""
    client = get_client()
    if client is None:
        return pd.DataFrame({
            "category_l1": ["Hrana", "Piće", "Kozmetika", "Other"],
            "cnt": [224106, 42587, 27437, 182752],
        })

    try:
        q = _safe_query(client, "products", "category_l1,country_code")
        if country_code:
            q = q.eq("country_code", country_code)
        resp = q.not_.is_("category_l1", "null").limit(20000).execute()
        if not resp.data:
            return pd.DataFrame()
        df = pd.DataFrame(resp.data)
        return (
            df.groupby("category_l1")
            .size()
            .reset_index(name="cnt")
            .sort_values("cnt", ascending=False)
            .head(15)
        )
    except Exception:
        return pd.DataFrame()


def get_overview_stats(country_code: str | None = None) -> dict[str, Any]:
    """Gibt KPI-Metriken für das Dashboard zurück."""
    client = get_client()

    # Feste Gesamtzahlen aus der DB-Exploration
    base = {
        "total_products": 997_487,
        "total_catalogs": 1_769,
        "total_retailers": 212,
        "total_countries": 6,
        "connected": client is not None,
    }

    if client is None:
        base["active_promos"] = 0
        return base

    try:
        today = date.today().isoformat()
        q = _safe_query(client, "products", "id,country_code")
        q = q.lte("valid_from", today).gte("valid_until", today)
        if country_code:
            q = q.eq("country_code", country_code)
        resp = q.limit(1000).execute()
        base["active_promos"] = len(resp.data) if resp.data else 0
    except Exception:
        base["active_promos"] = 0

    return base


def get_distinct_stores(country_code: str | None = None) -> list[str]:
    """Alle Händlernamen für den Dropdown-Filter."""
    client = get_client()
    if client is None:
        return ["Konzum", "Kaufland", "Lidl", "Spar", "Bingo", "Tinex", "Maxi", "Voli", "Plodine", "Mercator"]

    try:
        q = _safe_query(client, "products", "store_name,country_code")
        if country_code:
            q = q.eq("country_code", country_code)
        resp = q.limit(5000).execute()
        if not resp.data:
            return []
        df = pd.DataFrame(resp.data)
        return sorted(df["store_name"].dropna().unique().tolist())
    except Exception:
        return []


def get_distinct_categories(country_code: str | None = None) -> list[str]:
    """Alle Kategorien L1 für den Dropdown-Filter."""
    client = get_client()
    if client is None:
        return ["Hrana", "Piće", "Kozmetika", "Other", "Food"]

    try:
        q = _safe_query(client, "products", "category_l1,country_code")
        if country_code:
            q = q.eq("country_code", country_code)
        resp = q.not_.is_("category_l1", "null").limit(5000).execute()
        if not resp.data:
            return []
        df = pd.DataFrame(resp.data)
        counts = df["category_l1"].value_counts()
        return counts.index.tolist()
    except Exception:
        return []


# ── Normalisierung ────────────────────────────────────────────────────────────

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column types, prices, and retailer names."""
    for col in ["valid_from", "valid_until"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["price", "original_price", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize store_name to canonical form (deduplicates BIPA/Bipa/bipa etc.)
    if "store_name" in df.columns:
        df["store_name"] = df["store_name"].apply(
            lambda x: normalize_retailer_name(x) if pd.notna(x) else x
        )

    # Preise auf EUR normieren (Währungsspalte bleibt für Anzeige erhalten)
    if "currency" in df.columns and "price" in df.columns:
        fx = df["currency"].map(_FX_TO_EUR).fillna(1.0)
        df["price_eur"] = (df["price"] * fx).round(2)
        if "original_price" in df.columns:
            df["original_price_eur"] = (df["original_price"] * fx).round(2)
    else:
        df["price_eur"] = df.get("price", np.nan)
        df["original_price_eur"] = df.get("original_price", np.nan)

    # Discount-Tiefe auf EUR-Basis
    if "price_eur" in df.columns and "original_price_eur" in df.columns:
        df["discount_depth"] = (
            1 - df["price_eur"] / df["original_price_eur"].replace(0, np.nan)
        ).clip(0, 1).round(4)
    else:
        df["discount_depth"] = np.nan

    return df


def to_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """Konvertiert Supabase-Produkt-DF ins Format für features.py/model.py.

    Mappt die echten Spaltennamen auf das interne Feature-Schema.
    """
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df.get("valid_from", pd.Series(dtype="datetime64[ns]")))
    out["product"] = df.get("name", "")
    out["retailer"] = df.get("store_name", "")
    out["price_promo"] = pd.to_numeric(df.get("price", np.nan), errors="coerce")
    out["price_regular"] = pd.to_numeric(df.get("original_price", np.nan), errors="coerce")

    # is_on_promo: 1 wenn discount_label oder promotion_type gesetzt oder price < original
    has_discount_label = df.get("discount_label", pd.Series(dtype=object)).notna()
    has_promo_type = df.get("promotion_type", pd.Series(dtype=object)).notna()
    lower_price = (out["price_promo"] < out["price_regular"].fillna(np.inf))
    out["is_on_promo"] = (has_discount_label | has_promo_type | lower_price).astype(int)
    out["country_code"] = df.get("country_code", "")
    out["category_l1"] = df.get("category_l1", "")

    return out.dropna(subset=["date", "price_promo"])


# ── Mock-Daten ────────────────────────────────────────────────────────────────

def _mock_products(n: int = 20) -> pd.DataFrame:
    """Synthetische Produktdaten für Offline-Betrieb."""
    rng = np.random.default_rng(42)
    names = [
        "Milka Schokolade 300g", "Coca-Cola 1,5L", "Haribo Goldbären 200g",
        "Nutella 450g", "Red Bull 250ml", "Pringles Original 185g",
        "Ariel Pods 35 Kom.", "Pampers Gr.3 44 Stk.", "Bio Vollmilch 1L",
        "Ritter Sport Voll-Nuss 100g",
    ]
    stores = ["Konzum", "Kaufland", "Lidl", "Spar", "Bingo", "Tinex", "Maxi", "Voli", "Plodine", "Mercator"]
    countries = ["HR", "MK", "SI", "RS", "BA", "ME"]
    today = pd.Timestamp.today()

    return pd.DataFrame({
        "id": [str(i) for i in range(n)],
        "name": rng.choice(names, n),
        "brand": rng.choice(["Milka", "Coca-Cola", "Haribo", "Nestlé", "P&G"], n),
        "price": rng.uniform(0.49, 9.99, n).round(2),
        "original_price": rng.uniform(1.0, 12.99, n).round(2),
        "store_name": rng.choice(stores, n),
        "country_code": rng.choice(countries, n),
        "valid_from": [today - pd.Timedelta(days=int(d)) for d in rng.integers(0, 5, n)],
        "valid_until": [today + pd.Timedelta(days=int(d)) for d in rng.integers(3, 14, n)],
        "category_l1": rng.choice(["Hrana", "Piće", "Kozmetika", "Other"], n),
        "category_l2": None,
        "unit": rng.choice(["g", "ml", "Stk", "KG"], n),
        "amount": rng.uniform(0.1, 1.5, n).round(3),
        "discount_label": rng.choice(["-20%", "-30%", None, None], n),
        "promotion_type": None,
        "approval_status": "approved",
        "image_url": None,
        "created_at": today.isoformat(),
        "discount_depth": rng.uniform(0.05, 0.40, n).round(4),
    })


def _mock_promo_history() -> pd.DataFrame:
    """Synthetische Promo-Historie für Demo-Modus.

    Erzeugt 6 Produkt × 3 Händler-Kombinationen mit jeweils 5–8 historischen
    Aktionen über die letzten 12 Monate, damit die Forecast-Engine
    sinnvolle Zyklen erkennen kann.
    """
    rng = np.random.default_rng(2026)
    products = [
        ("Red Bull 250ml", "Red Bull", "Piće", "Getränke / Energy"),
        ("Milka Schokolade 300g", "Milka / Mondelez", "Slatkiši", "Süßwaren & Snacks"),
        ("Coca-Cola 1,5L", "Coca-Cola", "Piće", "Getränke"),
        ("Nutella 450g", "Ferrero", "Slatkiši", "Süßwaren & Snacks"),
        ("Ariel Pods 35 Kom.", "P&G", "Kozmetika", "Drogerie & Waschmittel"),
        ("Haribo Goldbären 200g", "Haribo", "Slatkiši", "Süßwaren & Snacks"),
        ("Pampers Gr.3 44 Stk.", "P&G", "Dječja hrana", "Babyprodukte"),
        ("Vegeta 500g", "Podravka", "Hrana", "Lebensmittel"),
    ]
    retailers = ["Konzum", "Kaufland", "Lidl", "Spar", "Bingo", "Tinex"]
    countries = ["HR", "SI", "BA", "RS", "MK"]

    today = pd.Timestamp.today().normalize()
    rows: list[dict] = []
    pid = 0

    for product, brand, cat_l1, cat_de in products:
        chosen_retailers = rng.choice(retailers, 3, replace=False)
        for retailer in chosen_retailers:
            country = str(rng.choice(countries))
            n_promos = int(rng.integers(5, 9))
            base_price = float(rng.uniform(1.0, 5.0))
            cycle_days = int(rng.integers(35, 60))

            for i in range(n_promos):
                # Älteste Aktion zuerst, gleichmäßige Zyklen mit Jitter
                offset = (n_promos - i) * cycle_days + int(rng.integers(-5, 5))
                start = today - pd.Timedelta(days=max(offset, 0))
                duration = int(rng.integers(5, 11))
                end = start + pd.Timedelta(days=duration)

                # Leichter Preis-Drift (steigend, fallend oder stabil je Produkt)
                drift = rng.uniform(-0.04, 0.06)
                price_factor = 1.0 + drift * (i / max(n_promos - 1, 1))
                orig_price = round(base_price * 1.0, 2)
                promo_price = round(base_price * 0.75 * price_factor, 2)
                disc_pct = round((1 - promo_price / orig_price) * 100, 1)

                rows.append({
                    "id": f"mock-{pid}",
                    "name": product,
                    "brand": brand,
                    "price": promo_price,
                    "original_price": orig_price,
                    "price_eur": promo_price,
                    "original_price_eur": orig_price,
                    "store_name": retailer,
                    "country_code": country,
                    "valid_from": start,
                    "valid_until": end,
                    "category_l1": cat_l1,
                    "category_de": cat_de,
                    "category_l2": None,
                    "unit": "Stk",
                    "amount": 1.0,
                    "discount_label": f"-{int(disc_pct)}%",
                    "discount_pct": disc_pct,
                    "discount_depth": round(disc_pct / 100, 4),
                    "promotion_type": "promo",
                    "approval_status": "approved",
                    "image_url": None,
                    "created_at": today.isoformat(),
                    "currency": "EUR",
                })
                pid += 1

    return pd.DataFrame(rows)


def _mock_price_history(product_name: str = "Bio Vollmilch 1L") -> pd.DataFrame:
    """Synthetische Preishistorie."""
    rng = np.random.default_rng(7)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30, freq="W")
    return pd.DataFrame({
        "product_name": product_name,
        "retailer": rng.choice(["Konzum", "Spar", "Kaufland", "Lidl", "Bingo"], len(dates)),
        "country_code": "HR",
        "price": (1.29 + rng.uniform(-0.3, 0.4, len(dates))).round(2),
        "recorded_at": dates,
    })
