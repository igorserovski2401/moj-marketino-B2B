"""Data Quality Pipeline – Brand-Mapping, Kategorie-Normierung, Preis-Validierung.

Wird nach jedem DataFrame-Load aus Supabase angewendet, bevor Daten
im Dashboard erscheinen. Gibt ausschließlich logisch korrekte Datensätze zurück.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

# ── Brand Anchors ─────────────────────────────────────────────────────────────
# Schlüsselwörter (lowercase) → kanonischer Markenname.
# Reihenfolge ist wichtig: spezifischere Patterns zuerst.

BRAND_ANCHORS: list[tuple[str, str]] = [
    # Getränke
    ("red bull",         "Red Bull"),
    ("coca-cola",        "Coca-Cola"),
    ("coca cola",        "Coca-Cola"),
    ("fanta",            "Coca-Cola"),
    ("sprite",           "Coca-Cola"),
    ("pepsi",            "PepsiCo"),
    ("heineken",         "Heineken"),
    ("carlsberg",        "Carlsberg"),
    ("staropramen",      "Staropramen"),
    ("stella artois",    "AB InBev"),
    ("corona",           "AB InBev"),
    ("cockta",           "Cockta d.o.o."),
    ("radenska",         "Radenska"),
    ("jamnica",          "Jamnica"),
    ("cedevita",         "Atlantic Grupa"),
    ("multivita",        "Atlantic Grupa"),
    # Süßwaren & Snacks
    ("milka",            "Milka / Mondelez"),
    ("oreo",             "Mondelez"),
    ("toblerone",        "Mondelez"),
    ("philadelphia",     "Mondelez"),
    ("haribo",           "Haribo"),
    ("nutella",          "Ferrero"),
    ("nutela",           "Ferrero"),
    ("kinder",           "Ferrero"),
    ("ferrero",          "Ferrero"),
    ("pringles",         "Kellogg's"),
    ("ritter sport",     "Ritter Sport"),
    ("čokolino",         "Podravka"),
    ("cokolino",         "Podravka"),
    ("kraš",             "Kraš"),
    ("kras ",            "Kraš"),
    ("ledo",             "Ledo"),
    # Haushalt & Körperpflege
    ("ariel",            "P&G"),
    ("fairy",            "P&G"),
    ("lenor",            "P&G"),
    ("pampers",          "P&G"),
    ("gillette",         "P&G"),
    ("oral-b",           "P&G"),
    ("oral b",           "P&G"),
    ("braun",            "Braun / P&G"),
    ("head & shoulders", "P&G"),
    ("persil",           "Henkel"),
    ("somat",            "Henkel"),
    ("bref",             "Henkel"),
    ("finish",           "Reckitt"),
    ("vanish",           "Reckitt"),
    ("durex",            "Reckitt"),
    ("nivea",            "Beiersdorf"),
    ("dove",             "Unilever"),
    ("axe",              "Unilever"),
    ("rexona",           "Unilever"),
    ("signal",           "Unilever"),
    ("lipton",           "Unilever"),
    ("knorr",            "Unilever"),
    ("hellmann",         "Unilever"),
    ("colgate",          "Colgate-Palmolive"),
    ("palmolive",        "Colgate-Palmolive"),
    ("oriflame",         "Oriflame"),
    ("avon",             "Avon"),
    # Lebensmittel (Hersteller)
    ("vegeta",           "Podravka"),
    ("podravka",         "Podravka"),
    ("argeta",           "Atlantic Grupa"),
    ("atlantic",         "Atlantic Grupa"),
    ("nestlé",           "Nestlé"),
    ("nestle",           "Nestlé"),
    ("nescafe",          "Nestlé"),
    ("nescafé",          "Nestlé"),
    ("kit kat",          "Nestlé"),
    ("kitkat",           "Nestlé"),
    ("maggi",            "Nestlé"),
    ("maggie",           "Nestlé"),
    ("dr. oetker",       "Dr. Oetker"),
    ("dr oetker",        "Dr. Oetker"),
    ("bonduelle",        "Bonduelle"),
    ("heinz",            "Heinz"),
    ("mc cain",          "McCain"),
    ("mccain",           "McCain"),
    ("gavrilović",       "Gavrilović"),
    ("gavrilovic",       "Gavrilović"),
    ("dukat",            "Dukat"),
    ("vindija",          "Vindija"),
    ("mlinar",           "Mlinar"),
    ("perutnina",        "Perutnina Ptuj"),
    # Technik & Haushalt
    ("bosch",            "Bosch"),
    ("philips",          "Philips"),
    ("tefal",            "Tefal"),
    ("gorenje",          "Gorenje"),
    ("samsung",          "Samsung"),
    ("huawei",           "Huawei"),
    ("still",            "Still d.o.o."),
]

# ── Category Rules ────────────────────────────────────────────────────────────
# (name_keywords, raw_cat_keywords, cat_de, cat_l1_normalized)
# Reihenfolge: spezifisch → allgemein; erster Match gewinnt.

_CAT_RULES: list[tuple[list[str], list[str], str, str]] = [
    (
        ["red bull", "energy drink", "energetski", "energy"],
        [],
        "Getränke / Energy", "Piće",
    ),
    (
        ["cola", "fanta", "sprite", "pepsi", "7up", "limonada", "limonade",
         "sok ", "juice", "mineral water", "mineralna voda", "wasser",
         "voda ", "beer", "bier", "pivo ", "bira ", "wine", "vino ",
         "cocktail", "radenska", "jamnica", "cockta"],
        ["piće", "pice", "drinks", "beverages", "getränke", "пијалоци"],
        "Getränke", "Piće",
    ),
    (
        ["ariel", "persil", "pods", "waschmittel", "deterdžent", "detergent",
         "deterdzent", "prašak za", "prasak za", "fabric softener", "omekšivač",
         "omeksivac", "waschpulver", "gel za pranje", "čistilo", "cistilo",
         "wc ", "toilet cleaner", "sredstvo za"],
        ["kućanska kemija", "kucanska kemija", "sredstva za čišćenje",
         "sredstva za ciscenje", "čišćenje", "ciscenje", "household chemicals"],
        "Drogerie & Waschmittel", "Kozmetika",
    ),
    (
        ["shampoo", "šampon", "sampon", "conditioner", "hair mask",
         "gel za tuširanje", "duschgel", "shower gel", "serum za kosu",
         "hair serum", "leave-in", "curl defining", "haarspülung", "haarmaske"],
        [],
        "Haarpflege", "Kozmetika",
    ),
    (
        ["cream", "krema za", "body lotion", "body milk", "sunscreen", "spf ",
         "moisturizer", "feuchtigkeitscreme", "deodorant", "deo ", "deo-",
         "parfüm", "parfum", "perfume", "lip balm", "lippenpflege",
         "nalpice za stopala", "foot cream", "rašpica"],
        ["kozmetika", "drugstore", "drogerija", "beauty", "health"],
        "Körperpflege", "Kozmetika",
    ),
    (
        ["pampers", "windel", "pelene", "baby food", "baby nahrung",
         "milupa", "hipp", "beikost", "dječja hrana"],
        ["dječja hrana", "djecja hrana", "baby"],
        "Babyprodukte", "Dječja hrana",
    ),
    (
        ["milch", "mleko", "mlijeko", "milk", "joghurt", "jogurt", "yogurt",
         "käse", "sir ", "cheese", "butter", "maslac", "maslo",
         "skyr", "fromage"],
        ["mliječni", "mljekarstvo", "dairy"],
        "Milchprodukte", "Mliječni proizvodi",
    ),
    (
        ["fleisch", "meso", "wurst", "salami", "schinken", "šunka", "sunka",
         "kobasica", "sausage", "ham ", "chicken", "piletina", "govedina",
         "beef", "pork", "svinjetina", "argeta", "pašteta", "pasteta"],
        ["mesni", "meso", "meat", "fleisch"],
        "Fleisch & Wurst", "Mesni proizvodi",
    ),
    (
        ["tiefkühl", "gefroren", "frozen", "zamrznut", "smrznuto",
         "sladoled", "ice cream", "eis "],
        ["zamrznuta hrana", "frozen", "tiefkühl"],
        "Tiefkühlkost", "Zamrznuta hrana",
    ),
    (
        ["milka", "čokolad", "cokolad", "chocolate", "schokolade",
         "haribo", "bonbon", "candy", "gummi bear", "keks", "cookie",
         "keksi", "nutella", "toblerone", "ferrero", "wafer", "wafle",
         "chips", "čips", "cips", "pringles", "snack", "nuss ",
         "orah ", "pistaz"],
        ["slatkiši", "slatkisi", "süßwaren", "sweets", "snacks", "čips"],
        "Süßwaren & Snacks", "Slatkiši",
    ),
    (
        ["kaffee", "kava ", "kahva", "coffee", "espresso", "cappuccino",
         "tee ", "čaj ", "caj ", "tea ", "kakao", "cocoa"],
        ["kava i čaj", "kaffee", "coffee", "tea"],
        "Kaffee & Tee", "Kava i čaj",
    ),
    (
        ["pasta", "nudeln", "tjestenina", "spaghetti", "riža ", "rice ",
         "mehl ", "brašno", "brasno", "flour ", "reis ", "couscous"],
        ["teigwaren", "nudeln"],
        "Teigwaren & Grundnahrung", "Hrana",
    ),
    (
        ["brot ", "kruh ", "bread", "toast", "baguette", "croissant",
         "burek", "pecivo", "pita "],
        ["pekarski", "bakery"],
        "Backwaren", "Pekarski proizvodi",
    ),
    (
        ["voće", "voce", "povrće", "povrce", "obst", "gemüse", "salat",
         "tomato", "tomat ", "jabuka", "banana", "jagoda", "limun",
         "paprika", "krastavac", "luk ", "kelj", "brokula", "kupus"],
        ["voće i povrće", "voce i povrce"],
        "Obst & Gemüse", "Voće i povrće",
    ),
    # Fallback-Kategorien für roh-Kategorie-Matching
    ([], ["hrana", "food", "lebensmittel"],       "Lebensmittel",        "Hrana"),
    ([], ["piće", "pice", "drinks", "beverages"], "Getränke",            "Piće"),
    ([], ["kozmetika", "drugstore", "drogerija"],  "Kosmetik & Pflege",   "Kozmetika"),
    ([], ["household", "kućanstvo", "kucanstvo",
          "home & garden", "dom i vrt",
          "proizvodi za kućanstvo"],               "Haushalt & Garten",   "Kućanstvo"),
    ([], ["electronics", "elektronika"],           "Elektronik",          "Elektronika"),
    ([], ["fashion", "odjeća", "odjeca"],          "Mode",                "Odjeća"),
    ([], ["pets", "kućni ljubimci"],               "Tiernahrung",         "Kućni ljubimci"),
    ([], ["other", "ostalo"],                      "Sonstiges",           "Other"),
]


# ── Public helpers ─────────────────────────────────────────────────────────────

class QualityReport(NamedTuple):
    n_total:        int
    n_brand_fixed:  int
    n_cat_fixed:    int
    n_price_swapped: int
    n_excluded:     int
    n_clean:        int


def run_quality_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    """Vollständige Qualitätspipeline: Brand → Kategorie → Preis.

    Returns:
        (bereinigter_df, QualityReport)
    """
    if df.empty:
        return df, QualityReport(0, 0, 0, 0, 0, 0)

    df = df.copy()
    n_total = len(df)

    df, n_brand  = _fix_brands(df)
    df, n_cat    = _fix_categories(df)
    df, n_swap, n_excl = _validate_prices(df)

    n_clean = len(df)
    return df, QualityReport(n_total, n_brand, n_cat, n_swap, n_excl, n_clean)


# ── Internal pipeline steps ───────────────────────────────────────────────────

def _fix_brands(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Überschreibt falsche/fehlende Marken anhand von Produktnamen-Keywords."""
    if "name" not in df.columns:
        return df, 0

    name_lower = df["name"].fillna("").str.lower()
    n_fixed = 0

    if "brand" not in df.columns:
        df["brand"] = pd.NA

    for keyword, brand_override in BRAND_ANCHORS:
        mask = name_lower.str.contains(keyword, regex=False)
        if not mask.any():
            continue
        # Override: Marke fehlt ODER enthält nicht den ersten Token des Ziel-Brands
        target_token = brand_override.lower().split()[0]
        already_correct = df["brand"].fillna("").str.lower().str.contains(
            target_token, regex=False
        )
        needs_fix = mask & ~already_correct
        if needs_fix.any():
            df.loc[needs_fix, "brand"] = brand_override
            n_fixed += int(needs_fix.sum())

    return df, n_fixed


def _fix_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Normiert Kategorien anhand von Produkt-Keywords und Roh-Kategorie."""
    if "name" not in df.columns and "category_l1" not in df.columns:
        return df, 0

    name_lower = df["name"].fillna("").str.lower() if "name" in df.columns else pd.Series("", index=df.index)
    cat_lower  = df["category_l1"].fillna("").str.lower() if "category_l1" in df.columns else pd.Series("", index=df.index)

    if "category_de" not in df.columns:
        df["category_de"] = pd.NA
    if "category_l1_norm" not in df.columns:
        df["category_l1_norm"] = df.get("category_l1", pd.NA)

    already_set = pd.Series(False, index=df.index)
    n_fixed = 0

    # Pass 1: Produktname-Keywords (höchste Priorität)
    for name_kws, _cat_kws, cat_de_val, cat_l1_val in _CAT_RULES:
        if not name_kws or already_set.all():
            continue
        name_hit = pd.Series(False, index=df.index)
        for kw in name_kws:
            name_hit |= name_lower.str.contains(kw, regex=False)
        match = name_hit & ~already_set
        if match.any():
            df.loc[match, "category_de"]      = cat_de_val
            df.loc[match, "category_l1_norm"] = cat_l1_val
            already_set |= match
            n_fixed += int(match.sum())

    # Pass 2: Roh-Kategorie-Fallback für noch unzugeordnete Zeilen
    for _name_kws, cat_kws, cat_de_val, cat_l1_val in _CAT_RULES:
        if not cat_kws or already_set.all():
            continue
        cat_hit = pd.Series(False, index=df.index)
        for kw in cat_kws:
            cat_hit |= cat_lower.str.contains(kw, regex=False)
        match = cat_hit & ~already_set
        if match.any():
            df.loc[match, "category_de"]      = cat_de_val
            df.loc[match, "category_l1_norm"] = cat_l1_val
            already_set |= match
            n_fixed += int(match.sum())

    # Fehlende: Fallback auf Roh-Kategorie-Text
    still_missing = ~already_set & df["category_de"].isna()
    df.loc[still_missing, "category_de"] = df.loc[still_missing, "category_l1"].fillna("—")

    return df, n_fixed


def _validate_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Preis-Sanity-Check: Tauscht vertauschte Preise, entfernt unmögliche Rabatte.

    Returns:
        (df, n_swapped, n_excluded)
    """
    p_col  = "price_eur"      if "price_eur"      in df.columns else "price"
    op_col = "original_price_eur" if "original_price_eur" in df.columns else "original_price"

    if p_col not in df.columns:
        return df, 0, 0

    df = df.copy()
    n_swapped = 0
    n_excluded = 0

    if op_col in df.columns:
        both_valid = df[p_col].notna() & df[op_col].notna() & (df[op_col] > 0)

        # Promo > Regulär → wahrscheinlich vertauscht (Extraktionsfehler)
        inverted = both_valid & (df[p_col] > df[op_col] * 1.05)  # 5 % Toleranz
        if inverted.any():
            tmp = df.loc[inverted, p_col].copy()
            df.loc[inverted, p_col]  = df.loc[inverted, op_col]
            df.loc[inverted, op_col] = tmp
            n_swapped = int(inverted.sum())

        # Rabattprozentsatz neu berechnen
        has_orig = df[op_col].notna() & (df[op_col] > 0)
        df["discount_pct"] = np.where(
            has_orig,
            ((df[op_col] - df[p_col]) / df[op_col] * 100).round(1),
            np.nan,
        )

        # Datensätze ausschließen: negativer Rabatt nach Tausch
        # (Promo immer noch > Regulär ohne plausiblen Grund)
        impossible = df["discount_pct"].notna() & (df["discount_pct"] < -5)
        n_excluded = int(impossible.sum())
        df = df[~impossible].copy()

        # discount_depth aktualisieren
        df["discount_depth"] = np.where(
            df["discount_pct"].notna(),
            (df["discount_pct"] / 100).clip(0, 1).round(4),
            df.get("discount_depth", np.nan),
        )
    else:
        df["discount_pct"] = np.nan

    return df, n_swapped, n_excluded


# ── Preis-Validierung (Einzelzeile) ──────────────────────────────────────────

def validate_promo_price(row: "pd.Series") -> str:
    """Klassifiziert die Preisqualität einer einzelnen Zeile.

    Returns:
        "valid_discount"   – Promo < Regulär, echter Rabatt
        "no_real_discount" – Promo ≥ Regulär (kein echtes Angebot)
        "missing_price"    – Preis-Felder fehlen
        "invalid_price"    – Negativer oder Null-Preis
    """
    p_col  = "price_eur"          if "price_eur"          in row.index else "price"
    op_col = "original_price_eur" if "original_price_eur" in row.index else "original_price"

    price    = row.get(p_col)
    orig     = row.get(op_col)

    if pd.isna(price):
        return "missing_price"
    if price <= 0:
        return "invalid_price"
    if pd.isna(orig) or orig <= 0:
        return "missing_price"
    if price < orig * 0.95:
        return "valid_discount"
    return "no_real_discount"
