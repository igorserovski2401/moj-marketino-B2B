"""moj-marketino – B2B Promo Intelligence Platform.

Dashboard für Key Account Manager (KAM) von FMCG-Herstellern am Balkanmarkt.
Datenquelle: Supabase MarketinoDATABASE (997k+ Produkte, 6 Länder).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.quality import run_quality_pipeline

sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings
from src.database import (
    COUNTRY_NAMES,
    get_category_distribution,
    get_client,
    get_distinct_categories,
    get_distinct_stores,
    get_overview_stats,
    get_retailer_distribution,
    load_active_promos,
    load_price_history,
    load_products,
    load_promo_history_for_forecast,
    load_upcoming_promos,
    to_feature_df,
)
from src.features import create_features
from src.forecasting import (
    MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER,
    MIN_HISTORY_DAYS,
    MIN_OBSERVATIONS,
    apply_forecast_filters,
    build_forecasts_from_promo_history,
    compute_data_quality_score,
    forecasts_to_dataframe,
    validate_price_series,
)
from src.i18n import (
    DEFAULT_LANGUAGE_BY_MARKET,
    LANG_LABELS,
    MARKET_CURRENCY,
    SUPPORTED_LANGS,
    format_price,
    get_market_currency,
    t,
    translate_category,
)
from src.matching import MIN_MATCH_SCORE, UNKNOWN_MASTER_PRODUCT, ProductMatcher
from src.model import compute_backtest_mape, forecast_price_trend, get_feature_importance, predict, train_lgbm

# ── Build & runtime identity ──────────────────────────────────────────────────
import os
import re
import unicodedata

APP_VERSION: str = os.getenv("GIT_COMMIT_SHA", "")
if not APP_VERSION:
    try:
        from subprocess import check_output, DEVNULL
        APP_VERSION = check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent.parent), stderr=DEVNULL,
        ).decode().strip() or "local-dev"
    except Exception:
        APP_VERSION = "local-dev"

CACHE_VERSION: str = APP_VERSION

# ── Country/category helpers ──────────────────────────────────────────────────

COUNTRY_KEYS: list[str] = ["HR", "SI", "BA", "RS", "MK", "ME"]


def _country_label(code: str, lang: str) -> str:
    return t(f"country.{code}", lang)


def cat_local(name: str | None, lang: str = "EN") -> str:
    if not name:
        return "—"
    return translate_category(name, lang)


MAX_CALENDAR_ROWS = 500
MIN_TREEMAP_ROWS = 5


def _apply_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run quality pipeline, return cleaned data + report dict."""
    if df.empty:
        return df, {"n_total": 0, "n_brand_fixed": 0, "n_cat_fixed": 0,
                    "n_price_swapped": 0, "n_excluded": 0, "n_clean": 0}
    clean, report = run_quality_pipeline(df)
    return clean, report._asdict()


# ── Central filter pipeline – SINGLE source of truth for ALL components ──────

SEARCHABLE_FIELDS: list[str] = [
    "brand", "brand_clean", "manufacturer", "manufacturer_clean",
    "product", "product_name", "name", "raw_product_name",
    "normalized_product_name", "master_product", "description",
]


def normalize_search_text(value) -> str:
    """Lowercase + Unicode NFKD-strip + whitespace-collapse for fuzzy match."""
    if value is None:
        return ""
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", value)


def apply_brand_product_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Multi-column normalized search across all product/brand fields."""
    q = normalize_search_text(query)
    if not q or df is None or df.empty:
        return df
    existing = [c for c in SEARCHABLE_FIELDS if c in df.columns]
    if not existing:
        return df
    mask = pd.Series(False, index=df.index)
    for col in existing:
        mask = mask | df[col].apply(lambda x: q in normalize_search_text(x))
    return df[mask]


def build_filtered_view(
    raw_df: pd.DataFrame,
    *,
    country: str | None = None,
    category: str | None = None,
    retailer: str | None = None,
    brand_query: str = "",
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Single source of truth — every visible widget MUST use this pipeline.

    Returns (filtered_df, audit_trace) where audit_trace records row counts
    after each filter step. The audit is what gets shown in the debug expander.
    """
    audit = {
        "raw_rows": 0, "after_quality": 0, "after_market": 0,
        "after_category": 0, "after_retailer": 0, "after_brand": 0,
        "final": 0,
    }
    if raw_df is None or raw_df.empty:
        return (raw_df.copy() if raw_df is not None else pd.DataFrame()), audit

    audit["raw_rows"] = len(raw_df)
    df, _ = _apply_quality(raw_df)
    audit["after_quality"] = len(df)

    if country and "country_code" in df.columns:
        df = df[df["country_code"] == country]
    audit["after_market"] = len(df)

    if category and "category_l1" in df.columns:
        df = df[df["category_l1"] == category]
    audit["after_category"] = len(df)

    if retailer and "store_name" in df.columns:
        df = df[df["store_name"] == retailer]
    audit["after_retailer"] = len(df)

    if brand_query:
        df = apply_brand_product_search(df, brand_query)
    audit["after_brand"] = len(df)

    if limit is not None:
        df = df.head(limit)
    audit["final"] = len(df)
    return df, audit


def validate_forecast_selection(
    product: str | None,
    retailer: str | None,
    market: str | None,
    currency: str | None,
) -> str:
    """Returns 'ok' or a missing_* status. Use as a gate before running Prophet."""
    if not product:
        return "missing_product"
    if not retailer or retailer == "__all__":
        return "missing_retailer"
    if not market or market == "__all__":
        return "missing_market"
    if not currency:
        return "missing_currency"
    return "ok"


def compute_trust_level(mape: float, observations: int, history_days: int) -> str:
    """Return trust level key ('belastbar' / 'eingeschr' / 'nicht_belastbar')."""
    import math
    if observations < MIN_OBSERVATIONS or history_days < MIN_HISTORY_DAYS:
        return "nicht_belastbar"
    if math.isnan(mape):
        return "eingeschr"
    if mape <= 0.20:
        return "belastbar"
    if mape <= 0.35:
        return "eingeschr"
    return "nicht_belastbar"


def build_clean_view_df(
    raw_df: pd.DataFrame,
    country: str | None = None,
    category: str | None = None,
    retailer: str | None = None,
    brand_search: str = "",
    limit: int | None = None,
) -> pd.DataFrame:
    """Einheitliche Filter-Pipeline für alle Tabs und KPIs.

    Reihenfolge: quality → country → category → retailer → brand → limit.
    Alle Tabs verwenden diese Funktion um Konsistenz sicherzustellen.
    """
    if raw_df.empty:
        return raw_df
    df, _ = _apply_quality(raw_df)

    if country and "country_code" in df.columns:
        df = df[df["country_code"] == country]
    if category and "category_l1" in df.columns:
        df = df[df["category_l1"] == category]
    if retailer and "store_name" in df.columns:
        df = df[df["store_name"] == retailer]
    if brand_search:
        search_cols = [c for c in ["brand", "manufacturer", "name", "master_product"] if c in df.columns]
        if search_cols:
            mask = pd.Series(False, index=df.index)
            for col in search_cols:
                mask |= df[col].fillna("").str.contains(brand_search, case=False, regex=False)
            df = df[mask]
    if limit is not None:
        df = df.head(limit)
    return df


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="moj-marketino | Promo Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#1A56DB"
ACCENT  = "#F05252"
GREEN   = "#059669"
AMBER   = "#D97706"

st.markdown(
    f"""
    <style>
      html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
      .block-container {{ padding-top: 1rem; }}
      .kpi-card {{
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        border: 1px solid #E5E7EB;
        border-top: 4px solid {PRIMARY};
        box-shadow: 0 1px 6px rgba(0,0,0,.06);
        height: 100%;
      }}
      .kpi-card-green {{ border-top-color: {GREEN} !important; }}
      .kpi-card-red   {{ border-top-color: {ACCENT} !important; }}
      .kpi-card-amber {{ border-top-color: {AMBER} !important; }}
      .kpi-label {{ font-size:.73rem; color:#6B7280; font-weight:700;
                    text-transform:uppercase; letter-spacing:.07em; }}
      .kpi-value {{ font-size:2rem; font-weight:800; color:#111827; line-height:1.15; }}
      .kpi-delta {{ font-size:.8rem; color:#6B7280; margin-top:.2rem; }}
      .badge-ok    {{ background:#D1FAE5; color:#065F46; border-radius:6px;
                      padding:3px 10px; font-size:.75rem; font-weight:700; }}
      .badge-warn  {{ background:#FEF3C7; color:#92400E; border-radius:6px;
                      padding:3px 10px; font-size:.75rem; font-weight:700; }}
      .badge-info  {{ background:#DBEAFE; color:#1E40AF; border-radius:6px;
                      padding:3px 10px; font-size:.75rem; font-weight:700; }}
      .section-header {{
        font-size: 1.05rem; font-weight: 700; color: #111827;
        border-left: 4px solid {PRIMARY}; padding-left: .6rem;
        margin-bottom: .4rem;
      }}
      div[data-testid="stMetric"] label {{ font-size: .78rem !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

db_connected = settings.has_supabase and get_client() is not None
demo_mode = not db_connected
production_mode = db_connected

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:.4rem 0 .8rem;'>
          <span style='font-size:1.6rem;'>📊</span><br>
          <strong style='font-size:1.1rem; color:#111827;'>moj-marketino</strong><br>
          <span style='font-size:.78rem; color:#6B7280;'>B2B Promo Intelligence</span><br>
          <span style='font-size:.7rem; color:#9CA3AF;'>v2.3 · Balkan Edition</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Market FIRST so we can derive default language
    _country_keys_extended = ["__all__"] + COUNTRY_KEYS

    if "sel_country" not in st.session_state:
        st.session_state["sel_country"] = "__all__"

    # Language with default-by-market (only if not manually overridden)
    if "ui_lang_manual" not in st.session_state:
        st.session_state["ui_lang_manual"] = False

    _country = st.session_state["sel_country"]
    if not st.session_state["ui_lang_manual"]:
        st.session_state["ui_lang"] = DEFAULT_LANGUAGE_BY_MARKET.get(
            _country if _country != "__all__" else "", "EN"
        )

    current_lang = st.session_state.get("ui_lang", "EN")

    def _on_lang_change():
        st.session_state["ui_lang_manual"] = True

    st.divider()

    # Build / Deploy proof block — always visible
    _mode_label = t("build.production", current_lang) if production_mode else t("build.demo", current_lang)
    _source_label = t("build.supabase", current_lang) if production_mode else t("build.mock", current_lang)
    _mode_color = "#059669" if production_mode else "#D97706"
    st.markdown(
        f"""
        <div style='background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;
                    padding:.55rem .7rem;font-size:.76rem;line-height:1.45;color:#374151;'>
          <div><b>{t('build.version', current_lang)}:</b> <code>{APP_VERSION}</code></div>
          <div><b>{t('build.mode', current_lang)}:</b>
               <span style='color:{_mode_color};font-weight:700;'>{_mode_label}</span></div>
          <div><b>{t('build.data_source', current_lang)}:</b> {_source_label}</div>
          <div><b>{t('build.language', current_lang)}:</b> {LANG_LABELS.get(current_lang, current_lang)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if production_mode:
        st.markdown('<br><span class="badge-ok">● Live data active</span>', unsafe_allow_html=True)
        st.caption("MarketinoDATABASE · EU-Central-2")
    else:
        st.markdown(
            '<br><span class="badge-warn">⚠ DEMO MODE – sample data only</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    lang_idx = st.selectbox(
        "🌐 " + t("sidebar.language", current_lang) + " / Language",
        range(len(SUPPORTED_LANGS)),
        format_func=lambda i: LANG_LABELS[SUPPORTED_LANGS[i]],
        index=SUPPORTED_LANGS.index(current_lang) if current_lang in SUPPORTED_LANGS else 0,
        key="lang_picker",
        on_change=_on_lang_change,
    )
    ui_lang = SUPPORTED_LANGS[lang_idx]
    if st.session_state["ui_lang_manual"]:
        st.session_state["ui_lang"] = ui_lang
    # Re-read effective language
    current_lang = st.session_state.get("ui_lang", ui_lang)
    ui_lang = current_lang

    st.markdown(f"### 🔍 {t('sidebar.filter', ui_lang)}")

    country_labels = [t("sidebar.all_markets", ui_lang)] + [
        t(f"country.{k}", ui_lang) for k in COUNTRY_KEYS
    ]
    sel_country_idx = st.selectbox(
        t("sidebar.market", ui_lang),
        range(len(_country_keys_extended)),
        format_func=lambda i: country_labels[i],
        index=_country_keys_extended.index(st.session_state["sel_country"])
            if st.session_state["sel_country"] in _country_keys_extended else 0,
        key="market_picker",
    )
    sel_country = None if sel_country_idx == 0 else _country_keys_extended[sel_country_idx]
    st.session_state["sel_country"] = sel_country if sel_country else "__all__"

    market_currency = get_market_currency(sel_country)

    @st.cache_data(ttl=600, show_spinner=False)
    def _sidebar_cats(c, _v=CACHE_VERSION):
        return [t("sidebar.all_categories", "EN")] + get_distinct_categories(c)

    sidebar_cats = _sidebar_cats(sel_country)
    sel_cat_idx = st.selectbox(
        t("sidebar.category", ui_lang),
        range(len(sidebar_cats)),
        format_func=lambda i: (
            f"{translate_category(sidebar_cats[i], ui_lang)}"
            if i > 0 else sidebar_cats[0]
        ),
    )
    sel_cat = None if sel_cat_idx == 0 else sidebar_cats[sel_cat_idx]

    brand_filter = st.text_input(
        t("sidebar.brand", ui_lang),
        placeholder="Podravka, Milka, Wudy…",
    )

    with st.expander(t("section.advanced_settings", ui_lang)):
        match_threshold = st.slider(
            "Match confidence (min. 85 %)",
            min_value=MIN_MATCH_SCORE,
            max_value=0.99,
            value=MIN_MATCH_SCORE,
            step=0.05,
        )
        kw_vorschau = st.slider("Preview weeks", 1, 8, 4)

    st.divider()
    st.caption(f"© 2026 moj-marketino · Build {APP_VERSION}")

# ── Header ────────────────────────────────────────────────────────────────────

market_label = t(f"country.{sel_country}", ui_lang) if sel_country else t("sidebar.all_markets", ui_lang)

# Demo mode banner – prominent, impossible to miss
if demo_mode:
    st.warning(
        "⚠️ **DEMO MODE** – This dashboard is running on sample data only. "
        "No real market data is displayed. Connect a Supabase database to activate live data.",
        icon="⚠️",
    )

st.markdown(
    f"""
    <h1 style='margin-bottom:.1rem; font-size:1.7rem; font-weight:800; color:#111827;'>
      📊 {t('app.title', ui_lang)}
    </h1>
    <p style='color:#6B7280; margin-top:.1rem; font-size:.92rem;'>
      {t('app.subtitle', ui_lang)} · {t('sidebar.market', ui_lang)}: <strong>{market_label}</strong>
    </p>
    """,
    unsafe_allow_html=True,
)

# ── KPI-Zeile ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _cached_stats(country: str | None) -> dict:
    return get_overview_stats(country)

stats = _cached_stats(sel_country)

k1, k2, k3, k4, k5 = st.columns(5)
_kpis = [
    (k1, "Beobachtete Artikel",       f"{stats['total_products']:,}".replace(",", "."), "MarketinoDB"),
    (k2, "Aktive Händler",            str(stats["total_retailers"]),                    "Balkan-Märkte"),
    (k3, "Flugblätter / Kataloge",    str(stats["total_catalogs"]),                     "Prospekte"),
    (k4, "Märkte",                    str(stats["total_countries"]),                    "HR · SI · BA · RS · MK · ME"),
    (k5, "Aktive Promotionen (KW)",   str(stats["active_promos"]),                     "Laufende Aktionen"),
]
for col, label, val, sub in _kpis:
    col.markdown(
        f"""<div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-delta">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )
st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    t("tab.predictor", ui_lang),
    t("tab.price", ui_lang),
    t("tab.quality", ui_lang),
])

# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Promotion Predictor
# ╚══════════════════════════════════════════════════════════════════════════════

with tab1:
    days_ahead = kw_vorschau * 7

    # ╭────────────────────────────────────────────────────────────────────────╮
    # │  TEIL 1 · KAM-Forecast (regelbasiert, historiengestützt)              │
    # ╰────────────────────────────────────────────────────────────────────────╯
    st.markdown(
        f'<div class="section-header">🎯 {t("kam.title", ui_lang)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        t("kam.min_req", ui_lang,
          n=MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER,
          d=MIN_HISTORY_DAYS)
    )

    # ── Filterleiste ─────────────────────────────────────────────────────────
    with st.container():
        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
        with fc1:
            product_query = st.text_input(
                t("kam.product_search", ui_lang),
                value="",
                placeholder="Milka, Coca-Cola, Red Bull…",
                key="kam_product_query",
            )
        with fc2:
            @st.cache_data(ttl=600, show_spinner=False)
            def _fc_stores(c):
                return [t("kam.all_retailers", "EN")] + get_distinct_stores(c)
            fc_stores = _fc_stores(sel_country)
            fc_store_idx = st.selectbox(
                t("kam.retailer", ui_lang), range(len(fc_stores)),
                format_func=lambda i: fc_stores[i], key="kam_store",
            )
            fc_retailer = None if fc_store_idx == 0 else fc_stores[fc_store_idx]
        with fc3:
            min_probability = st.slider(
                t("kam.min_probability", ui_lang),
                0.0, 0.95, 0.50, 0.05, key="kam_min_prob",
            )
        with fc4:
            pred_window = st.slider(
                t("kam.prediction_window", ui_lang),
                7, 90, 30, 7, key="kam_pred_window",
            )

        with st.expander("⚙️ Advanced forecast filters"):
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                _signal_options_raw = ["Alle", "Hoch relevant", "Beobachten", "Normal"]
                sel_signal = st.selectbox(
                    t("kam.signal", ui_lang),
                    _signal_options_raw,
                    key="kam_signal",
                )
            with ac2:
                sel_trend = st.selectbox(
                    t("kam.price_trend", ui_lang),
                    ["Alle", "steigend", "fallend", "stabil", "unbekannt"],
                    key="kam_trend",
                )
            with ac3:
                only_future = st.checkbox(
                    t("kam.only_future", ui_lang), value=True, key="kam_only_future",
                )
            with ac4:
                only_sufficient = st.checkbox(
                    t("kam.only_sufficient", ui_lang),
                    value=True, key="kam_only_sufficient",
                )

    # ── Load history and apply central filter pipeline ──────────────────────
    @st.cache_data(ttl=300, show_spinner=False)
    def _load_forecast_history(country, cat, _v=CACHE_VERSION):
        return load_promo_history_for_forecast(
            country_code=country, category_l1=cat, days_back=365, limit=5000,
        )

    with st.spinner(t("general.loading", ui_lang)):
        history_raw = _load_forecast_history(sel_country, sel_cat)
        # Central filter — brand search applies HERE too, not just to upcoming.
        history_df, _hist_audit = build_filtered_view(
            history_raw,
            country=sel_country,
            category=sel_cat,
            brand_query=brand_filter,
        )
        forecasts = build_forecasts_from_promo_history(history_df, lang=ui_lang)
        fc_df_all = forecasts_to_dataframe(forecasts)

    fc_df = apply_forecast_filters(
        fc_df_all,
        retailer=fc_retailer,
        product_query=product_query,
        min_probability=min_probability,
        signal=sel_signal,
        price_trend=sel_trend,
        only_future=only_future,
        prediction_window_days=pred_window,
    )

    # Optionaler Status-Filter für "ungenügende Historie"
    if not only_sufficient:
        # Bei deaktiviertem Filter: zeige zusätzlich Forecasts mit Status != ok
        # (Datenquelle ist trotzdem auf forecasts beschränkt → kommt aus build_forecasts_from_promo_history,
        #  also bereits ≥3 Historien-Promos. Andere insufficient_history-Fälle sind hier ausgeschlossen.)
        pass

    # ── KAM-KPI-Karten ───────────────────────────────────────────────────────
    n_high = int((fc_df["signal"] == "Hoch relevant").sum()) if not fc_df.empty else 0
    n_rising = int((fc_df["price_trend"] == "steigend").sum()) if not fc_df.empty else 0
    n_overdue = 0
    if not fc_df.empty:
        n_overdue = int(
            fc_df.apply(
                lambda r: (
                    r.get("avg_cycle_days") is not None
                    and r.get("days_since_last_promo") is not None
                    and r["days_since_last_promo"] > r["avg_cycle_days"] * 1.2
                ),
                axis=1,
            ).sum()
        )
    n_excluded_no_hist = max(0, len(history_df) - sum(f.historical_count for f in forecasts)) if forecasts else 0
    avg_disc = float("nan")
    if not fc_df.empty and "typical_discount_pct_max" in fc_df.columns:
        avg_disc = fc_df["typical_discount_pct_max"].dropna().mean()

    kk1, kk2, kk3, kk4, kk5 = st.columns(5)
    kk1.metric(t("kpi.high_signal", ui_lang), str(n_high))
    kk2.metric(t("kpi.rising_price", ui_lang), str(n_rising))
    kk3.metric(t("kpi.overdue", ui_lang), str(n_overdue))
    kk4.metric(t("kpi.forecast_base", ui_lang), f"{len(fc_df_all)}")
    kk5.metric(t("kpi.avg_discount", ui_lang), f"{avg_disc:.0f} %" if pd.notna(avg_disc) else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KAM-Tabelle ──────────────────────────────────────────────────────────
    if fc_df.empty:
        if fc_df_all.empty:
            st.info(
                "🔍 **Keine Produkt-Händler-Kombination hat ausreichende Historie für eine belastbare Prognose.**  \n"
                f"Voraussetzung: mind. {MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER} historische Aktionen "
                f"und {MIN_HISTORY_DAYS} Tage Historie. "
                "Erweitere den Markt-Filter oder warte auf mehr Datenpunkte."
            )
        else:
            st.info("Keine Prognosen entsprechen den aktiven Filtern. Filter zurücksetzen oder erweitern.")
    else:
        # Anzeigespalten zusammenstellen
        def _fmt_period(row):
            s = row.get("expected_start")
            e = row.get("expected_end")
            if s is None:
                return "—"
            s_str = s.strftime("%d.%m.%Y") if hasattr(s, "strftime") else str(s)
            if e is None:
                return s_str
            e_str = e.strftime("%d.%m.%Y") if hasattr(e, "strftime") else str(e)
            return f"{s_str} – {e_str}"

        _trend_labels = {
            "steigend": t("trend.rising", ui_lang),
            "fallend":  t("trend.falling", ui_lang),
            "stabil":   t("trend.stable", ui_lang),
            "unbekannt": t("trend.unknown", ui_lang),
        }

        def _fmt_trend(row):
            raw = row.get("price_trend") or "unbekannt"
            pct = row.get("price_trend_pct")
            label = _trend_labels.get(raw, raw)
            if raw == "unbekannt" or pct is None:
                return label
            arrow = "▲" if raw == "steigend" else ("▼" if raw == "fallend" else "→")
            return f"{arrow} {label} ({pct:+.1f} %)"

        _days_label = t("general.days", ui_lang)
        _median_label = t("general.median", ui_lang)

        def _fmt_last_promo(row):
            d = row.get("days_since_last_promo")
            if d is None:
                return "—"
            return t("general.days_ago", ui_lang, d=d)

        def _fmt_cycle(row):
            avg = row.get("avg_cycle_days")
            med = row.get("median_cycle_days")
            if avg is None and med is None:
                return "—"
            if med is not None:
                return f"{med} {_days_label} ({_median_label})"
            return f"{avg} {_days_label} (Ø)"

        _cur_sym = "€" if market_currency == "EUR" else market_currency

        def _signal_emoji(s: str) -> str:
            _map = {
                "Hoch relevant":   t("signal.high", ui_lang),
                "Beobachten":      t("signal.watch", ui_lang),
                "Normal":          t("signal.normal", ui_lang),
                "Nicht belastbar": t("signal.unreliable", ui_lang),
                "Ungültig":        t("signal.invalid", ui_lang),
            }
            return _map.get(s, s)

        display = fc_df.copy()
        display[t("col.period", ui_lang)]      = display.apply(_fmt_period, axis=1)
        display[t("kam.price_trend", ui_lang)] = display.apply(_fmt_trend, axis=1)
        display[t("col.last_promo", ui_lang)]  = display.apply(_fmt_last_promo, axis=1)
        display[t("col.cycle", ui_lang)]       = display.apply(_fmt_cycle, axis=1)
        display[t("kam.signal", ui_lang)]      = display["signal"].apply(_signal_emoji)
        display[t("col.probability", ui_lang)] = (display["probability"] * 100).round(0)

        _period_col   = t("col.period", ui_lang)
        _trend_col    = t("kam.price_trend", ui_lang)
        _last_col     = t("col.last_promo", ui_lang)
        _cycle_col    = t("col.cycle", ui_lang)
        _signal_col   = t("kam.signal", ui_lang)
        _prob_col     = t("col.probability", ui_lang)
        _exp_price_col  = f"{t('col.expected_price', ui_lang)} ({_cur_sym})"
        _last_price_col = f"{t('col.last_price', ui_lang)} ({_cur_sym})"

        kam_cols = [
            "priority", "product", "brand", "retailer", "country", "category",
            _period_col, _prob_col,
            "expected_price", "last_promo_price", _trend_col,
            _cycle_col, _last_col, "historical_count",
            _signal_col, "justification",
        ]
        display = display[[c for c in kam_cols if c in display.columns]]
        display = display.rename(columns={
            "priority":        t("col.priority", ui_lang),
            "product":         t("col.product", ui_lang),
            "brand":           t("col.brand", ui_lang),
            "retailer":        t("col.retailer", ui_lang),
            "country":         t("col.market", ui_lang),
            "category":        t("col.category", ui_lang),
            "expected_price":  _exp_price_col,
            "last_promo_price": _last_price_col,
            "historical_count": t("col.history_count", ui_lang),
            "justification":   t("col.justification", ui_lang),
        })

        _price_fmt = "%.2f €" if market_currency == "EUR" else f"%.2f {market_currency}"
        col_cfg = {
            _prob_col: st.column_config.ProgressColumn(
                _prob_col, min_value=0, max_value=100, format="%.0f %%",
            ),
            _exp_price_col: st.column_config.NumberColumn(
                _exp_price_col, format=_price_fmt,
            ),
            _last_price_col: st.column_config.NumberColumn(
                _last_price_col, format=_price_fmt,
            ),
        }

        st.dataframe(
            display,
            column_config=col_cfg,
            use_container_width=True, hide_index=True, height=440,
        )

        st.caption(f"📊 {len(display)} · {t('col.priority', ui_lang)} → {t('col.probability', ui_lang)}")

        # ── Detail view ──────────────────────────────────────────────────────
        st.markdown(
            f'<div class="section-header">{t("section.detail", ui_lang)}</div>',
            unsafe_allow_html=True,
        )
        fc_df["_label"] = (
            fc_df["product"].fillna("?") + " · " +
            fc_df["retailer"].fillna("?") + " (" +
            fc_df["country"].fillna("") + ")"
        )
        sel_idx = st.selectbox(
            t("detail.select_combo", ui_lang),
            options=range(len(fc_df)),
            format_func=lambda i: fc_df["_label"].iloc[i],
            key="kam_detail_select",
        )
        sel_row = fc_df.iloc[sel_idx]

        _det_cur_sym = "€" if market_currency == "EUR" else market_currency

        def _fmt_money(v):
            return f"{v:.2f} {_det_cur_sym}" if v is not None and pd.notna(v) else "—"

        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown(f"**{t('detail.product', ui_lang)}:** {sel_row['product']}")
            st.markdown(f"**{t('detail.brand', ui_lang)}:** {sel_row.get('brand') or '—'}")
            st.markdown(f"**{t('detail.retailer', ui_lang)}:** {sel_row['retailer']}")
            st.markdown(f"**{t('detail.market', ui_lang)}:** {sel_row.get('country') or '—'}")
        with d2:
            st.markdown(f"**{t('detail.price_module', ui_lang)}**")
            st.markdown(f"{t('detail.expected', ui_lang)}: **{_fmt_money(sel_row.get('expected_price'))}**")
            if pd.notna(sel_row.get("last_promo_price")):
                st.markdown(f"{t('detail.last_promo_price', ui_lang)}: **{_fmt_money(sel_row['last_promo_price'])}**")
            if pd.notna(sel_row.get("avg_promo_price_90d")):
                st.markdown(f"{t('detail.avg_90d', ui_lang)}: {_fmt_money(sel_row['avg_promo_price_90d'])}")
            if pd.notna(sel_row.get("avg_promo_price_180d")):
                st.markdown(f"{t('detail.avg_180d', ui_lang)}: {_fmt_money(sel_row['avg_promo_price_180d'])}")
            if pd.notna(sel_row.get("min_promo_price_12m")) and pd.notna(sel_row.get("max_promo_price_12m")):
                st.markdown(
                    f"{t('detail.minmax_12m', ui_lang)}: "
                    f"{_fmt_money(sel_row['min_promo_price_12m'])} / "
                    f"{_fmt_money(sel_row['max_promo_price_12m'])}"
                )
            if pd.notna(sel_row.get("price_change_vs_last_pct")):
                arrow = "▲" if sel_row['price_change_vs_last_pct'] > 0 else "▼"
                st.markdown(
                    f"{t('detail.change_vs_last', ui_lang)}: {arrow} "
                    f"{sel_row['price_change_vs_last_eur']:+.2f} {_det_cur_sym} "
                    f"({sel_row['price_change_vs_last_pct']:+.1f} %)"
                )
        with d3:
            st.markdown(f"**{t('detail.cycle_signal', ui_lang)}**")
            if sel_row.get("median_cycle_days") is not None:
                st.markdown(f"{t('detail.avg_cycle', ui_lang)}: {sel_row['median_cycle_days']} {t('general.days', ui_lang)} ({t('general.median', ui_lang)})")
            if sel_row.get("days_since_last_promo") is not None:
                st.markdown(f"{t('detail.last_promo', ui_lang)}: {t('general.days_ago', ui_lang, d=sel_row['days_since_last_promo'])}")
            if sel_row.get("typical_duration_days") is not None:
                st.markdown(f"{t('detail.typical_duration', ui_lang)}: {sel_row['typical_duration_days']} {t('general.days', ui_lang)}")
            if sel_row.get("typical_discount_pct_min") is not None and sel_row.get("typical_discount_pct_max") is not None:
                st.markdown(
                    f"{t('detail.typical_discount', ui_lang)}: "
                    f"{sel_row['typical_discount_pct_min']:.0f} – "
                    f"{sel_row['typical_discount_pct_max']:.0f} %"
                )
            _signal_map = {
                "Hoch relevant":   t("signal.high", ui_lang),
                "Beobachten":      t("signal.watch", ui_lang),
                "Normal":          t("signal.normal", ui_lang),
                "Nicht belastbar": t("signal.unreliable", ui_lang),
                "Ungültig":        t("signal.invalid", ui_lang),
            }
            _signal_disp = _signal_map.get(sel_row["signal"], sel_row["signal"])
            st.markdown(f"**{t('detail.signal', ui_lang)}:** {_signal_disp}")
            st.markdown(f"**{t('detail.probability', ui_lang)}:** {sel_row['probability']*100:.0f} %")

        st.markdown(f"**{t('detail.justification', ui_lang)}:** " + (sel_row.get("justification") or "—"))

        # Historical promos
        hist_promos = sel_row.get("last_promos") or []
        if hist_promos:
            st.markdown(f"**{t('detail.history_5', ui_lang)}**")
            hist_rows = []
            for p in hist_promos:
                start = p.get("start").strftime("%d.%m.%Y") if p.get("start") else "—"
                end = p.get("end").strftime("%d.%m.%Y") if p.get("end") else "—"
                price = _fmt_money(p.get("price"))
                disc = f"-{p['discount_pct']:.0f} %" if p.get("discount_pct") is not None else "—"
                hist_rows.append({
                    "Start": start, "End": end,
                    t("detail.last_promo_price", ui_lang): price,
                    "Rabatt": disc,
                })
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ╭────────────────────────────────────────────────────────────────────────╮
    # │  TEIL 2 · Bevorstehende Aktionen (gefiltert + Data-Quality-Badges)    │
    # ╰────────────────────────────────────────────────────────────────────────╯
    st.markdown(
        f'<div class="section-header">{t("section.upcoming", ui_lang)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(t("upcoming.caption", ui_lang, w=kw_vorschau, market=market_label))

    with st.expander(t("section.dq_filter", ui_lang), expanded=False):
        bf1, bf2, bf3 = st.columns(3)
        with bf1:
            f_only_brand = st.checkbox(t("dq.brand_missing", ui_lang) + " ✗", value=False, key="up_only_brand")
            f_only_discount = st.checkbox(t("dq.no_real_discount", ui_lang) + " ✗", value=False, key="up_only_disc")
        with bf2:
            f_only_orig = st.checkbox(t("dq.orig_price_missing", ui_lang) + " ✗", value=False, key="up_only_orig")
            f_only_real_disc = st.checkbox("≥ 5 % " + t("upcoming.avg_discount", ui_lang), value=False, key="up_only_real")
        with bf3:
            f_no_other = st.checkbox(t("dq.category_uncertain", ui_lang) + " ✗", value=False, key="up_no_other")

    @st.cache_data(ttl=180, show_spinner=False)
    def _upcoming(country, days, cat, _v=CACHE_VERSION):
        df = load_upcoming_promos(country_code=country, days_ahead=days, limit=400)
        if cat and "category_l1" in df.columns:
            df = df[df["category_l1"] == cat]
        return df

    with st.spinner(t("general.loading", ui_lang)):
        upcoming_raw = _upcoming(sel_country, days_ahead, sel_cat)
        upcoming_df, _upcoming_audit = build_filtered_view(
            upcoming_raw,
            country=sel_country,
            category=sel_cat,
            brand_query=brand_filter,
        )
        _uq = {"n_total": len(upcoming_raw), "n_excluded": len(upcoming_raw) - len(upcoming_df)}

    # Daten-Qualitäts-Filter
    if not upcoming_df.empty:
        if f_only_brand and "brand" in upcoming_df.columns:
            upcoming_df = upcoming_df[upcoming_df["brand"].fillna("").str.strip() != ""]
        if f_only_discount and "discount_pct" in upcoming_df.columns:
            upcoming_df = upcoming_df[upcoming_df["discount_pct"].fillna(0) > 0]
        if f_only_orig and "original_price_eur" in upcoming_df.columns:
            upcoming_df = upcoming_df[upcoming_df["original_price_eur"].notna()]
        if f_only_real_disc and "discount_pct" in upcoming_df.columns:
            upcoming_df = upcoming_df[upcoming_df["discount_pct"].fillna(0) >= 5]
        if f_no_other and "category_de" in upcoming_df.columns:
            upcoming_df = upcoming_df[~upcoming_df["category_de"].isin(["Sonstiges", "Other", "—", "", None])]

    if not upcoming_df.empty:
        n_up = len(upcoming_df)
        n_retailers = upcoming_df["store_name"].nunique() if "store_name" in upcoming_df.columns else 0
        ua, ub, uc, ud = st.columns(4)
        ua.metric(t("upcoming.found", ui_lang), str(n_up))
        ub.metric(t("upcoming.retailers_involved", ui_lang), str(n_retailers))
        if "discount_pct" in upcoming_df.columns:
            avg_disc = upcoming_df["discount_pct"].dropna().mean()
            uc.metric(t("upcoming.avg_discount", ui_lang), f"{avg_disc:.1f} %" if pd.notna(avg_disc) else "—")
        if _uq["n_excluded"] > 0:
            ud.metric(t("upcoming.faulty", ui_lang), str(_uq["n_excluded"]))

        # Data-Quality-Badge per row (localised)
        def _badge(row):
            badges = []
            if pd.isna(row.get("brand")) or str(row.get("brand")).strip() == "":
                badges.append(t("dq.brand_missing", ui_lang))
            if pd.isna(row.get("original_price_eur")):
                badges.append(t("dq.orig_price_missing", ui_lang))
            elif pd.isna(row.get("discount_pct")) or row.get("discount_pct", 0) < 5:
                badges.append(t("dq.no_real_discount", ui_lang))
            if row.get("category_de") in (None, "", "Sonstiges", "Other", "—"):
                badges.append(t("dq.category_uncertain", ui_lang))
            return ", ".join(badges) if badges else t("dq.complete", ui_lang)

        upcoming_df = upcoming_df.copy()
        upcoming_df["_quality"] = upcoming_df.apply(_badge, axis=1)

        st.markdown("<br>", unsafe_allow_html=True)
        _up_cols = [
            "store_name", "name", "brand", "price_eur", "original_price_eur",
            "discount_pct", "currency", "category_de", "country_code",
            "valid_from", "valid_until", "discount_label", "_quality",
        ]
        up_show = upcoming_df[[c for c in _up_cols if c in upcoming_df.columns]].copy()
        _cur_sym = "€" if market_currency == "EUR" else market_currency
        _promo_col = f"{t('forecast.price_label', ui_lang)} ({_cur_sym})"
        _orig_col = f"{t('detail.last_promo_price', ui_lang)} ({_cur_sym})"
        _disc_col = f"% {t('upcoming.avg_discount', ui_lang)}"
        up_show = up_show.rename(columns={
            "store_name":         t("detail.retailer", ui_lang),
            "name":               t("detail.product", ui_lang),
            "brand":              t("detail.brand", ui_lang),
            "price_eur":          _promo_col,
            "original_price_eur": _orig_col,
            "discount_pct":       _disc_col,
            "currency":           t("forecast.currency", ui_lang),
            "category_de":        t("detail.market", ui_lang) if False else t("col.category", ui_lang),
            "country_code":       t("detail.market", ui_lang),
            "valid_from":         "Start",
            "valid_until":        "End",
            "discount_label":     t("upcoming.avg_discount", ui_lang),
            "_quality":           t("dq.complete", ui_lang).replace("✓ ", ""),
        })

        _price_fmt = "%.2f €" if market_currency == "EUR" else f"%.2f {market_currency}"
        col_cfg = {}
        if _promo_col in up_show.columns:
            col_cfg[_promo_col] = st.column_config.NumberColumn(_promo_col, format=_price_fmt)
        if _orig_col in up_show.columns:
            col_cfg[_orig_col] = st.column_config.NumberColumn(_orig_col, format=_price_fmt)
        if _disc_col in up_show.columns:
            col_cfg[_disc_col] = st.column_config.ProgressColumn(
                _disc_col, min_value=0, max_value=70, format="%.1f %%",
            )

        st.dataframe(up_show, column_config=col_cfg, use_container_width=True, hide_index=True, height=380)

        # ── Treemap with MIN_TREEMAP_ROWS guard ───────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        _tree_cat = "category_de" if "category_de" in upcoming_df.columns else "category_l1"
        if "store_name" in upcoming_df.columns and _tree_cat in upcoming_df.columns:
            cat_quality_bad = False
            if _tree_cat in upcoming_df.columns:
                bad_cats = upcoming_df[_tree_cat].isin(["Sonstiges", "Other", "—", "", None]).sum()
                cat_quality_bad = bad_cats / max(len(upcoming_df), 1) > 0.5

            if len(upcoming_df) < MIN_TREEMAP_ROWS:
                st.info(t("charts.treemap_min_rows", ui_lang, n=MIN_TREEMAP_ROWS, got=len(upcoming_df)))
            elif len(upcoming_df) > 300 or cat_quality_bad:
                st.warning(t("charts.treemap_disabled", ui_lang))
            else:
                treemap_df = (
                    upcoming_df.groupby(["store_name", _tree_cat])
                    .size().reset_index(name="n")
                )
                if _tree_cat != "category_de":
                    treemap_df["category_de"] = treemap_df[_tree_cat].apply(
                        lambda x: cat_local(x, ui_lang)
                    )
                    _tree_cat = "category_de"
                else:
                    treemap_df["category_de"] = treemap_df["category_de"].apply(
                        lambda x: cat_local(x, ui_lang)
                    )
                fig_tree = px.treemap(
                    treemap_df,
                    path=["store_name", "category_de"],
                    values="n",
                    title=t("charts.distribution_title", ui_lang, w=kw_vorschau),
                    color="n",
                    color_continuous_scale="Blues",
                )
                fig_tree.update_layout(margin=dict(t=50, b=10))
                st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info(t("upcoming.none", ui_lang))

    st.divider()

    # ╭────────────────────────────────────────────────────────────────────────╮
    # │  TEIL 3 · LightGBM (experimentell, optional)                           │
    # ╰────────────────────────────────────────────────────────────────────────╯
    with st.expander(t("section.ml_expander", ui_lang)):
        st.caption(t("ml.caption", ui_lang))

        train_btn = st.button(t("ml.retrain", ui_lang), key="train_lgbm_btn")
        trained_model = None
        if train_btn:
            if not db_connected:
                st.warning(t("ml.no_db", ui_lang))
            else:
                with st.spinner(t("general.loading", ui_lang)):
                    raw_train = load_products(country_code=sel_country, limit=2000)
                    if len(raw_train) > 50:
                        feat_df = to_feature_df(raw_train)
                        trained_model = train_lgbm(feat_df)
                        st.success(f"OK · {len(feat_df):,} samples")
                    else:
                        st.warning(t("ml.no_db", ui_lang))


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Preisanalyse & Wettbewerb
# ╚══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        f'<div class="section-header">🔮 {t("forecast.tab_title", ui_lang)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**{t('forecast.section_select', ui_lang)}**")

    # ── REQUIRED selection: Product + Retailer + Market + Currency ───────────
    @st.cache_data(ttl=600, show_spinner=False)
    def _p_stores(c, _v=CACHE_VERSION):
        return get_distinct_stores(c)

    @st.cache_data(ttl=120, show_spinner=False)
    def _hist(term, store, country, _v=CACHE_VERSION):
        return load_price_history(product_name=term or None, retailer=store, country_code=country)

    fc_col1, fc_col2 = st.columns([3, 2])
    with fc_col1:
        search_term = st.text_input(
            t("forecast.product", ui_lang) + " *",
            value=brand_filter or "",
            placeholder="Wudy, Coca-Cola, Cedevita…",
            key="fc_search_term",
        )
    with fc_col2:
        p_stores = _p_stores(sel_country)
        sel_price_store_idx = st.selectbox(
            t("forecast.retailer", ui_lang) + " *",
            range(len(p_stores) + 1),
            format_func=lambda i: "— select —" if i == 0 else p_stores[i - 1],
            key="fc_retailer_pick",
        )
        price_store_val = None if sel_price_store_idx == 0 else p_stores[sel_price_store_idx - 1]

    fc_col3, fc_col4, fc_col5 = st.columns([2, 1, 1])
    with fc_col3:
        fc_market = sel_country
        st.text_input(
            t("forecast.market", ui_lang) + " *",
            value=(t(f"country.{fc_market}", ui_lang) if fc_market else "— select market in sidebar —"),
            disabled=True, key="fc_market_disp",
        )
    with fc_col4:
        st.text_input(
            t("forecast.currency", ui_lang),
            value=market_currency,
            disabled=True, key="fc_currency_disp",
        )
    with fc_col5:
        forecast_periods = st.slider(
            t("forecast.horizon", ui_lang), 7, 90, 30, key="prophet_periods",
        )

    selection_status = validate_forecast_selection(
        product=search_term, retailer=price_store_val,
        market=fc_market, currency=market_currency,
    )

    run_forecast = st.button(t("forecast.run", ui_lang), type="primary", key="run_fc_btn")

    if selection_status != "ok":
        st.warning(t("forecast.missing_selection", ui_lang))
        st.stop()

    # ── Load history strictly for the chosen product/retailer/market ─────────
    with st.spinner(t("general.loading", ui_lang)):
        hist_raw = _hist(search_term, price_store_val, fc_market)
        hist_df, hist_audit = build_filtered_view(
            hist_raw,
            country=fc_market,
            brand_query=search_term,
        )

    _hist_price_col = "price_eur" if not hist_df.empty and "price_eur" in hist_df.columns else "price"
    _p_sym = "€" if market_currency == "EUR" else market_currency

    # Plot historical prices
    if not hist_df.empty and "recorded_at" in hist_df.columns and _hist_price_col in hist_df.columns:
        st.markdown(
            f'<div class="section-header">📈 {t("forecast.historical_price", ui_lang)} – {search_term}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{len(hist_df):,} · {price_store_val} · {fc_market} · {market_currency}")
        fig_hist = px.area(
            hist_df.sort_values("recorded_at"),
            x="recorded_at", y=_hist_price_col,
            markers=True,
            color_discrete_sequence=[PRIMARY],
            labels={
                "recorded_at": t("forecast.axis_date", ui_lang),
                _hist_price_col: t("forecast.axis_price", ui_lang, currency=market_currency),
            },
        )
        fig_hist.update_layout(
            hovermode="x unified", margin=dict(t=20, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
            yaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info(t("forecast.too_few_points", ui_lang, n=len(hist_df)))

    # ── Eligibility gate before Prophet ──────────────────────────────────────
    if not run_forecast:
        st.info("👆 " + t("forecast.run", ui_lang))
        st.stop()

    obs_count = int(len(hist_df))
    history_days = 0
    if obs_count > 0 and "recorded_at" in hist_df.columns:
        _ts = pd.to_datetime(hist_df["recorded_at"], errors="coerce").dropna()
        if len(_ts) > 1:
            history_days = int((_ts.max() - _ts.min()).days)

    eligible = (obs_count >= MIN_OBSERVATIONS and history_days >= MIN_HISTORY_DAYS)

    if production_mode and not eligible:
        st.error(
            t("forecast.too_few_points", ui_lang, n=obs_count)
            + f"  ·  history: {history_days} days (need ≥ {MIN_HISTORY_DAYS})"
        )
        st.stop()

    # ── Build the prophet forecast ───────────────────────────────────────────
    with st.spinner(t("general.loading", ui_lang)):
        prophet_fig, forecast_df = forecast_price_trend(
            df=hist_df,
            product_id=search_term,
            periods=forecast_periods,
            allow_mock=not production_mode,
            currency=market_currency,
            lang=ui_lang,
            title_prefix=f"{search_term} · {price_store_val}",
        )

    st.plotly_chart(prophet_fig, use_container_width=True)

    if forecast_df.empty:
        st.warning(t("forecast.prophet_failed", ui_lang))
        st.stop()

    # ── KPI math: clean, no negative drops ───────────────────────────────────
    fc_p = forecast_df.reset_index(drop=True)
    _hist_sorted = hist_df.sort_values("recorded_at") if "recorded_at" in hist_df.columns else hist_df
    _last_series = _hist_sorted[_hist_price_col].dropna() if _hist_price_col in _hist_sorted.columns else pd.Series(dtype=float)
    last_hist_price = float(_last_series.iloc[-1]) if len(_last_series) > 0 else float(fc_p["yhat"].iloc[0])

    end_price = float(fc_p["yhat"].iloc[-1])
    min_price = float(fc_p["yhat"].min())
    max_price = float(fc_p["yhat"].max())
    min_pos = int(fc_p["yhat"].idxmin())
    max_pos = int(fc_p["yhat"].idxmax())

    if last_hist_price > 0:
        price_change_pct = (end_price - last_hist_price) / last_hist_price * 100
        max_drop_pct     = max(0.0, (last_hist_price - min_price) / last_hist_price * 100)
        max_rise_pct     = max(0.0, (max_price - last_hist_price) / last_hist_price * 100)
    else:
        price_change_pct = max_drop_pct = max_rise_pct = 0.0

    try:
        min_date = fc_p["ds"].iloc[min_pos].strftime("%d.%m.%Y")
        max_date = fc_p["ds"].iloc[max_pos].strftime("%d.%m.%Y")
    except (IndexError, AttributeError):
        min_date = "—"
        max_date = "—"

    drop_delta = (
        t("forecast.no_drop", ui_lang) if max_drop_pct == 0
        else f"−{max_drop_pct:.1f} % vs. today"
    )
    rise_delta = (
        t("forecast.no_rise", ui_lang) if max_rise_pct == 0
        else f"+{max_rise_pct:.1f} % vs. today"
    )

    fi1, fi2, fi3, fi4 = st.columns(4)
    _kpi_cards = [
        (fi1, t("forecast.kpi.future_price", ui_lang, days=forecast_periods),
         f"{end_price:.2f} {_p_sym}",
         f"{'▼' if price_change_pct < 0 else '▲'} {abs(price_change_pct):.1f} % vs. today",
         "kpi-card"),
        (fi2, t("forecast.kpi.lowest", ui_lang),
         f"{min_price:.2f} {_p_sym}", f"{min_date}", "kpi-card-green"),
        (fi3, t("forecast.kpi.highest", ui_lang),
         f"{max_price:.2f} {_p_sym}", f"{max_date}", "kpi-card-red"),
        (fi4, t("forecast.kpi.max_drop", ui_lang),
         f"{max_drop_pct:.1f} %", drop_delta, "kpi-card-amber"),
    ]
    for col, label, val, delta, css in _kpi_cards:
        col.markdown(
            f"""<div class="kpi-card {css}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-delta">{delta}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Forecast audit block ────────────────────────────────────────────────
    work_for_mape = pd.DataFrame({
        "ds": pd.to_datetime(hist_df["recorded_at"], errors="coerce"),
        "y": pd.to_numeric(hist_df[_hist_price_col], errors="coerce"),
    }).dropna().sort_values("ds")
    mape, mae, bias = compute_backtest_mape(work_for_mape)
    trust_key = compute_trust_level(mape, obs_count, history_days)

    trust_label = {
        "belastbar":       "✅ " + t("trust.belastbar", ui_lang),
        "eingeschr":       "⚠️ " + t("trust.eingeschr", ui_lang),
        "nicht_belastbar": "❌ " + t("trust.nicht_belastbar", ui_lang),
    }[trust_key]

    with st.expander(t("forecast.audit_title", ui_lang), expanded=True):
        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown(f"**{t('forecast.audit.status', ui_lang)}:** "
                        + ("ok" if eligible else t("status.insufficient_history", ui_lang)))
            st.markdown(f"**{t('forecast.audit.trust', ui_lang)}:** {trust_label}")
            st.markdown(f"**{t('forecast.audit.source', ui_lang)}:** "
                        + (t('build.supabase', ui_lang) if production_mode else t('build.mock', ui_lang)))
        with a2:
            st.markdown(f"**{t('forecast.audit.observations', ui_lang)}:** {obs_count}")
            st.markdown(f"**{t('forecast.audit.history_days', ui_lang)}:** {history_days}")
            st.markdown(f"**{t('forecast.audit.outliers', ui_lang)}:** —")
        with a3:
            mape_disp = f"{mape*100:.1f} %" if not pd.isna(mape) else "—"
            st.markdown(f"**{t('forecast.audit.mape', ui_lang)}:** {mape_disp}")
            st.markdown(f"**{t('forecast.currency', ui_lang)}:** {market_currency}")
            st.markdown(f"**{t('forecast.retailer', ui_lang)}:** {price_store_val}")

    # ── Historical evidence table ────────────────────────────────────────────
    with st.expander("📋 " + t("detail.history_5", ui_lang).replace(":", "")):
        ev = hist_df.copy()
        if "recorded_at" in ev.columns:
            ev["recorded_at"] = pd.to_datetime(ev["recorded_at"], errors="coerce").dt.strftime("%d.%m.%Y")
        ev = ev[[c for c in ["recorded_at", "product_name", "retailer", _hist_price_col]
                 if c in ev.columns]].rename(columns={
            "recorded_at": "Date",
            "product_name": t("detail.product", ui_lang),
            "retailer": t("detail.retailer", ui_lang),
            _hist_price_col: f"{t('forecast.price_label', ui_lang)} ({_p_sym})",
        })
        st.dataframe(ev.head(50), use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Datenqualität & Pipeline
# ╚══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">🗄️ Datenbank-Übersicht</div>', unsafe_allow_html=True)

    db_c1, db_c2 = st.columns([1, 2])
    with db_c1:
        status_color = "#059669" if db_connected else "#D97706"
        status_text  = "Verbunden" if db_connected else "Demo-Modus"
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Datenbankstatus</div>
              <div class="kpi-value" style="font-size:1.2rem; color:{status_color};">● {status_text}</div>
              <div class="kpi-delta">MarketinoDATABASE · Supabase EU-Central-2</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Abgedeckte Märkte</div>
              <div class="kpi-value" style="font-size:1.1rem;">HR · SI · BA · RS · MK · ME</div>
              <div class="kpi-delta">Kroatien · Slowenien · Bosnien · Serbien · Nordmazedonien · Montenegro</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with db_c2:
        @st.cache_data(ttl=300, show_spinner=False)
        def _retailer_dist(c): return get_retailer_distribution(c)
        dist_df = _retailer_dist(sel_country)

        if not dist_df.empty:
            top10 = dist_df.head(10)
            fig_pie = px.pie(
                top10, names="store_name", values="cnt",
                title="Top-10 Händler (Produktanzahl)",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.45,
            )
            fig_pie.update_layout(
                margin=dict(t=40, b=10, l=10, r=10),
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Produktdaten & Filter ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Produktdaten</div>', unsafe_allow_html=True)

    qc1, qc2, qc3, qc4 = st.columns([2, 2, 2, 1])
    with qc1:
        @st.cache_data(ttl=600, show_spinner=False)
        def _stores_q(c): return ["Alle Händler"] + get_distinct_stores(c)
        q_stores = _stores_q(sel_country)
        q_store_idx = st.selectbox("Händler", range(len(q_stores)), format_func=lambda i: q_stores[i], key="q_store")
        q_store = None if q_store_idx == 0 else q_stores[q_store_idx]
    with qc2:
        q_n = st.select_slider("Datensätze", [50, 100, 250, 500], value=100)
    with qc3:
        st.markdown("<br>", unsafe_allow_html=True)
        q_load = st.button("🔄 Laden", type="primary", use_container_width=True, key="q_load")

    @st.cache_data(ttl=120, show_spinner=False)
    def _q_load(country, store, cat, n):
        return load_products(country_code=country, store_name=store, category_l1=cat, limit=n)

    with st.spinner("Lade & bereinige Produktdaten…"):
        q_raw = _q_load(sel_country, q_store, sel_cat, q_n)
        q_df, q_report = _apply_quality(q_raw)

    # ── Quality Report ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🛡️ Datenqualitäts-Report (letzter Datenabruf)</div>', unsafe_allow_html=True)
    rq1, rq2, rq3, rq4, rq5 = st.columns(5)
    rq1.metric("Datensätze gesamt",           str(q_report["n_total"]))
    rq2.metric("✅ Marken korrigiert",         str(q_report["n_brand_fixed"]),
               help="Markenname wurde per Produktname-Keyword überschrieben")
    rq3.metric("✅ Kategorien normiert",       str(q_report["n_cat_fixed"]),
               help="Kategorie wurde per Rule-based Engine neu gesetzt")
    rq4.metric("🔄 Preise getauscht",          str(q_report["n_price_swapped"]),
               help="Promo > Regulär → Werte wurden getauscht (Extraktionsfehler)")
    rq5.metric("🚫 Fehlerhafte Datensätze",    str(q_report["n_excluded"]),
               delta=f"-{q_report['n_excluded']} ausgeschlossen" if q_report["n_excluded"] > 0 else None,
               delta_color="inverse",
               help="Preis-Anomalie: nach Tausch immer noch logisch falsch → aus Anzeige entfernt")

    st.caption(
        f"**Datenqualitätsrate: "
        f"{(q_report['n_clean'] / max(q_report['n_total'], 1) * 100):.1f} %** "
        f"({q_report['n_clean']} von {q_report['n_total']} Datensätzen bestehen Qualitätscheck)"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if not q_df.empty:
        q_show = q_df[[c for c in [
            "store_name", "name", "brand", "price_eur", "original_price_eur",
            "discount_pct", "currency", "category_de", "country_code",
            "valid_from", "valid_until", "discount_label",
        ] if c in q_df.columns]].copy()
        if "category_de" not in q_show.columns and "category_l1" in q_df.columns:
            q_show["category_de"] = q_df["category_l1"].apply(cat_local)
        q_show = q_show.rename(columns={
            "store_name": "Händler", "name": "Produkt", "brand": "Marke (bereinigt)",
            "price_eur": "Preis (€)", "original_price_eur": "Orig.-Preis (€)",
            "discount_pct": "Rabatt %", "currency": "Währung",
            "category_de": "Kategorie DE", "country_code": "Markt",
            "valid_from": "Von", "valid_until": "Bis", "discount_label": "Rabatt-Label",
        })
        q_cfg: dict = {}
        if "Preis (€)" in q_show.columns:
            q_cfg["Preis (€)"] = st.column_config.NumberColumn("Preis (€)", format="%.2f €")
        if "Orig.-Preis (€)" in q_show.columns:
            q_cfg["Orig.-Preis (€)"] = st.column_config.NumberColumn("Orig.-Preis (€)", format="%.2f €")
        if "Rabatt %" in q_show.columns:
            q_cfg["Rabatt %"] = st.column_config.NumberColumn("Rabatt %", format="%.1f %%")

        st.dataframe(q_show, column_config=q_cfg, use_container_width=True, hide_index=True, height=350)

    st.divider()

    # ── Kategorie-Verteilung ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Kategorie-Verteilung</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=300, show_spinner=False)
    def _cat_dist(c): return get_category_distribution(c)
    cat_df = _cat_dist(sel_country)

    if not cat_df.empty:
        cat_df["__cat"] = cat_df["category_l1"].apply(lambda x: cat_local(x, ui_lang))
        cat_df["label"] = cat_df.apply(
            lambda r: f"{r['__cat']} ({r['category_l1']})" if r["__cat"] != r["category_l1"] else r["category_l1"],
            axis=1,
        )
        fig_cat = px.bar(
            cat_df.head(15),
            x="cnt", y="label", orientation="h",
            color="cnt", color_continuous_scale="Blues",
            labels={"cnt": "Anzahl Produkte", "label": "Kategorie"},
        )
        fig_cat.update_layout(
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    st.divider()

    # ── Entity Matching ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔗 Entity-Matching – Produktnormierung</div>', unsafe_allow_html=True)
    st.caption(
        "KI-Sprachmodell (Sentence-Transformer) ordnet rohe Händler-Produktbezeichnungen "
        f"einer normierten Master-Produktliste zu. Mindest-Score: {match_threshold:.0%}"
    )

    MASTER_LIST = [
        "Milka Schokolade 300g",
        "Coca-Cola 1,5L PET",
        "Ritter Sport Voll-Nuss 100g",
        "Haribo Goldbären 200g",
        "Nutella Nuss-Nougat-Creme 450g",
        "Bio Vollmilch 3,5% 1L",
        "Ariel Waschmittel 20 WL",
        "Pampers Baby-Dry Gr.3 44 Stk",
        "Red Bull Energy Drink 250ml",
        "Pringles Original 185g",
    ]

    match_sample = q_df.head(20) if not q_df.empty else _q_load(sel_country, None, None, 20).head(20)
    matcher = ProductMatcher(threshold=match_threshold)
    raw_cats = match_sample["category_l1"].fillna("").tolist() if "category_l1" in match_sample.columns else None

    with st.spinner("Entity Resolution…"):
        results = matcher.batch_match(match_sample["name"].tolist(), MASTER_LIST, raw_categories=raw_cats)

    _show_debug = st.checkbox(
        "🔍 Nicht-zugeordnete Produkte anzeigen (Debug)",
        value=False,
        help="Zeigt auch Produkte mit Score < Schwellenwert oder Kategorie-Konflikt",
    )

    match_df = match_sample[[c for c in ["store_name", "name", "price", "country_code"] if c in match_sample.columns]].copy()
    match_df["master_produkt"] = [r.master_product for r in results]
    match_df["score"]          = [round(r.score, 4) for r in results]
    match_df["methode"]        = [r.match_method for r in results]
    match_df["status"]         = [r.match_status for r in results]
    match_df["OK"]             = ["✅" if r.is_confident else "⚠️" for r in results]

    # Dashboard-Guard: zeige standardmäßig nur konfidente Matches
    if not _show_debug:
        match_df = match_df[match_df["status"].isin(["keyword_exact", "embedding_high_confidence"])]

    if match_df.empty:
        st.info(
            "Keine zugeordneten Produkte im Sample. "
            "Aktiviere 'Nicht-zugeordnete Produkte anzeigen' für Details."
        )
    else:
        match_cfg = {
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.2f"),
        }
        st.dataframe(
            match_df.rename(columns={
                "store_name": "Händler",
                "name": "Roh-Bezeichnung",
                "price": "Preis",
                "country_code": "Markt",
                "master_produkt": "→ Master-Produkt",
                "methode": "Methode",
                "status": "Status",
                "OK": "OK",
            }),
            column_config=match_cfg,
            use_container_width=True, hide_index=True,
        )

    conf_rate = sum(1 for r in results if r.is_confident) / max(len(results), 1) * 100
    unmatched_count = sum(1 for r in results if not r.is_confident)
    m1, m2, m3 = st.columns(3)
    m1.metric("Match-Rate (Sample)", f"{conf_rate:.1f} %")
    m2.metric("Konfidenz-Schwelle", f"{match_threshold:.0%}")
    m3.metric("Nicht zugeordnet", str(unmatched_count), help="Score < Schwellenwert oder Kategorie-Konflikt")

    st.divider()

    # ── Aktionskalender ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📆 Aktionskalender</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=180, show_spinner=False)
    def _cal_data(country, store):
        return load_products(country_code=country, store_name=store, limit=500)

    cal_df = _cal_data(sel_country, q_store)

    if cal_df.empty:
        st.info("Keine Kalenderdaten verfügbar.")
    elif "valid_from" not in cal_df.columns or "valid_until" not in cal_df.columns:
        st.info("Aktionskalender benötigt Datums-Spalten (valid_from / valid_until).")
    else:
        cal_clean = cal_df.dropna(subset=["valid_from", "valid_until"])
        if cal_clean.empty:
            st.info("Keine Aktionen mit vollständigen Datumsinformationen.")
        elif not q_store and sel_country is None:
            st.info(
                "Bitte wähle einen **Händler** oder einen **Markt** um den Kalender anzuzeigen. "
                "Ohne Filter würde der Kalender zu viele Zeilen enthalten."
            )
        else:
            n_cal = len(cal_clean)
            if n_cal > MAX_CALENDAR_ROWS:
                st.warning(
                    f"Kalender enthält {n_cal:,} Einträge – zeige nur die ersten {MAX_CALENDAR_ROWS}. "
                    "Verwende den Händler-Filter für eine präzisere Ansicht."
                )
                cal_clean = cal_clean.sort_values("valid_from").head(MAX_CALENDAR_ROWS).copy()
            else:
                cal_clean = cal_clean.sort_values("valid_from").copy()

            if "category_l1" in cal_clean.columns:
                cal_clean["Kategorie"] = cal_clean["category_l1"].apply(lambda x: cat_local(x, ui_lang))
            if "store_name" in cal_clean.columns:
                fig_gantt = px.timeline(
                    cal_clean,
                    x_start="valid_from",
                    x_end="valid_until",
                    y="store_name",
                    color="Kategorie" if "Kategorie" in cal_clean.columns else "store_name",
                    hover_name="name" if "name" in cal_clean.columns else None,
                    labels={"store_name": "Händler"},
                )
                fig_gantt.update_layout(margin=dict(t=20, b=10))
                st.plotly_chart(fig_gantt, use_container_width=True)
            else:
                st.info(t("upcoming.none", ui_lang))


# ╔══════════════════════════════════════════════════════════════════════════════
#  Global Audit / Debug expander (footer)
# ╚══════════════════════════════════════════════════════════════════════════════

with st.expander("🔬 " + t("section.audit", ui_lang)):
    st.markdown(f"""
| Field | Value |
|---|---|
| `app_version`           | `{APP_VERSION}` |
| `cache_version`         | `{CACHE_VERSION}` |
| `app_mode`              | `{'production' if production_mode else 'demo'}` |
| `data_source`           | `{'supabase' if production_mode else 'mock'}` |
| `language`              | `{ui_lang}` |
| `market`                | `{sel_country or '__all__'}` |
| `market_currency`       | `{market_currency}` |
| `brand_query`           | `{brand_filter or '(empty)'}` |
| `category`              | `{sel_cat or '(all)'}` |
| `raw_rows (history)`    | `{_hist_audit.get('raw_rows', 0) if '_hist_audit' in dir() else 'n/a'}` |
| `after_market`          | `{_hist_audit.get('after_market', 0) if '_hist_audit' in dir() else 'n/a'}` |
| `after_category`        | `{_hist_audit.get('after_category', 0) if '_hist_audit' in dir() else 'n/a'}` |
| `after_brand`           | `{_hist_audit.get('after_brand', 0) if '_hist_audit' in dir() else 'n/a'}` |
| `forecast_rows`         | `{len(fc_df_all) if 'fc_df_all' in dir() else 'n/a'}` |
| `upcoming_raw_rows`     | `{_upcoming_audit.get('raw_rows', 0) if '_upcoming_audit' in dir() else 'n/a'}` |
| `upcoming_final_rows`   | `{_upcoming_audit.get('final', 0) if '_upcoming_audit' in dir() else 'n/a'}` |
""")
