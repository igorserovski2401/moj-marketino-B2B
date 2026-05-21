"""moj-marketino – B2B Promo Intelligence Platform.
Commercial intelligence for Key Account Managers in FMCG – Balkan markets.
Data source: Supabase MarketinoDATABASE (997k+ products, 6 countries).
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
        ).decode().strip() or "7tab-rebuild"
    except Exception:
        APP_VERSION = "7tab-rebuild"

CACHE_VERSION: str = APP_VERSION

# ── Country/category helpers ──────────────────────────────────────────────────

COUNTRY_KEYS: list[str] = ["HR", "SI", "BA", "RS", "MK", "ME"]

MAX_CALENDAR_ROWS = 500
MIN_TREEMAP_ROWS = 5


def _country_label(code: str, lang: str) -> str:
    return t(f"country.{code}", lang)


def cat_local(name: str | None, lang: str = "EN") -> str:
    if not name:
        return "—"
    return translate_category(name, lang)


def _apply_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run quality pipeline, return cleaned data + report dict."""
    if df.empty:
        return df, {"n_total": 0, "n_brand_fixed": 0, "n_cat_fixed": 0,
                    "n_price_swapped": 0, "n_excluded": 0, "n_clean": 0}
    clean, report = run_quality_pipeline(df)
    return clean, report._asdict()


# ── Central filter pipeline ───────────────────────────────────────────────────

SEARCHABLE_FIELDS: list[str] = [
    "brand", "brand_clean", "manufacturer", "manufacturer_clean",
    "product", "product_name", "name", "raw_product_name",
    "normalized_product_name", "master_product", "description",
]


def normalize_search_text(value) -> str:
    """Lowercase + NFKD Unicode + whitespace-collapse for fuzzy match."""
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
    """Single source of truth — every visible widget MUST use this pipeline."""
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
    """Returns 'ok' or a missing_* status. Gate before running Prophet."""
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
    """Return trust level key: 'belastbar' / 'eingeschr' / 'nicht_belastbar'."""
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


def _dq_badge(row: pd.Series) -> str:
    """Return a DQ status badge string for a single product row."""
    issues = []
    brand = str(row.get("brand") or row.get("brand_clean") or "")
    if not brand or brand.lower() in ("", "unbekannt", "unknown", "none"):
        issues.append("no_brand")
    orig = row.get("original_price_eur") or row.get("original_price")
    if pd.isna(orig) or orig is None:
        issues.append("no_regular")
    disc = row.get("discount_pct")
    if pd.isna(disc) or disc is None or float(disc) <= 0:
        issues.append("no_discount")
    cat = str(row.get("category_l1") or "")
    if not cat or cat.lower() in ("", "other", "unbekannt"):
        issues.append("cat_uncertain")
    score = row.get("match_score")
    if score is not None and not pd.isna(score) and float(score) < MIN_MATCH_SCORE:
        issues.append("low_match")
    return "complete" if not issues else issues[0]


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
      .alert-high  {{ background:#FEF2F2; border-left: 4px solid {ACCENT}; border-radius:6px; padding:.6rem 1rem; margin:.3rem 0; }}
      .alert-med   {{ background:#FFFBEB; border-left: 4px solid {AMBER}; border-radius:6px; padding:.6rem 1rem; margin:.3rem 0; }}
      .alert-low   {{ background:#F0FDF4; border-left: 4px solid {GREEN}; border-radius:6px; padding:.6rem 1rem; margin:.3rem 0; }}
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
          <span style='font-size:.7rem; color:#9CA3AF;'>v3.0 · Balkan Edition</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _country_keys_extended = ["__all__"] + COUNTRY_KEYS

    if "sel_country" not in st.session_state:
        st.session_state["sel_country"] = "__all__"

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
            translate_category(sidebar_cats[i], ui_lang) if i > 0 else sidebar_cats[0]
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
        kw_vorschau = st.slider(t("sidebar.preview_weeks", ui_lang) if t("sidebar.preview_weeks", ui_lang) != "sidebar.preview_weeks" else "Preview weeks", 1, 8, 4)

    st.divider()
    st.caption(f"© 2026 moj-marketino · Build {APP_VERSION}")

# ── Header ────────────────────────────────────────────────────────────────────

market_label = t(f"country.{sel_country}", ui_lang) if sel_country else t("sidebar.all_markets", ui_lang)

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

# ── Global KPI row ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _cached_stats(country: str | None, _v=CACHE_VERSION) -> dict:
    return get_overview_stats(country)

stats = _cached_stats(sel_country)

k1, k2, k3, k4, k5 = st.columns(5)
_kpis = [
    (k1, t("kpi.observed_products", ui_lang),   f"{stats['total_products']:,}".replace(",", "."), "MarketinoDB"),
    (k2, t("kpi.active_retailers", ui_lang),    str(stats["total_retailers"]),                    "Balkan"),
    (k3, t("kpi.catalogs", ui_lang),            str(stats["total_catalogs"]),                     ""),
    (k4, t("kpi.markets", ui_lang),             str(stats["total_countries"]),                    "HR · SI · BA · RS · MK · ME"),
    (k5, t("kpi.active_promos_week", ui_lang),  str(stats.get("active_promos", 0)),               ""),
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

# ── Load shared data ──────────────────────────────────────────────────────────
# Shared forecast history used by Cockpit, Alerts, and Experimental tab.

@st.cache_data(ttl=300, show_spinner=False)
def _load_forecast_history(country, cat, _v=CACHE_VERSION):
    return load_promo_history_for_forecast(
        country_code=country, category_l1=cat, days_back=365, limit=5000,
    )

with st.spinner(t("general.loading", ui_lang)):
    _history_raw = _load_forecast_history(sel_country, sel_cat)
    history_df, _hist_audit = build_filtered_view(
        _history_raw,
        country=sel_country,
        category=sel_cat,
        brand_query=brand_filter,
    )
    _forecasts_all = build_forecasts_from_promo_history(history_df, lang=ui_lang)
    fc_df_all = forecasts_to_dataframe(_forecasts_all)

# ── 7 Tabs ────────────────────────────────────────────────────────────────────

_tab_labels = [
    t("tab.cockpit", ui_lang),
    t("tab.monitor", ui_lang),
    t("tab.history", ui_lang),
    t("tab.benchmark", ui_lang),
    t("tab.alerts", ui_lang),
    t("tab.quality", ui_lang),
    t("tab.forecast", ui_lang),
]
tab_cockpit, tab_monitor, tab_history, tab_bench, tab_alerts, tab_quality, tab_exp = st.tabs(_tab_labels)

# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 1 – KAM Cockpit
# ╚══════════════════════════════════════════════════════════════════════════════

with tab_cockpit:
    st.markdown(
        f'<div class="section-header">🎯 {t("cockpit.action_briefing", ui_lang)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(t("cockpit.subtitle", ui_lang))

    # ── KAM KPIs ─────────────────────────────────────────────────────────────
    _n_overdue = 0
    _n_near_cycle = 0
    _avg_disc = float("nan")
    if not fc_df_all.empty:
        for _fc in _forecasts_all:
            if _fc.days_since_last_promo and _fc.avg_cycle_days:
                if _fc.days_since_last_promo > _fc.avg_cycle_days * 1.2:
                    _n_overdue += 1
                elif _fc.days_since_last_promo > _fc.avg_cycle_days * 0.8:
                    _n_near_cycle += 1
        _disc_vals = [f.typical_discount_pct_max for f in _forecasts_all if f.typical_discount_pct_max]
        if _disc_vals:
            _avg_disc = sum(_disc_vals) / len(_disc_vals)

    ck1, ck2, ck3, ck4 = st.columns(4)
    ck1.metric(t("cockpit.overdue", ui_lang),   str(_n_overdue),          delta=None)
    ck2.metric(t("cockpit.near_cycle", ui_lang), str(_n_near_cycle),      delta=None)
    ck3.metric(t("kpi.forecast_base", ui_lang),  str(len(fc_df_all)),     delta=None)
    ck4.metric(t("kpi.avg_discount", ui_lang),
               f"{_avg_disc:.0f} %" if pd.notna(_avg_disc) else "—",
               delta=None)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Build action briefing table ───────────────────────────────────────────
    if not _forecasts_all:
        st.info(t("cockpit.no_briefings", ui_lang))
    else:
        _PRIORITY_MAP = {
            "Hoch":   (t("alert.sev_high", ui_lang),   "#FEF2F2"),
            "Mittel": (t("alert.sev_medium", ui_lang),  "#FFFBEB"),
            "Niedrig": (t("alert.sev_low", ui_lang),    "#F0FDF4"),
        }

        _rows = []
        for _fc in sorted(
            _forecasts_all,
            key=lambda f: (
                0 if (f.days_since_last_promo and f.avg_cycle_days
                      and f.days_since_last_promo > f.avg_cycle_days * 1.2) else
                1 if (f.days_since_last_promo and f.avg_cycle_days
                      and f.days_since_last_promo > f.avg_cycle_days * 0.8) else 2,
                -(f.probability or 0),
            ),
        ):
            _prio_label, _ = _PRIORITY_MAP.get(_fc.priority, (_fc.priority, "#FFFFFF"))
            _cycle = (
                f"{_fc.median_cycle_days} {t('general.days', ui_lang)}"
                if _fc.median_cycle_days
                else (f"{_fc.avg_cycle_days} {t('general.days', ui_lang)}" if _fc.avg_cycle_days else "—")
            )
            _since = (
                t("general.days_ago", ui_lang, d=_fc.days_since_last_promo)
                if _fc.days_since_last_promo is not None else "—"
            )
            _rec = (
                t("cockpit.recommend_prepare", ui_lang)
                if _fc.days_since_last_promo and _fc.avg_cycle_days
                   and _fc.days_since_last_promo > _fc.avg_cycle_days * 1.2
                else t("cockpit.recommend_monitor", ui_lang)
            )
            _rows.append({
                t("cockpit.col_priority", ui_lang): _prio_label,
                t("cockpit.col_product", ui_lang):  _fc.product[:40] if _fc.product else "—",
                t("cockpit.col_brand", ui_lang):    _fc.brand or "—",
                t("cockpit.col_retailer", ui_lang): _fc.retailer or "—",
                t("cockpit.col_days_since", ui_lang): _since,
                t("cockpit.col_cycle", ui_lang):    _cycle,
                t("cockpit.col_price", ui_lang):    (
                    format_price(_fc.last_promo_price, market_currency)
                    if _fc.last_promo_price else "—"
                ),
                t("cockpit.col_trust", ui_lang):    f"{_fc.probability:.0%}",
                t("cockpit.recommendation", ui_lang): _rec,
            })

        briefing_df = pd.DataFrame(_rows)
        if not briefing_df.empty:
            # Colour-code priority column
            prio_col = t("cockpit.col_priority", ui_lang)
            st.dataframe(
                briefing_df,
                use_container_width=True,
                hide_index=True,
                height=min(35 * len(briefing_df) + 45, 600),
            )

    # ── Product filter inside cockpit ─────────────────────────────────────────
    with st.expander("🔍 " + t("sidebar.filter", ui_lang)):
        _ck_q = st.text_input(t("filter.search_product", ui_lang), key="ck_product_q")
        if _ck_q and not briefing_df.empty:
            _prod_col = t("cockpit.col_product", ui_lang)
            if _prod_col in briefing_df.columns:
                briefing_df = briefing_df[
                    briefing_df[_prod_col].str.contains(_ck_q, case=False, na=False)
                ]


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Promo Monitor
# ╚══════════════════════════════════════════════════════════════════════════════

with tab_monitor:
    st.markdown(
        f'<div class="section-header">📋 {t("monitor.title", ui_lang)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(t("monitor.subtitle", ui_lang))

    @st.cache_data(ttl=180, show_spinner=False)
    def _load_upcoming(country, _v=CACHE_VERSION):
        return load_upcoming_promos(country_code=country, days_ahead=21, limit=300)

    with st.spinner(t("general.loading", ui_lang)):
        _upcoming_raw = _load_upcoming(sel_country)
        _upcoming_df, _ = build_filtered_view(
            _upcoming_raw,
            country=sel_country,
            category=sel_cat,
            brand_query=brand_filter,
        )

    # ── Filter row ────────────────────────────────────────────────────────────
    _mf1, _mf2, _mf3 = st.columns(3)
    with _mf1:
        _mon_real_disc = st.checkbox(t("monitor.filter_real_discount", ui_lang), value=False, key="mon_real_disc")
    with _mf2:
        _mon_with_brand = st.checkbox(t("monitor.filter_with_brand", ui_lang), value=False, key="mon_brand")
    with _mf3:
        _mon_reliable = st.checkbox(t("monitor.filter_reliable", ui_lang), value=False, key="mon_reliable")

    _mon_df = _upcoming_df.copy()

    if not _mon_df.empty:
        # Apply DQ badges
        _mon_df["__dq"] = _mon_df.apply(_dq_badge, axis=1)

        if _mon_real_disc:
            _disc_col = "discount_pct"
            if _disc_col in _mon_df.columns:
                _mon_df = _mon_df[_mon_df[_disc_col].fillna(0) > 0]

        if _mon_with_brand:
            _brand_col = "brand_clean" if "brand_clean" in _mon_df.columns else "brand"
            if _brand_col in _mon_df.columns:
                _mon_df = _mon_df[_mon_df[_brand_col].notna() & (_mon_df[_brand_col] != "")]

        if _mon_reliable and "match_score" in _mon_df.columns:
            _mon_df = _mon_df[_mon_df["match_score"].fillna(0) >= MIN_MATCH_SCORE]

    if _mon_df.empty:
        st.info(t("monitor.no_data", ui_lang))
    else:
        # ── DQ summary ───────────────────────────────────────────────────────
        _dq_counts = _mon_df["__dq"].value_counts() if "__dq" in _mon_df.columns else pd.Series(dtype=int)
        _n_complete = int(_dq_counts.get("complete", 0))
        _n_total_mon = len(_mon_df)
        _pct_complete = _n_complete / max(_n_total_mon, 1) * 100

        _dm1, _dm2, _dm3 = st.columns(3)
        _dm1.metric(t("dq.total_records", ui_lang), str(_n_total_mon))
        _dm2.metric(t("dq.complete_pct", ui_lang), f"{_pct_complete:.0f} %")
        _dm3.metric(t("dq.avg_match_score", ui_lang),
                    f"{_mon_df['match_score'].mean():.2f}" if "match_score" in _mon_df.columns else "—")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── DQ badge mapping ─────────────────────────────────────────────────
        _DQ_LABEL = {
            "complete":    t("monitor.dq_complete", ui_lang),
            "no_brand":    t("monitor.dq_no_brand", ui_lang),
            "no_regular":  t("monitor.dq_no_regular", ui_lang),
            "no_discount": t("monitor.dq_no_discount", ui_lang),
            "cat_uncertain": t("monitor.dq_cat_uncertain", ui_lang),
            "low_match":   t("monitor.dq_low_match", ui_lang),
        }

        # ── Display table ─────────────────────────────────────────────────────
        _price_col = "price_eur" if "price_eur" in _mon_df.columns else "price"
        _orig_col  = "original_price_eur" if "original_price_eur" in _mon_df.columns else "original_price"

        _show_cols_src = [
            ("name",            t("monitor.col_product", ui_lang)),
            ("brand_clean",     t("monitor.col_brand", ui_lang)),
            ("store_name",      t("monitor.col_retailer", ui_lang)),
            ("country_code",    t("monitor.col_market", ui_lang)),
            ("category_l1",     t("monitor.col_category", ui_lang)),
            ("valid_from",      t("monitor.col_valid_from", ui_lang)),
            ("valid_until",     t("monitor.col_valid_until", ui_lang)),
            ("discount_pct",    t("monitor.col_discount", ui_lang)),
            (_price_col,        t("monitor.col_promo_price", ui_lang)),
            (_orig_col,         t("monitor.col_regular_price", ui_lang)),
            ("currency",        t("monitor.col_currency", ui_lang)),
            ("__dq",            t("monitor.col_dq", ui_lang)),
        ]
        _avail_src = [(s, d) for s, d in _show_cols_src if s in _mon_df.columns]
        _mon_show = _mon_df[[s for s, _ in _avail_src]].copy()
        _mon_show.columns = [d for _, d in _avail_src]

        _dq_disp_col = t("monitor.col_dq", ui_lang)
        if _dq_disp_col in _mon_show.columns:
            _mon_show[_dq_disp_col] = _mon_show[_dq_disp_col].map(_DQ_LABEL).fillna("—")

        if _pct_complete < 100:
            _bad_pct = int(100 - _pct_complete)
            if "match_score" in _mon_df.columns:
                _low_match_pct = int((_mon_df["match_score"].fillna(0) < MIN_MATCH_SCORE).mean() * 100)
                if _low_match_pct > 10:
                    st.warning(t("dq.low_score_warning", ui_lang, pct=_low_match_pct))

        _cfg_mon: dict = {}
        _disc_disp_col = t("monitor.col_discount", ui_lang)
        _price_disp_col = t("monitor.col_promo_price", ui_lang)
        _orig_disp_col  = t("monitor.col_regular_price", ui_lang)
        if _disc_disp_col in _mon_show.columns:
            _cfg_mon[_disc_disp_col] = st.column_config.NumberColumn(_disc_disp_col, format="%.1f %%")
        if _price_disp_col in _mon_show.columns:
            _cfg_mon[_price_disp_col] = st.column_config.NumberColumn(_price_disp_col, format="%.2f")
        if _orig_disp_col in _mon_show.columns:
            _cfg_mon[_orig_disp_col] = st.column_config.NumberColumn(_orig_disp_col, format="%.2f")

        st.dataframe(_mon_show, use_container_width=True, hide_index=True,
                     column_config=_cfg_mon, height=400)

        # ── CSV export ────────────────────────────────────────────────────────
        _csv_bytes = _mon_show.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=t("monitor.export", ui_lang),
            data=_csv_bytes,
            file_name=f"promo_monitor_{sel_country or 'all'}.csv",
            mime="text/csv",
        )


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Product History
# ╚══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.markdown(
        f'<div class="section-header">📈 {t("history.title", ui_lang)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(t("history.select_prompt", ui_lang))

    # ── Product + Retailer selectors ──────────────────────────────────────────
    @st.cache_data(ttl=600, show_spinner=False)
    def _hist_stores(c, _v=CACHE_VERSION):
        return get_distinct_stores(c)

    _hist_available_stores = _hist_stores(sel_country)

    _hc1, _hc2 = st.columns(2)
    with _hc1:
        _hist_product_q = st.text_input(
            t("history.select_product", ui_lang),
            placeholder="Milka, Coca-Cola, Podravka…",
            key="hist_product_q",
        )
    with _hc2:
        _hist_store_opts = ["—"] + _hist_available_stores
        _hist_store_idx = st.selectbox(
            t("history.select_retailer", ui_lang),
            range(len(_hist_store_opts)),
            format_func=lambda i: _hist_store_opts[i],
            key="hist_store_pick",
        )
        _hist_retailer = None if _hist_store_idx == 0 else _hist_store_opts[_hist_store_idx]

    if not _hist_product_q and not _hist_retailer:
        st.info(t("history.select_prompt", ui_lang))
    else:
        # Filter history using the global pipeline
        _hist_view, _ = build_filtered_view(
            _history_raw,
            country=sel_country,
            category=sel_cat,
            retailer=_hist_retailer,
            brand_query=_hist_product_q,
        )

        if _hist_view.empty:
            st.info(t("history.no_data", ui_lang))
        else:
            # ── Stats row ─────────────────────────────────────────────────────
            _n_hist = len(_hist_view)
            _price_col_h = "price_eur" if "price_eur" in _hist_view.columns else "price"
            _disc_col_h  = "discount_pct"

            _hs1, _hs2, _hs3, _hs4 = st.columns(4)
            _hs1.metric(t("history.stat_total", ui_lang), str(_n_hist))

            _avg_d = _hist_view[_disc_col_h].dropna().mean() if _disc_col_h in _hist_view.columns else float("nan")
            _hs2.metric(t("history.stat_median_price", ui_lang),
                        format_price(_hist_view[_price_col_h].dropna().median(), market_currency)
                        if _price_col_h in _hist_view.columns else "—")
            _hs3.metric("Avg. discount", f"{_avg_d:.0f} %" if pd.notna(_avg_d) else "—")

            # Days since last promo
            _date_col_h = "valid_from"
            if _date_col_h in _hist_view.columns:
                _dates_h = pd.to_datetime(_hist_view[_date_col_h], errors="coerce").dropna()
                if len(_dates_h) > 0:
                    _days_since_h = (pd.Timestamp.now() - _dates_h.max()).days
                    _hs4.metric(t("history.stat_days_since", ui_lang), str(_days_since_h))

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Timeline chart ────────────────────────────────────────────────
            if _date_col_h in _hist_view.columns and _price_col_h in _hist_view.columns:
                _chart_df = _hist_view.dropna(subset=[_date_col_h, _price_col_h]).copy()
                _chart_df[_date_col_h] = pd.to_datetime(_chart_df[_date_col_h], errors="coerce")
                _chart_df = _chart_df.sort_values(_date_col_h)

                if len(_chart_df) > 0:
                    _color_col = "store_name" if "store_name" in _chart_df.columns else None
                    _fig_h = px.scatter(
                        _chart_df,
                        x=_date_col_h,
                        y=_price_col_h,
                        color=_color_col,
                        size=_disc_col_h if _disc_col_h in _chart_df.columns else None,
                        hover_name="name" if "name" in _chart_df.columns else None,
                        hover_data={_disc_col_h: ":.1f"} if _disc_col_h in _chart_df.columns else {},
                        title=t("history.chart_title", ui_lang)
                              if t("history.chart_title", ui_lang) != "history.chart_title"
                              else "Promo Price Timeline",
                        labels={
                            _date_col_h: t("history.col_period", ui_lang),
                            _price_col_h: f"{t('history.col_price', ui_lang)} ({market_currency})",
                        },
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    _fig_h.update_layout(
                        hovermode="x unified",
                        margin=dict(t=30, b=20),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        xaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
                        yaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
                    )
                    st.plotly_chart(_fig_h, use_container_width=True)

            # ── Detail table ──────────────────────────────────────────────────
            _hist_show_cols = [
                (_date_col_h,   t("history.col_period", ui_lang)),
                ("store_name",  t("monitor.col_retailer", ui_lang)),
                ("name",        t("monitor.col_product", ui_lang)),
                ("brand_clean", t("monitor.col_brand", ui_lang)),
                (_price_col_h,  t("history.col_price", ui_lang)),
                ("original_price_eur", t("monitor.col_regular_price", ui_lang)),
                (_disc_col_h,   t("history.col_discount", ui_lang)),
            ]
            _hist_avail = [(s, d) for s, d in _hist_show_cols if s in _hist_view.columns]
            _hist_table = _hist_view[[s for s, _ in _hist_avail]].sort_values(_date_col_h, ascending=False).head(200)
            _hist_table.columns = [d for _, d in _hist_avail]
            st.dataframe(_hist_table, use_container_width=True, hide_index=True, height=300)


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Competitive Benchmark
# ╚══════════════════════════════════════════════════════════════════════════════

with tab_bench:
    st.markdown(
        f'<div class="section-header">⚔️ {t("bench.title", ui_lang)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(t("bench.subtitle", ui_lang))

    _bench_df = history_df.copy()

    if _bench_df.empty:
        st.info(t("bench.no_data", ui_lang))
    else:
        # ── Category selector ─────────────────────────────────────────────────
        _bench_cats = ["—"]
        if "category_l1" in _bench_df.columns:
            _bench_cats += sorted(_bench_df["category_l1"].dropna().unique().tolist())
        _bench_cat_idx = st.selectbox(
            t("bench.select_category", ui_lang),
            range(len(_bench_cats)),
            format_func=lambda i: cat_local(_bench_cats[i], ui_lang) if i > 0 else "—",
            key="bench_cat_pick",
        )
        _sel_bench_cat = None if _bench_cat_idx == 0 else _bench_cats[_bench_cat_idx]

        if _sel_bench_cat and "category_l1" in _bench_df.columns:
            _bench_df = _bench_df[_bench_df["category_l1"] == _sel_bench_cat]

        _brand_col_b = "brand_clean" if "brand_clean" in _bench_df.columns else "brand"
        _store_col_b = "store_name"

        _has_brand = _brand_col_b in _bench_df.columns and not _bench_df[_brand_col_b].isna().all()
        _has_store = _store_col_b in _bench_df.columns

        if not _has_brand or not _has_store:
            st.info(t("bench.no_data", ui_lang))
        else:
            _bench_agg_cols = {
                t("bench.col_promo_count", ui_lang): ("name", "count"),
            }
            _disc_col_b = "discount_pct"
            _price_col_b = "price_eur" if "price_eur" in _bench_df.columns else "price"

            _gb_keys = [_brand_col_b, _store_col_b]
            if "country_code" in _bench_df.columns:
                _gb_keys.append("country_code")

            _agg_dict: dict = {"name": "count"}
            if _disc_col_b in _bench_df.columns:
                _agg_dict[_disc_col_b] = "mean"
            if _price_col_b in _bench_df.columns:
                _agg_dict[_price_col_b] = "median"

            _bench_grp = (
                _bench_df.groupby(_gb_keys, dropna=True)
                .agg(_agg_dict)
                .reset_index()
            )
            _bench_grp.columns = (
                [_brand_col_b, _store_col_b]
                + (["country_code"] if "country_code" in _gb_keys else [])
                + [t("bench.col_promo_count", ui_lang)]
                + ([t("bench.col_avg_discount", ui_lang)] if _disc_col_b in _agg_dict else [])
                + ([t("bench.col_median_price", ui_lang)] if _price_col_b in _agg_dict else [])
            )
            _bench_grp = _bench_grp.rename(columns={
                _brand_col_b: t("bench.col_brand", ui_lang),
                _store_col_b: t("bench.col_retailer", ui_lang),
            })

            if len(_bench_grp) < 2:
                st.info(t("bench.no_data", ui_lang))
            else:
                _brand_col_label  = t("bench.col_brand", ui_lang)
                _retail_col_label = t("bench.col_retailer", ui_lang)
                _count_col_label  = t("bench.col_promo_count", ui_lang)

                # ── Heatmap ───────────────────────────────────────────────────
                _hm_brands = _bench_grp[_brand_col_label].dropna().unique()
                _hm_stores = _bench_grp[_retail_col_label].dropna().unique()

                if len(_hm_brands) >= 2 and len(_hm_stores) >= 2:
                    _pivot = _bench_grp.pivot_table(
                        index=_brand_col_label,
                        columns=_retail_col_label,
                        values=_count_col_label,
                        aggfunc="sum",
                        fill_value=0,
                    )
                    _fig_hm = px.imshow(
                        _pivot,
                        color_continuous_scale="Blues",
                        aspect="auto",
                        title=t("bench.heatmap_title", ui_lang)
                              if t("bench.heatmap_title", ui_lang) != "bench.heatmap_title"
                              else "Promo Frequency",
                        labels={"color": _count_col_label},
                    )
                    _fig_hm.update_layout(margin=dict(t=40, b=20))
                    st.plotly_chart(_fig_hm, use_container_width=True)

                # ── Summary table ─────────────────────────────────────────────
                st.dataframe(
                    _bench_grp.sort_values(_count_col_label, ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    height=350,
                )


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 5 – Alerts
# ╚══════════════════════════════════════════════════════════════════════════════

with tab_alerts:
    st.markdown(
        f'<div class="section-header">🔔 {t("alert.title", ui_lang)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(t("alert.subtitle", ui_lang))

    # ── Compute rule-based alerts from shared forecast data ───────────────────
    _alert_rows = []

    for _fc in _forecasts_all:
        _a_product  = _fc.product or "—"
        _a_brand    = _fc.brand or "—"
        _a_retailer = _fc.retailer or "—"
        _a_market   = _fc.country or "—"

        if _fc.days_since_last_promo is not None and _fc.avg_cycle_days is not None:
            _ratio = _fc.days_since_last_promo / max(_fc.avg_cycle_days, 1)

            if _ratio > 1.2:
                _days_over = _fc.days_since_last_promo - _fc.avg_cycle_days
                _alert_rows.append({
                    t("alert.col_severity", ui_lang): t("alert.sev_high", ui_lang),
                    t("alert.col_type", ui_lang):     t("alert.type_overdue", ui_lang),
                    t("alert.col_product", ui_lang):  _a_product[:40],
                    t("alert.col_brand", ui_lang):    _a_brand,
                    t("alert.col_retailer", ui_lang): _a_retailer,
                    t("alert.col_market", ui_lang):   _a_market,
                    t("alert.col_reason", ui_lang):   f"{_fc.days_since_last_promo}d / {_fc.avg_cycle_days}d avg",
                    "_sort": 0,
                })
            elif _ratio > 0.85:
                _alert_rows.append({
                    t("alert.col_severity", ui_lang): t("alert.sev_medium", ui_lang),
                    t("alert.col_type", ui_lang):     t("alert.type_near_cycle", ui_lang),
                    t("alert.col_product", ui_lang):  _a_product[:40],
                    t("alert.col_brand", ui_lang):    _a_brand,
                    t("alert.col_retailer", ui_lang): _a_retailer,
                    t("alert.col_market", ui_lang):   _a_market,
                    t("alert.col_reason", ui_lang):   f"{_fc.days_since_last_promo}d / {_fc.avg_cycle_days}d avg",
                    "_sort": 1,
                })

        # Price below historical minimum
        if (_fc.last_promo_price and _fc.min_promo_price_12m
                and _fc.last_promo_price < _fc.min_promo_price_12m * 0.9):
            _alert_rows.append({
                t("alert.col_severity", ui_lang): t("alert.sev_high", ui_lang),
                t("alert.col_type", ui_lang):     t("alert.type_price_low", ui_lang),
                t("alert.col_product", ui_lang):  _a_product[:40],
                t("alert.col_brand", ui_lang):    _a_brand,
                t("alert.col_retailer", ui_lang): _a_retailer,
                t("alert.col_market", ui_lang):   _a_market,
                t("alert.col_reason", ui_lang):   (
                    f"{format_price(_fc.last_promo_price, market_currency)} < "
                    f"{format_price(_fc.min_promo_price_12m, market_currency)} min"
                ),
                "_sort": 0,
            })

    if not _alert_rows:
        st.success(t("alert.no_alerts", ui_lang))
    else:
        _alert_df = pd.DataFrame(_alert_rows).sort_values("_sort").drop(columns=["_sort"])

        # ── Alert KPIs ────────────────────────────────────────────────────────
        _sev_col = t("alert.col_severity", ui_lang)
        _n_high_a  = int((_alert_df[_sev_col] == t("alert.sev_high", ui_lang)).sum()) if _sev_col in _alert_df.columns else 0
        _n_med_a   = int((_alert_df[_sev_col] == t("alert.sev_medium", ui_lang)).sum()) if _sev_col in _alert_df.columns else 0
        _n_low_a   = int((_alert_df[_sev_col] == t("alert.sev_low", ui_lang)).sum()) if _sev_col in _alert_df.columns else 0

        _al1, _al2, _al3 = st.columns(3)
        _al1.metric(t("alert.sev_high", ui_lang),   str(_n_high_a))
        _al2.metric(t("alert.sev_medium", ui_lang), str(_n_med_a))
        _al3.metric(t("alert.sev_low", ui_lang),    str(_n_low_a))

        st.markdown("<br>", unsafe_allow_html=True)

        if _n_high_a > 0:
            st.warning(f"⚠️ {_n_high_a} {t('alert.type_overdue', ui_lang)} or price alerts — review recommended.")

        st.dataframe(
            _alert_df,
            use_container_width=True,
            hide_index=True,
            height=min(35 * len(_alert_df) + 45, 500),
        )


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 6 – Data Quality & Audit
# ╚══════════════════════════════════════════════════════════════════════════════

with tab_quality:
    st.markdown(
        f'<div class="section-header">🛡️ {t("db.title", ui_lang)}</div>',
        unsafe_allow_html=True,
    )

    # ── Database status + markets ─────────────────────────────────────────────
    db_c1, db_c2 = st.columns([1, 2])
    with db_c1:
        _dbs_color = "#059669" if db_connected else "#D97706"
        _dbs_text  = t("build.production", ui_lang) if db_connected else t("build.demo", ui_lang)
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">{t('db.status', ui_lang)}</div>
              <div class="kpi-value" style="font-size:1.2rem; color:{_dbs_color};">● {_dbs_text}</div>
              <div class="kpi-delta">{t('db.source', ui_lang)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">{t('db.covered_markets', ui_lang)}</div>
              <div class="kpi-value" style="font-size:1.1rem;">HR · SI · BA · RS · MK · ME</div>
              <div class="kpi-delta">{" · ".join(t(f"country.{k}", ui_lang) for k in COUNTRY_KEYS)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with db_c2:
        @st.cache_data(ttl=300, show_spinner=False)
        def _retailer_dist(c, _v=CACHE_VERSION): return get_retailer_distribution(c)
        dist_df = _retailer_dist(sel_country)

        if not dist_df.empty:
            top10 = dist_df.head(10)
            fig_pie = px.pie(
                top10, names="store_name", values="cnt",
                title=t("db.top_retailers", ui_lang),
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.45,
            )
            fig_pie.update_layout(
                margin=dict(t=40, b=10, l=10, r=10),
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Product data & filter ─────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">📋 {t("db.product_data", ui_lang)}</div>', unsafe_allow_html=True)

    qc1, qc2, qc3 = st.columns([2, 2, 1])
    with qc1:
        @st.cache_data(ttl=600, show_spinner=False)
        def _stores_q(c, _v=CACHE_VERSION): return [t("db.all_retailers", ui_lang)] + get_distinct_stores(c)
        q_stores = _stores_q(sel_country)
        q_store_idx = st.selectbox(
            t("db.retailer_filter", ui_lang),
            range(len(q_stores)),
            format_func=lambda i: q_stores[i],
            key="q_store",
        )
        q_store = None if q_store_idx == 0 else q_stores[q_store_idx]
    with qc2:
        q_n = st.select_slider(t("db.sample_size", ui_lang), [50, 100, 250, 500], value=100)
    with qc3:
        st.markdown("<br>", unsafe_allow_html=True)
        q_load = st.button(t("db.load_btn", ui_lang), type="primary", use_container_width=True, key="q_load")

    @st.cache_data(ttl=120, show_spinner=False)
    def _q_load(country, store, cat, n, _v=CACHE_VERSION):
        return load_products(country_code=country, store_name=store, category_l1=cat, limit=n)

    with st.spinner(t("general.loading", ui_lang)):
        q_raw = _q_load(sel_country, q_store, sel_cat, q_n)
        q_df, q_report = _apply_quality(q_raw)

    # ── DQ Report ─────────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">🛡️ {t("db.dq_report", ui_lang)}</div>', unsafe_allow_html=True)
    rq1, rq2, rq3, rq4, rq5 = st.columns(5)
    rq1.metric(t("db.records_total", ui_lang),     str(q_report["n_total"]))
    rq2.metric(t("db.brands_fixed", ui_lang),      str(q_report["n_brand_fixed"]))
    rq3.metric(t("db.categories_fixed", ui_lang),  str(q_report["n_cat_fixed"]))
    rq4.metric(t("db.prices_valid", ui_lang),
               str(q_report["n_total"] - q_report.get("n_price_swapped", 0) - q_report.get("n_excluded", 0)))
    rq5.metric(t("db.prices_invalid", ui_lang),
               str(q_report.get("n_excluded", 0)),
               delta=f"-{q_report.get('n_excluded', 0)}" if q_report.get("n_excluded", 0) > 0 else None,
               delta_color="inverse")

    _dq_rate = q_report["n_clean"] / max(q_report["n_total"], 1) * 100
    st.caption(f"**DQ rate: {_dq_rate:.1f} %** ({q_report['n_clean']} / {q_report['n_total']})")
    st.markdown("<br>", unsafe_allow_html=True)

    if not q_df.empty:
        _price_col_q = "price_eur" if "price_eur" in q_df.columns else "price"
        _orig_col_q  = "original_price_eur" if "original_price_eur" in q_df.columns else "original_price"
        q_show_cols = [c for c in [
            "store_name", "name", "brand_clean", _price_col_q, _orig_col_q,
            "discount_pct", "currency", "category_l1", "country_code",
            "valid_from", "valid_until",
        ] if c in q_df.columns]
        q_show = q_df[q_show_cols].copy()
        if "category_l1" in q_show.columns:
            q_show["category_l1"] = q_show["category_l1"].apply(lambda x: cat_local(x, ui_lang))

        _col_rename = {
            "store_name": t("db.col_retailer", ui_lang) if t("db.col_retailer", ui_lang) != "db.col_retailer" else t("monitor.col_retailer", ui_lang),
            "name":       t("monitor.col_product", ui_lang),
            "brand_clean": t("db.col_brand", ui_lang) if t("db.col_brand", ui_lang) != "db.col_brand" else t("monitor.col_brand", ui_lang),
            _price_col_q: t("monitor.col_promo_price", ui_lang),
            _orig_col_q:  t("monitor.col_regular_price", ui_lang),
            "discount_pct": t("monitor.col_discount", ui_lang),
            "currency":   t("monitor.col_currency", ui_lang),
            "category_l1": t("db.col_category", ui_lang) if t("db.col_category", ui_lang) != "db.col_category" else t("monitor.col_category", ui_lang),
            "country_code": t("monitor.col_market", ui_lang),
            "valid_from": t("monitor.col_valid_from", ui_lang),
            "valid_until": t("monitor.col_valid_until", ui_lang),
        }
        q_show = q_show.rename(columns={k: v for k, v in _col_rename.items() if k in q_show.columns})

        _qcfg: dict = {}
        _p_col_disp = t("monitor.col_promo_price", ui_lang)
        _o_col_disp = t("monitor.col_regular_price", ui_lang)
        _d_col_disp = t("monitor.col_discount", ui_lang)
        if _p_col_disp in q_show.columns:
            _qcfg[_p_col_disp] = st.column_config.NumberColumn(_p_col_disp, format="%.2f")
        if _o_col_disp in q_show.columns:
            _qcfg[_o_col_disp] = st.column_config.NumberColumn(_o_col_disp, format="%.2f")
        if _d_col_disp in q_show.columns:
            _qcfg[_d_col_disp] = st.column_config.NumberColumn(_d_col_disp, format="%.1f %%")

        st.dataframe(q_show, column_config=_qcfg, use_container_width=True, hide_index=True, height=350)

    st.divider()

    # ── Category distribution ─────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">📊 {t("db.category_dist", ui_lang)}</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=300, show_spinner=False)
    def _cat_dist(c, _v=CACHE_VERSION): return get_category_distribution(c)
    cat_df = _cat_dist(sel_country)

    if not cat_df.empty:
        cat_df["_label"] = cat_df["category_l1"].apply(lambda x: cat_local(x, ui_lang))
        fig_cat = px.bar(
            cat_df.head(15),
            x="cnt", y="_label", orientation="h",
            color="cnt", color_continuous_scale="Blues",
            labels={"cnt": t("db.col_count", ui_lang) if t("db.col_count", ui_lang) != "db.col_count" else "Count", "_label": ""},
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
    st.markdown(f'<div class="section-header">🔗 {t("db.entity_matching", ui_lang)}</div>', unsafe_allow_html=True)
    st.caption(t("db.matching_subtitle", ui_lang) + f" · min score: {match_threshold:.0%}")

    MASTER_LIST = [
        "Milka Chocolate 300g",
        "Coca-Cola 1.5L PET",
        "Ritter Sport Milk Chocolate 100g",
        "Haribo Gummibears 200g",
        "Nutella Hazelnut Spread 450g",
        "Organic Whole Milk 3.5% 1L",
        "Ariel Laundry Detergent 20 WL",
        "Pampers Baby-Dry Size 3 44pcs",
        "Red Bull Energy Drink 250ml",
        "Pringles Original 185g",
    ]

    match_sample = q_df.head(20) if not q_df.empty else _q_load(sel_country, None, None, 20).head(20)
    matcher = ProductMatcher(threshold=match_threshold)
    raw_cats = match_sample["category_l1"].fillna("").tolist() if "category_l1" in match_sample.columns else None

    with st.spinner("Entity Resolution…"):
        results = matcher.batch_match(match_sample["name"].tolist(), MASTER_LIST, raw_categories=raw_cats)

    _show_debug = st.checkbox(
        "🔍 " + (t("monitor.filter_reliable", ui_lang) + " — debug mode"),
        value=False,
        key="em_debug",
    )

    match_df = match_sample[[c for c in ["store_name", "name", "price", "country_code"] if c in match_sample.columns]].copy()
    match_df["master_product"] = [r.master_product for r in results]
    match_df["score"]          = [round(r.score, 4) for r in results]
    match_df["method"]         = [r.match_method for r in results]
    match_df["status"]         = [r.match_status for r in results]
    match_df["OK"]             = ["✅" if r.is_confident else "⚠️" for r in results]

    if not _show_debug:
        match_df = match_df[match_df["status"].isin(["keyword_exact", "embedding_high_confidence"])]

    if match_df.empty:
        st.info("No matched products in sample. Enable debug mode for details.")
    else:
        _col_map_em = {
            "store_name":     t("monitor.col_retailer", ui_lang),
            "name":           t("monitor.col_product", ui_lang),
            "price":          t("monitor.col_promo_price", ui_lang),
            "country_code":   t("monitor.col_market", ui_lang),
            "master_product": "→ Master Product",
            "method":         "Method",
            "status":         "Status",
            "OK":             "OK",
        }
        match_cfg = {
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.2f"),
        }
        st.dataframe(
            match_df.rename(columns={k: v for k, v in _col_map_em.items() if k in match_df.columns}),
            column_config=match_cfg,
            use_container_width=True, hide_index=True,
        )

    conf_rate = sum(1 for r in results if r.is_confident) / max(len(results), 1) * 100
    unmatched_count = sum(1 for r in results if not r.is_confident)
    m1, m2, m3 = st.columns(3)
    m1.metric(t("dq.avg_match_score", ui_lang), f"{conf_rate:.1f} %")
    m2.metric("Min. score", f"{match_threshold:.0%}")
    m3.metric("Unmatched", str(unmatched_count))

    st.divider()

    # ── Promo Calendar ────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">📆 {t("db.calendar", ui_lang)}</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=180, show_spinner=False)
    def _cal_data(country, store, _v=CACHE_VERSION):
        return load_products(country_code=country, store_name=store, limit=500)

    cal_df = _cal_data(sel_country, q_store)

    if cal_df.empty:
        st.info(t("db.no_calendar", ui_lang))
    elif "valid_from" not in cal_df.columns or "valid_until" not in cal_df.columns:
        st.info(t("db.no_promo_dates", ui_lang))
    else:
        cal_clean = cal_df.dropna(subset=["valid_from", "valid_until"])
        if cal_clean.empty:
            st.info(t("db.no_promo_dates", ui_lang))
        elif not q_store and sel_country is None:
            st.info("Select a retailer or market to view the calendar.")
        else:
            n_cal = len(cal_clean)
            if n_cal > MAX_CALENDAR_ROWS:
                st.warning(f"Calendar has {n_cal:,} entries — showing first {MAX_CALENDAR_ROWS}.")
                cal_clean = cal_clean.sort_values("valid_from").head(MAX_CALENDAR_ROWS).copy()
            else:
                cal_clean = cal_clean.sort_values("valid_from").copy()

            if "category_l1" in cal_clean.columns:
                cal_clean["_cat_label"] = cal_clean["category_l1"].apply(lambda x: cat_local(x, ui_lang))
            if "store_name" in cal_clean.columns:
                fig_gantt = px.timeline(
                    cal_clean,
                    x_start="valid_from",
                    x_end="valid_until",
                    y="store_name",
                    color="_cat_label" if "_cat_label" in cal_clean.columns else "store_name",
                    hover_name="name" if "name" in cal_clean.columns else None,
                    labels={"store_name": t("monitor.col_retailer", ui_lang)},
                )
                fig_gantt.update_layout(margin=dict(t=20, b=10))
                st.plotly_chart(fig_gantt, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 7 – Experimental Forecast
# ╚══════════════════════════════════════════════════════════════════════════════

with tab_exp:
    st.warning(t("exp.disclaimer", ui_lang), icon="⚠️")

    _exp_tabs = st.tabs([t("exp.tab_kam", ui_lang), t("exp.tab_price", ui_lang)])

    # ── Sub-tab A: Rule-Based KAM Forecast ───────────────────────────────────
    with _exp_tabs[0]:
        st.markdown(
            f'<div class="section-header">🎯 {t("kam.title", ui_lang)}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            t("kam.min_req", ui_lang,
              n=MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER,
              d=MIN_HISTORY_DAYS)
        )

        # ── Forecast filters ──────────────────────────────────────────────────
        with st.container():
            fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
            with fc1:
                product_query = st.text_input(
                    t("kam.product_search", ui_lang),
                    value="",
                    placeholder="Milka, Coca-Cola, Red Bull…",
                    key="exp_product_query",
                )
            with fc2:
                @st.cache_data(ttl=600, show_spinner=False)
                def _fc_stores(c, _v=CACHE_VERSION):
                    return [t("kam.all_retailers", "EN")] + get_distinct_stores(c)
                fc_stores = _fc_stores(sel_country)
                fc_store_idx = st.selectbox(
                    t("kam.retailer", ui_lang), range(len(fc_stores)),
                    format_func=lambda i: fc_stores[i], key="exp_store",
                )
                fc_retailer = None if fc_store_idx == 0 else fc_stores[fc_store_idx]
            with fc3:
                min_probability = st.slider(
                    t("kam.min_probability", ui_lang),
                    0.0, 0.95, 0.50, 0.05, key="exp_min_prob",
                )
            with fc4:
                pred_window = st.slider(
                    t("kam.prediction_window", ui_lang),
                    7, 90, 30, 7, key="exp_pred_window",
                )

        fc_df = apply_forecast_filters(
            fc_df_all,
            retailer=fc_retailer,
            product_query=product_query,
            min_probability=min_probability,
            signal=None,
            price_trend=None,
            only_future=True,
            prediction_window_days=pred_window,
        )

        # ── KAM KPI row ───────────────────────────────────────────────────────
        n_high = int((fc_df["signal"] == "Hoch relevant").sum()) if not fc_df.empty else 0
        n_rising = int((fc_df["price_trend"] == "steigend").sum()) if not fc_df.empty else 0
        n_overdue_exp = 0
        if not fc_df.empty:
            n_overdue_exp = int(
                fc_df.apply(
                    lambda r: (
                        r.get("avg_cycle_days") is not None
                        and r.get("days_since_last_promo") is not None
                        and r["days_since_last_promo"] > r["avg_cycle_days"] * 1.2
                    ),
                    axis=1,
                ).sum()
            )
        avg_disc_exp = float("nan")
        if not fc_df.empty and "typical_discount_pct_max" in fc_df.columns:
            avg_disc_exp = fc_df["typical_discount_pct_max"].dropna().mean()

        kk1, kk2, kk3, kk4, kk5 = st.columns(5)
        kk1.metric(t("kpi.high_signal", ui_lang), str(n_high))
        kk2.metric(t("kpi.rising_price", ui_lang), str(n_rising))
        kk3.metric(t("kpi.overdue", ui_lang), str(n_overdue_exp))
        kk4.metric(t("kpi.forecast_base", ui_lang), f"{len(fc_df_all)}")
        kk5.metric(t("kpi.avg_discount", ui_lang), f"{avg_disc_exp:.0f} %" if pd.notna(avg_disc_exp) else "—")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Forecast table ────────────────────────────────────────────────────
        if fc_df.empty:
            if fc_df_all.empty:
                st.info(t("kam.no_history", ui_lang,
                          n=MIN_HISTORICAL_PROMOS_PER_PRODUCT_RETAILER,
                          d=MIN_HISTORY_DAYS))
            else:
                st.info(t("kam.no_filter_match", ui_lang))
        else:
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
                    return f"{med} {t('general.days', ui_lang)}"
                return f"{avg} {t('general.days', ui_lang)}"

            _signal_map = {
                "Hoch relevant":   t("signal.high", ui_lang),
                "Beobachten":      t("signal.watch", ui_lang),
                "Normal":          t("signal.normal", ui_lang),
                "Nicht belastbar": t("signal.unreliable", ui_lang),
                "Ungültig":        t("signal.invalid", ui_lang),
            }

            display = fc_df.copy()
            display[t("col.period", ui_lang)]      = display.apply(_fmt_period, axis=1)
            display[t("kam.price_trend", ui_lang)] = display.apply(_fmt_trend, axis=1)
            display[t("col.last_promo", ui_lang)]  = display.apply(_fmt_last_promo, axis=1)
            display[t("col.cycle", ui_lang)]       = display.apply(_fmt_cycle, axis=1)
            if "signal" in display.columns:
                display[t("col.signal", ui_lang)]  = display["signal"].map(_signal_map).fillna(display["signal"])
            if "probability" in display.columns:
                display[t("col.probability", ui_lang)] = display["probability"].apply(lambda x: f"{x:.0%}")
            if "typical_discount_pct_max" in display.columns:
                display[t("col.discount", ui_lang)] = display["typical_discount_pct_max"].apply(
                    lambda x: f"{x:.0f} %" if pd.notna(x) else "—"
                )

            _show_exp_cols = [
                "product", "brand", "retailer", "country",
                t("col.period", ui_lang),
                t("col.last_promo", ui_lang),
                t("col.cycle", ui_lang),
                t("col.signal", ui_lang),
                t("col.probability", ui_lang),
                t("col.discount", ui_lang),
                t("kam.price_trend", ui_lang),
                "justification",
            ]
            _show_exp_cols = [c for c in _show_exp_cols if c in display.columns]
            st.dataframe(
                display[_show_exp_cols].rename(columns={
                    "product":      t("cockpit.col_product", ui_lang),
                    "brand":        t("cockpit.col_brand", ui_lang),
                    "retailer":     t("cockpit.col_retailer", ui_lang),
                    "country":      t("monitor.col_market", ui_lang),
                    "justification": t("cockpit.col_justification", ui_lang),
                }),
                use_container_width=True,
                hide_index=True,
                height=min(35 * len(display) + 45, 600),
            )

    # ── Sub-tab B: Price Trend Forecast (Prophet) ─────────────────────────────
    with _exp_tabs[1]:
        st.markdown(
            f'<div class="section-header">📈 {t("forecast.tab_title", ui_lang)}</div>',
            unsafe_allow_html=True,
        )

        # ── REQUIRED: Product + Retailer + Market + Currency ──────────────────
        @st.cache_data(ttl=600, show_spinner=False)
        def _p_stores(c, _v=CACHE_VERSION):
            return get_distinct_stores(c)

        @st.cache_data(ttl=120, show_spinner=False)
        def _hist_price(term, store, country, _v=CACHE_VERSION):
            return load_price_history(product_name=term or None, retailer=store, country_code=country)

        fc_col1, fc_col2 = st.columns([3, 2])
        with fc_col1:
            search_term = st.text_input(
                t("forecast.product", ui_lang) + " *",
                value=brand_filter or "",
                placeholder="Wudy, Coca-Cola, Cedevita…",
                key="exp_fc_term",
            )
        with fc_col2:
            p_stores_fc = _p_stores(sel_country)
            sel_price_store_idx = st.selectbox(
                t("forecast.retailer", ui_lang) + " *",
                range(len(p_stores_fc) + 1),
                format_func=lambda i: "— select —" if i == 0 else p_stores_fc[i - 1],
                key="exp_retailer_pick",
            )
            price_store_val = None if sel_price_store_idx == 0 else p_stores_fc[sel_price_store_idx - 1]

        fc_col3, fc_col4, fc_col5 = st.columns([2, 1, 1])
        with fc_col3:
            fc_market = sel_country
            st.text_input(
                t("forecast.market", ui_lang) + " *",
                value=(t(f"country.{fc_market}", ui_lang) if fc_market else "— select market in sidebar —"),
                disabled=True, key="exp_market_disp",
            )
        with fc_col4:
            st.text_input(
                t("forecast.currency", ui_lang),
                value=market_currency,
                disabled=True, key="exp_currency_disp",
            )
        with fc_col5:
            forecast_periods = st.slider(
                t("forecast.horizon", ui_lang), 7, 90, 30, key="exp_prophet_periods",
            )

        selection_status = validate_forecast_selection(
            product=search_term, retailer=price_store_val,
            market=fc_market, currency=market_currency,
        )

        run_forecast = st.button(t("forecast.run", ui_lang), type="primary", key="exp_run_fc")

        if selection_status != "ok":
            st.warning(t("forecast.missing_selection", ui_lang))
        else:
            with st.spinner(t("general.loading", ui_lang)):
                hist_raw_fc = _hist_price(search_term, price_store_val, fc_market)
                hist_df_fc, hist_audit_fc = build_filtered_view(
                    hist_raw_fc,
                    country=fc_market,
                    brand_query=search_term,
                )

            _hist_price_col = "price_eur" if not hist_df_fc.empty and "price_eur" in hist_df_fc.columns else "price"
            _p_sym = "€" if market_currency == "EUR" else market_currency

            if not hist_df_fc.empty and "recorded_at" in hist_df_fc.columns and _hist_price_col in hist_df_fc.columns:
                st.markdown(
                    f'<div class="section-header">📈 {t("forecast.historical_price", ui_lang)} – {search_term}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"{len(hist_df_fc):,} · {price_store_val} · {fc_market} · {market_currency}")
                fig_hist = px.area(
                    hist_df_fc.sort_values("recorded_at"),
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
                st.info(t("forecast.too_few_points", ui_lang, n=len(hist_df_fc)))

            if run_forecast:
                obs_count_fc = int(len(hist_df_fc))
                history_days_fc = 0
                if obs_count_fc > 0 and "recorded_at" in hist_df_fc.columns:
                    _ts = pd.to_datetime(hist_df_fc["recorded_at"], errors="coerce").dropna()
                    if len(_ts) > 1:
                        history_days_fc = int((_ts.max() - _ts.min()).days)

                eligible = (obs_count_fc >= MIN_OBSERVATIONS and history_days_fc >= MIN_HISTORY_DAYS)

                if production_mode and not eligible:
                    st.error(
                        t("exp.eligibility_fail", ui_lang,
                          min_obs=MIN_OBSERVATIONS, min_days=MIN_HISTORY_DAYS)
                        + f"  ·  ({obs_count_fc} obs, {history_days_fc} days)"
                    )
                else:
                    with st.spinner(t("general.loading", ui_lang)):
                        prophet_fig, forecast_df = forecast_price_trend(
                            df=hist_df_fc,
                            product_id=search_term,
                            periods=forecast_periods,
                            allow_mock=not production_mode,
                            currency=market_currency,
                            lang=ui_lang,
                            title_prefix=f"{search_term} · {price_store_val}",
                        )

                    st.plotly_chart(prophet_fig, use_container_width=True)

                    if not forecast_df.empty:
                        fc_p = forecast_df.reset_index(drop=True)
                        _hist_sorted = hist_df_fc.sort_values("recorded_at") if "recorded_at" in hist_df_fc.columns else hist_df_fc
                        _last_series = _hist_sorted[_hist_price_col].dropna() if _hist_price_col in _hist_sorted.columns else pd.Series(dtype=float)
                        last_hist_price = float(_last_series.iloc[-1]) if len(_last_series) > 0 else float(fc_p["yhat"].iloc[0])

                        end_price  = float(fc_p["yhat"].iloc[-1])
                        min_price  = float(fc_p["yhat"].min())
                        max_price  = float(fc_p["yhat"].max())
                        min_pos    = int(fc_p["yhat"].idxmin())
                        max_pos    = int(fc_p["yhat"].idxmax())

                        if last_hist_price > 0:
                            price_change_pct = (end_price - last_hist_price) / last_hist_price * 100
                            max_drop_pct     = max(0.0, (last_hist_price - min_price) / last_hist_price * 100)
                            max_rise_pct     = max(0.0, (max_price - last_hist_price) / last_hist_price * 100)
                        else:
                            price_change_pct = max_drop_pct = max_rise_pct = 0.0

                        try:
                            min_date_str = fc_p["ds"].iloc[min_pos].strftime("%d.%m.%Y")
                            max_date_str = fc_p["ds"].iloc[max_pos].strftime("%d.%m.%Y")
                        except (IndexError, AttributeError):
                            min_date_str = "—"
                            max_date_str = "—"

                        drop_delta = t("forecast.no_drop", ui_lang) if max_drop_pct == 0 else f"−{max_drop_pct:.1f} %"
                        rise_delta = t("forecast.no_rise", ui_lang) if max_rise_pct == 0 else f"+{max_rise_pct:.1f} %"

                        fi1, fi2, fi3, fi4 = st.columns(4)
                        _exp_kpi_cards = [
                            (fi1, t("forecast.kpi.future_price", ui_lang, days=forecast_periods),
                             f"{end_price:.2f} {_p_sym}",
                             f"{'▼' if price_change_pct < 0 else '▲'} {abs(price_change_pct):.1f} %",
                             "kpi-card"),
                            (fi2, t("forecast.kpi.lowest", ui_lang),
                             f"{min_price:.2f} {_p_sym}", min_date_str, "kpi-card-green"),
                            (fi3, t("forecast.kpi.highest", ui_lang),
                             f"{max_price:.2f} {_p_sym}", max_date_str, "kpi-card-red"),
                            (fi4, t("forecast.kpi.max_drop", ui_lang),
                             f"{max_drop_pct:.1f} %", drop_delta, "kpi-card-amber"),
                        ]
                        for col, label, val, delta, css in _exp_kpi_cards:
                            col.markdown(
                                f"""<div class="kpi-card {css}">
                                  <div class="kpi-label">{label}</div>
                                  <div class="kpi-value">{val}</div>
                                  <div class="kpi-delta">{delta}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                        st.markdown("<br>", unsafe_allow_html=True)

                        # ── Forecast audit ────────────────────────────────────
                        work_for_mape = pd.DataFrame({
                            "ds": pd.to_datetime(hist_df_fc["recorded_at"], errors="coerce"),
                            "y": pd.to_numeric(hist_df_fc[_hist_price_col], errors="coerce"),
                        }).dropna().sort_values("ds")
                        mape, mae, bias = compute_backtest_mape(work_for_mape)
                        trust_key = compute_trust_level(mape, obs_count_fc, history_days_fc)

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
                                st.markdown(f"**{t('forecast.audit.observations', ui_lang)}:** {obs_count_fc}")
                                st.markdown(f"**{t('forecast.audit.history_days', ui_lang)}:** {history_days_fc}")
                            with a3:
                                mape_disp = f"{mape*100:.1f} %" if not pd.isna(mape) else "—"
                                st.markdown(f"**{t('forecast.audit.mape', ui_lang)}:** {mape_disp}")
                                st.markdown(f"**{t('forecast.currency', ui_lang)}:** {market_currency}")
                                st.markdown(f"**{t('forecast.retailer', ui_lang)}:** {price_store_val}")
                    else:
                        st.warning(t("forecast.prophet_failed", ui_lang))
