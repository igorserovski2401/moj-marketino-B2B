"""moj-marketino – B2B Promo Intelligence Platform.
Commercial intelligence for Key Account Managers in FMCG – Balkan markets.
Data source: Supabase MarketinoDATABASE (997k+ products, 6 countries).
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings
from src.database import (
    get_client,
    get_distinct_categories,
    get_distinct_stores,
    get_store_name_variants,
    load_active_promos,
    load_promo_history_for_forecast,
    normalize_retailer_name,
)
from src.i18n import (
    DEFAULT_LANGUAGE_BY_MARKET,
    LANG_LABELS,
    SUPPORTED_LANGS,
    format_price,
    get_market_currency,
    t,
    translate_category,
)
from src.matching import MIN_MATCH_SCORE
from src.quality import run_quality_pipeline

# ── Build & runtime identity ──────────────────────────────────────────────────

APP_VERSION: str = os.getenv("GIT_COMMIT_SHA", "")
if not APP_VERSION:
    try:
        from subprocess import DEVNULL, check_output
        APP_VERSION = check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent.parent), stderr=DEVNULL,
        ).decode().strip() or "3tab-minimal"
    except Exception:
        APP_VERSION = "3tab-minimal"

CACHE_VERSION: str = APP_VERSION

COUNTRY_KEYS: list[str] = ["HR", "SI", "BA", "RS", "MK", "ME"]

# ── Helper functions ──────────────────────────────────────────────────────────

SEARCHABLE_FIELDS: list[str] = [
    "brand", "brand_clean", "manufacturer",
    "product", "product_name", "name", "raw_product_name",
    "normalized_product_name", "master_product", "description",
]

# Constants kept for backward-compatibility with existing tests
MIN_TREEMAP_ROWS = 5
_MIN_OBSERVATIONS = 12
_MIN_HISTORY_DAYS = 90


def _normalize_text(value) -> str:
    if value is None:
        return ""
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", value)


def _text_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = _normalize_text(query)
    if not q or df.empty:
        return df
    existing = [c for c in SEARCHABLE_FIELDS if c in df.columns]
    if not existing:
        return df
    mask = pd.Series(False, index=df.index)
    for col in existing:
        mask = mask | df[col].apply(lambda x: q in _normalize_text(x))
    return df[mask]


def _apply_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df.empty:
        return df, {"n_total": 0, "n_brand_fixed": 0, "n_cat_fixed": 0,
                    "n_price_swapped": 0, "n_excluded": 0, "n_clean": 0}
    clean, report = run_quality_pipeline(df)
    return clean, report._asdict()


def _dq_badge(row: pd.Series) -> str:
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


def build_filtered_view(
    raw_df: pd.DataFrame,
    *,
    country: str | None = None,
    category: str | None = None,
    retailer: str | None = None,
    brand_query: str = "",
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Single source of truth — all visible widgets use this pipeline."""
    audit = {
        "raw_rows": 0, "after_quality": 0, "after_market": 0,
        "after_category": 0, "after_retailer": 0, "after_brand": 0, "final": 0,
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
        # Expand canonical name to all raw DB variants (e.g. "Vero" → Vero 11/12/13/…)
        _variants = get_store_name_variants(retailer)
        if len(_variants) > 1:
            df = df[df["store_name"].isin(_variants)]
        else:
            df = df[df["store_name"] == retailer]
    audit["after_retailer"] = len(df)

    if brand_query:
        df = _text_search(df, brand_query)
    audit["after_brand"] = len(df)

    if limit is not None:
        df = df.head(limit)
    audit["final"] = len(df)
    return df, audit


def _get_normalized_retailers(country: str | None = None) -> list[str]:
    """Return deduplicated canonical retailer names."""
    raw = get_distinct_stores(country)
    seen: dict[str, str] = {}
    for name in raw:
        if not name:
            continue
        canonical = normalize_retailer_name(name)
        key = canonical.upper()
        if key not in seen:
            seen[key] = canonical
    return sorted(seen.values())


def _empty_state(msg: str) -> None:
    st.markdown(
        f"""<div style='padding:2rem;text-align:center;color:#6B7280;background:#F9FAFB;
                        border-radius:8px;border:1px dashed #D1D5DB;margin:.5rem 0;'>
              <span style='font-size:2rem;'>📭</span><br><br>{msg}</div>""",
        unsafe_allow_html=True,
    )


# ── Public aliases (kept for test-suite compatibility) ────────────────────────

def normalize_search_text(value) -> str:
    return _normalize_text(value)


def apply_brand_product_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    return _text_search(df, query)


def validate_forecast_selection(
    product: str | None,
    retailer: str | None,
    market: str | None,
    currency: str | None,
) -> str:
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
    import math
    if observations < _MIN_OBSERVATIONS or history_days < _MIN_HISTORY_DAYS:
        return "nicht_belastbar"
    if math.isnan(mape):
        return "eingeschr"
    if mape <= 0.20:
        return "belastbar"
    if mape <= 0.35:
        return "eingeschr"
    return "nicht_belastbar"


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
      .badge-ok   {{ background:#D1FAE5; color:#065F46; border-radius:6px;
                     padding:3px 10px; font-size:.75rem; font-weight:700; }}
      .badge-warn {{ background:#FEF3C7; color:#92400E; border-radius:6px;
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

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:.4rem 0 .8rem;'>
          <span style='font-size:1.6rem;'>📊</span><br>
          <strong style='font-size:1.1rem; color:#111827;'>moj-marketino</strong><br>
          <span style='font-size:.78rem; color:#6B7280;'>B2B Promo Intelligence</span>
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

    _mode_label = t("build.production", current_lang) if not demo_mode else t("build.demo", current_lang)
    _mode_color = "#059669" if not demo_mode else "#D97706"
    st.markdown(
        f"""
        <div style='background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;
                    padding:.55rem .7rem;font-size:.76rem;line-height:1.45;color:#374151;'>
          <div><b>{t('build.version', current_lang)}:</b> <code>{APP_VERSION}</code></div>
          <div><b>{t('build.mode', current_lang)}:</b>
               <span style='color:{_mode_color};font-weight:700;'>{_mode_label}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if demo_mode:
        st.markdown(
            '<br><span class="badge-warn">⚠ DEMO – sample data only</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<br><span class="badge-ok">● Live data active</span>', unsafe_allow_html=True)
        st.caption("MarketinoDATABASE · EU-Central-2")

    st.divider()

    lang_idx = st.selectbox(
        "🌐 Language",
        range(len(SUPPORTED_LANGS)),
        format_func=lambda i: LANG_LABELS[SUPPORTED_LANGS[i]],
        index=SUPPORTED_LANGS.index(current_lang) if current_lang in SUPPORTED_LANGS else 0,
        key="lang_picker",
        on_change=_on_lang_change,
    )
    ui_lang = SUPPORTED_LANGS[lang_idx]
    if st.session_state["ui_lang_manual"]:
        st.session_state["ui_lang"] = ui_lang
    ui_lang = st.session_state.get("ui_lang", ui_lang)

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
        return get_distinct_categories(c)

    _raw_cats = _sidebar_cats(sel_country)
    _all_cats_label = t("sidebar.all_categories", ui_lang)
    sidebar_cats = [_all_cats_label] + _raw_cats

    sel_cat_idx = st.selectbox(
        t("sidebar.category", ui_lang),
        range(len(sidebar_cats)),
        format_func=lambda i: translate_category(sidebar_cats[i], ui_lang) if i > 0 else sidebar_cats[0],
    )
    sel_cat = None if sel_cat_idx == 0 else sidebar_cats[sel_cat_idx]

    brand_filter = st.text_input(
        t("sidebar.brand", ui_lang),
        placeholder="Podravka, Milka, Wudy…",
    )

    st.divider()
    st.caption(f"© 2026 moj-marketino · Build {APP_VERSION}")

# ── Header ────────────────────────────────────────────────────────────────────

market_label = (
    t(f"country.{sel_country}", ui_lang) if sel_country
    else t("sidebar.all_markets", ui_lang)
)

if demo_mode:
    st.warning(
        "⚠️ **DEMO MODE** – Running on sample data only. "
        "Connect a Supabase database to activate live data.",
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

# ── Data loading ──────────────────────────────────────────────────────────────
# Active promos: valid_from <= today <= valid_until (the real snapshot)

@st.cache_data(ttl=300, show_spinner=False)
def _load_active(country, cat, _v=CACHE_VERSION):
    return load_active_promos(
        country_code=country,
        category_l1=cat,
        limit=2000,
        allow_mock=demo_mode,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _load_history(country, cat, _v=CACHE_VERSION):
    return load_promo_history_for_forecast(
        country_code=country, category_l1=cat, days_back=365, limit=5000,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _load_retailers(country, _v=CACHE_VERSION):
    return _get_normalized_retailers(country)


with st.spinner(t("general.loading", ui_lang)):
    _active_raw           = _load_active(sel_country, sel_cat)
    _history_raw          = _load_history(sel_country, sel_cat)
    _normalized_retailers = _load_retailers(sel_country)

# Apply brand text-search on top of DB result
_filtered_promos, _filter_audit = build_filtered_view(
    _active_raw,
    country=sel_country,
    category=sel_cat,
    brand_query=brand_filter,
)

# ── KPI row — from currently active promos ────────────────────────────────────

_n_promos    = len(_filtered_promos)
_n_retailers = (
    int(_filtered_promos["store_name"].nunique())
    if "store_name" in _filtered_promos.columns else 0
)
_n_brands    = (
    int(_filtered_promos["brand_clean"].nunique())
    if "brand_clean" in _filtered_promos.columns else 0
)
_avg_disc    = (
    _filtered_promos["discount_pct"].dropna().mean()
    if "discount_pct" in _filtered_promos.columns and not _filtered_promos.empty
    else float("nan")
)
_max_disc    = (
    _filtered_promos["discount_pct"].dropna().max()
    if "discount_pct" in _filtered_promos.columns and not _filtered_promos.empty
    else float("nan")
)

k1, k2, k3, k4 = st.columns(4)
for _col, _label, _val, _sub in [
    (k1, t("kpi.active_promos_week", ui_lang), f"{_n_promos:,}".replace(",", "."), market_label),
    (k2, t("kpi.active_retailers", ui_lang),   str(_n_retailers),                  ""),
    (k3, t("kpi.observed_products", ui_lang),  str(_n_brands),                     ""),
    (k4, t("kpi.avg_discount", ui_lang),
     f"{_avg_disc:.0f} %" if pd.notna(_avg_disc) else "—",
     f"max {_max_disc:.0f} %" if pd.notna(_max_disc) else ""),
]:
    _col.markdown(
        f"""<div class="kpi-card">
          <div class="kpi-label">{_label}</div>
          <div class="kpi-value">{_val}</div>
          <div class="kpi-delta">{_sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )
st.markdown("<br>", unsafe_allow_html=True)

# ── 3 Tabs ────────────────────────────────────────────────────────────────────

tab_monitor, tab_history, tab_briefing = st.tabs([
    "📋 " + t("tab.monitor", ui_lang),
    "📈 " + t("tab.history", ui_lang),
    "🎯 " + t("tab.cockpit", ui_lang),
])

# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Promo Monitor
# ╚══════════════════════════════════════════════════════════════════════════════

with tab_monitor:
    st.markdown(
        f'<div class="section-header">📋 {t("monitor.title", ui_lang)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(t("monitor.subtitle", ui_lang))

    _mon_df = _filtered_promos.copy()

    # ── Retailer filter inside tab ────────────────────────────────────────────
    _tf1, _tf2, _tf3 = st.columns([2, 2, 1])
    with _tf1:
        _mon_retailer_opts = [t("sidebar.all_markets", ui_lang)] + _normalized_retailers
        _mon_ret_idx = st.selectbox(
            t("monitor.col_retailer", ui_lang),
            range(len(_mon_retailer_opts)),
            format_func=lambda i: _mon_retailer_opts[i],
            key="mon_retailer",
        )
        _mon_retailer = None if _mon_ret_idx == 0 else _mon_retailer_opts[_mon_ret_idx]
    with _tf2:
        _mon_search = st.text_input(
            t("monitor.col_product", ui_lang),
            placeholder="Coca-Cola, Milka, Podravka…",
            key="mon_search",
        )
    with _tf3:
        _mon_only_disc = st.checkbox(
            t("monitor.filter_real_discount", ui_lang), value=True, key="mon_real_disc",
        )

    # Apply in-tab filters
    if _mon_retailer and "store_name" in _mon_df.columns:
        _mon_df = _mon_df[_mon_df["store_name"] == _mon_retailer]
    if _mon_search:
        _mon_df = _text_search(_mon_df, _mon_search)
    if _mon_only_disc and "discount_pct" in _mon_df.columns:
        _mon_df = _mon_df[_mon_df["discount_pct"].fillna(0) > 0]

    if _mon_df.empty:
        _empty_state(t("monitor.no_data", ui_lang))
    else:
        # ── Per-retailer discount bar chart (Nielsen IQ style) ────────────────
        _price_col = "price_eur" if "price_eur" in _mon_df.columns else "price"
        _orig_col  = "original_price_eur" if "original_price_eur" in _mon_df.columns else "original_price"

        if (
            "store_name" in _mon_df.columns
            and "discount_pct" in _mon_df.columns
            and len(_mon_df["store_name"].dropna().unique()) > 1
            and not _mon_retailer
        ):
            _chart_data = (
                _mon_df.groupby("store_name", as_index=False)
                .agg(
                    avg_disc=("discount_pct", "mean"),
                    sku_count=("name", "count"),
                )
                .sort_values("avg_disc", ascending=False)
                .head(15)
            )
            _fig_bar = px.bar(
                _chart_data,
                x="store_name",
                y="avg_disc",
                text="sku_count",
                labels={"store_name": t("monitor.col_retailer", ui_lang),
                        "avg_disc": t("kpi.avg_discount", ui_lang),
                        "sku_count": "SKUs"},
                color="avg_disc",
                color_continuous_scale=["#DBEAFE", "#1A56DB"],
                title=t("monitor.title", ui_lang) + f" — {market_label}",
            )
            _fig_bar.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(t=40, b=10, l=0, r=0),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#F3F4F6", ticksuffix=" %"),
            )
            _fig_bar.update_traces(texttemplate="%{text}", textposition="outside")
            st.plotly_chart(_fig_bar, use_container_width=True)

        # ── Table: Nielsen IQ column order ────────────────────────────────────
        # Sort: retailer → discount desc (deepest deals first per retailer)
        if "store_name" in _mon_df.columns and "discount_pct" in _mon_df.columns:
            _mon_df = _mon_df.sort_values(
                ["store_name", "discount_pct"], ascending=[True, False]
            )

        _show_cols_src = [
            ("store_name",   t("monitor.col_retailer", ui_lang)),
            ("country_code", t("monitor.col_market", ui_lang)),
            ("category_l1",  t("monitor.col_category", ui_lang)),
            ("brand_clean",  t("monitor.col_brand", ui_lang)),
            ("name",         t("monitor.col_product", ui_lang)),
            ("valid_until",  t("monitor.col_valid_until", ui_lang)),
            ("discount_pct", t("monitor.col_discount", ui_lang)),
            (_price_col,     t("monitor.col_promo_price", ui_lang)),
            (_orig_col,      t("monitor.col_regular_price", ui_lang)),
            ("currency",     t("monitor.col_currency", ui_lang)),
        ]
        _avail_src = [(s, d) for s, d in _show_cols_src if s in _mon_df.columns]
        _mon_show  = _mon_df[[s for s, _ in _avail_src]].copy()
        _mon_show.columns = [d for _, d in _avail_src]

        _cat_disp = t("monitor.col_category", ui_lang)
        if _cat_disp in _mon_show.columns:
            _mon_show[_cat_disp] = _mon_show[_cat_disp].apply(
                lambda x: translate_category(x, ui_lang) if pd.notna(x) else "—"
            )

        _cfg_mon: dict = {}
        _disc_d  = t("monitor.col_discount", ui_lang)
        _price_d = t("monitor.col_promo_price", ui_lang)
        _orig_d  = t("monitor.col_regular_price", ui_lang)
        if _disc_d in _mon_show.columns:
            _cfg_mon[_disc_d] = st.column_config.ProgressColumn(
                _disc_d, format="%.1f %%", min_value=0, max_value=100,
            )
        if _price_d in _mon_show.columns:
            _cfg_mon[_price_d] = st.column_config.NumberColumn(_price_d, format="%.2f")
        if _orig_d in _mon_show.columns:
            _cfg_mon[_orig_d] = st.column_config.NumberColumn(_orig_d, format="%.2f")

        st.markdown(
            f"**{len(_mon_show):,} SKUs** · {int(_mon_df['store_name'].nunique() if 'store_name' in _mon_df.columns else 0)} {t('kpi.active_retailers', ui_lang)}".replace(",", "."),
            unsafe_allow_html=False,
        )
        st.dataframe(
            _mon_show, use_container_width=True, hide_index=True,
            column_config=_cfg_mon, height=520,
        )

        _csv_bytes = _mon_show.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=t("monitor.export", ui_lang),
            data=_csv_bytes,
            file_name=f"promo_monitor_{sel_country or 'all'}.csv",
            mime="text/csv",
        )

# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Product History
# ╚══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.markdown(
        f'<div class="section-header">📈 {t("history.title", ui_lang)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(t("history.select_prompt", ui_lang))

    _hc1, _hc2 = st.columns(2)
    with _hc1:
        _hist_product_q = st.text_input(
            t("history.select_product", ui_lang),
            placeholder="Milka, Coca-Cola, Podravka…",
            key="hist_product_q",
        )
    with _hc2:
        _hist_store_opts = ["—"] + _normalized_retailers
        _hist_store_idx  = st.selectbox(
            t("history.select_retailer", ui_lang),
            range(len(_hist_store_opts)),
            format_func=lambda i: _hist_store_opts[i],
            key="hist_store_pick",
        )
        _hist_retailer = None if _hist_store_idx == 0 else _hist_store_opts[_hist_store_idx]

    if not _hist_product_q and not _hist_retailer:
        _empty_state(t("history.select_prompt", ui_lang))
    else:
        _hist_view, _ = build_filtered_view(
            _history_raw,
            country=sel_country,
            category=sel_cat,
            retailer=_hist_retailer,
            brand_query=_hist_product_q,
        )

        if _hist_view.empty:
            _empty_state(t("history.no_data", ui_lang))
        else:
            _n_hist      = len(_hist_view)
            _price_col_h = "price_eur" if "price_eur" in _hist_view.columns else "price"
            _disc_col_h  = "discount_pct"
            _date_col_h  = "valid_from"

            _hs1, _hs2, _hs3, _hs4 = st.columns(4)
            _hs1.metric(t("history.stat_total", ui_lang), str(_n_hist))

            _avg_d_h = (
                _hist_view[_disc_col_h].dropna().mean()
                if _disc_col_h in _hist_view.columns else float("nan")
            )
            _med_price_h = (
                _hist_view[_price_col_h].dropna().median()
                if _price_col_h in _hist_view.columns else None
            )
            _hs2.metric(
                t("history.stat_median_price", ui_lang),
                format_price(_med_price_h, market_currency) if _med_price_h is not None else "—",
            )
            _hs3.metric(
                t("kpi.avg_discount", ui_lang),
                f"{_avg_d_h:.0f} %" if pd.notna(_avg_d_h) else "—",
            )

            if _date_col_h in _hist_view.columns:
                _dates_h = pd.to_datetime(_hist_view[_date_col_h], errors="coerce").dropna()
                if len(_dates_h) > 0:
                    _days_since_h = (pd.Timestamp.now() - _dates_h.max()).days
                    _hs4.metric(t("history.stat_days_since", ui_lang), str(_days_since_h))

            st.markdown("<br>", unsafe_allow_html=True)

            if _date_col_h in _hist_view.columns and _price_col_h in _hist_view.columns:
                _chart_df = _hist_view.dropna(subset=[_date_col_h, _price_col_h]).copy()
                _chart_df[_date_col_h] = pd.to_datetime(_chart_df[_date_col_h], errors="coerce")
                _chart_df = _chart_df.sort_values(_date_col_h)
                if len(_chart_df) > 0:
                    _color_col_h = "store_name" if "store_name" in _chart_df.columns else None
                    _fig_h = px.scatter(
                        _chart_df,
                        x=_date_col_h,
                        y=_price_col_h,
                        color=_color_col_h,
                        size=_disc_col_h if _disc_col_h in _chart_df.columns else None,
                        hover_name="name" if "name" in _chart_df.columns else None,
                        hover_data={_disc_col_h: ":.1f"} if _disc_col_h in _chart_df.columns else {},
                        title=t("history.chart_title", ui_lang),
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

            _hist_show_cols = [
                (_date_col_h,          t("history.col_period", ui_lang)),
                ("store_name",         t("monitor.col_retailer", ui_lang)),
                ("name",               t("monitor.col_product", ui_lang)),
                ("brand_clean",        t("monitor.col_brand", ui_lang)),
                (_price_col_h,         t("history.col_price", ui_lang)),
                ("original_price_eur", t("monitor.col_regular_price", ui_lang)),
                (_disc_col_h,          t("history.col_discount", ui_lang)),
            ]
            _hist_avail  = [(s, d) for s, d in _hist_show_cols if s in _hist_view.columns]
            _hist_table  = (
                _hist_view[[s for s, _ in _hist_avail]]
                .sort_values(_date_col_h, ascending=False)
                .head(200)
            )
            _hist_table.columns = [d for _, d in _hist_avail]
            st.dataframe(_hist_table, use_container_width=True, hide_index=True, height=300)

# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 3 – KAM Account Briefing
# ╚══════════════════════════════════════════════════════════════════════════════

with tab_briefing:
    st.markdown(
        f'<div class="section-header">🎯 {t("cockpit.action_briefing", ui_lang)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(t("cockpit.subtitle", ui_lang))

    _br1, _br2 = st.columns(2)
    with _br1:
        _brief_retailer_opts = ["—"] + _normalized_retailers
        _brief_retailer_idx  = st.selectbox(
            t("cockpit.col_retailer", ui_lang),
            range(len(_brief_retailer_opts)),
            format_func=lambda i: _brief_retailer_opts[i],
            key="brief_retailer",
        )
        _brief_retailer = (
            None if _brief_retailer_idx == 0
            else _brief_retailer_opts[_brief_retailer_idx]
        )
    with _br2:
        _brief_cat_raw  = sidebar_cats[1:] if len(sidebar_cats) > 1 else []
        _brief_cat_opts = [t("sidebar.all_categories", ui_lang)] + _brief_cat_raw
        _brief_cat_idx  = st.selectbox(
            t("sidebar.category", ui_lang),
            range(len(_brief_cat_opts)),
            format_func=lambda i: translate_category(_brief_cat_opts[i], ui_lang) if i > 0 else _brief_cat_opts[0],
            key="brief_cat",
        )
        _brief_cat = None if _brief_cat_idx == 0 else _brief_cat_opts[_brief_cat_idx]

    # Always initialize briefing_df before any conditional block — prevents NameError
    briefing_df: pd.DataFrame = pd.DataFrame()

    if not _brief_retailer:
        _empty_state(t("cockpit.subtitle", ui_lang))
    else:
        _brief_view, _ = build_filtered_view(
            _history_raw,
            country=sel_country,
            category=_brief_cat or sel_cat,
            retailer=_brief_retailer,
            brand_query=brand_filter,
        )

        if _brief_view.empty:
            _empty_state(t("cockpit.no_briefings", ui_lang))
        else:
            _n_brief    = len(_brief_view)
            _n_brands_b = (
                int(_brief_view["brand_clean"].nunique())
                if "brand_clean" in _brief_view.columns else 0
            )
            _avg_d_b    = (
                _brief_view["discount_pct"].dropna().mean()
                if "discount_pct" in _brief_view.columns else float("nan")
            )
            _price_col_b = "price_eur" if "price_eur" in _brief_view.columns else "price"
            _date_col_b  = "valid_from"

            _days_last_b = None
            if _date_col_b in _brief_view.columns:
                _dates_b = pd.to_datetime(_brief_view[_date_col_b], errors="coerce").dropna()
                if len(_dates_b) > 0:
                    _days_last_b = (pd.Timestamp.now() - _dates_b.max()).days

            _bb1, _bb2, _bb3, _bb4 = st.columns(4)
            _bb1.metric(t("history.stat_total", ui_lang),     str(_n_brief))
            _bb2.metric(t("kpi.observed_products", ui_lang),  str(_n_brands_b))
            _bb3.metric(t("kpi.avg_discount", ui_lang),
                        f"{_avg_d_b:.0f} %" if pd.notna(_avg_d_b) else "—")
            _bb4.metric(t("history.stat_days_since", ui_lang),
                        str(_days_last_b) if _days_last_b is not None else "—")

            st.markdown("<br>", unsafe_allow_html=True)

            _brief_cols_src = [
                ("name",               t("cockpit.col_product", ui_lang)),
                ("brand_clean",        t("cockpit.col_brand", ui_lang)),
                ("category_l1",        t("monitor.col_category", ui_lang)),
                (_date_col_b,          t("history.col_period", ui_lang)),
                (_price_col_b,         t("history.col_price", ui_lang)),
                ("original_price_eur", t("monitor.col_regular_price", ui_lang)),
                ("discount_pct",       t("monitor.col_discount", ui_lang)),
                ("country_code",       t("monitor.col_market", ui_lang)),
            ]
            _brief_avail = [(s, d) for s, d in _brief_cols_src if s in _brief_view.columns]
            briefing_df  = (
                _brief_view[[s for s, _ in _brief_avail]]
                .sort_values(_date_col_b, ascending=False)
                .head(300)
                .copy()
            )
            briefing_df.columns = [d for _, d in _brief_avail]

            _cat_col_b = t("monitor.col_category", ui_lang)
            if _cat_col_b in briefing_df.columns:
                briefing_df[_cat_col_b] = briefing_df[_cat_col_b].apply(
                    lambda x: translate_category(x, ui_lang) if pd.notna(x) else "—"
                )

            # Inline search — placed before table so filtering is immediate
            _brief_q = st.text_input(
                t("filter.search_product", ui_lang),
                key="brief_product_q",
                placeholder="Milka, Coca-Cola…",
            )
            if _brief_q:
                _prod_col_b   = t("cockpit.col_product", ui_lang)
                _brand_col_b2 = t("cockpit.col_brand", ui_lang)
                _mask_b = pd.Series(False, index=briefing_df.index)
                for _sc in [_prod_col_b, _brand_col_b2]:
                    if _sc in briefing_df.columns:
                        _mask_b = _mask_b | briefing_df[_sc].str.contains(
                            _brief_q, case=False, na=False
                        )
                briefing_df = briefing_df[_mask_b]

            if briefing_df.empty:
                _empty_state(t("cockpit.no_briefings", ui_lang))
            else:
                _cfg_brief: dict = {}
                _disc_d_b  = t("monitor.col_discount", ui_lang)
                _price_d_b = t("history.col_price", ui_lang)
                _orig_d_b  = t("monitor.col_regular_price", ui_lang)
                if _disc_d_b in briefing_df.columns:
                    _cfg_brief[_disc_d_b] = st.column_config.NumberColumn(_disc_d_b, format="%.1f %%")
                if _price_d_b in briefing_df.columns:
                    _cfg_brief[_price_d_b] = st.column_config.NumberColumn(_price_d_b, format="%.2f")
                if _orig_d_b in briefing_df.columns:
                    _cfg_brief[_orig_d_b] = st.column_config.NumberColumn(_orig_d_b, format="%.2f")

                st.dataframe(
                    briefing_df, use_container_width=True, hide_index=True,
                    column_config=_cfg_brief, height=450,
                )

                _csv_b = briefing_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=t("monitor.export", ui_lang),
                    data=_csv_b,
                    file_name=f"briefing_{_brief_retailer}_{sel_country or 'all'}.csv",
                    mime="text/csv",
                )
