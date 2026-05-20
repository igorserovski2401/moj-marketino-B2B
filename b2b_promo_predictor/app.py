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
    load_upcoming_promos,
    to_feature_df,
)
from src.features import create_features
from src.matching import MIN_MATCH_SCORE, UNKNOWN_MASTER_PRODUCT, ProductMatcher
from src.model import forecast_price_trend, get_feature_importance, predict, train_lgbm

# ── Lokalisierung ─────────────────────────────────────────────────────────────

CATEGORY_DE: dict[str, str] = {
    "Hrana": "Lebensmittel",
    "Piće": "Getränke",
    "Pice": "Getränke",
    "Kozmetika": "Kosmetik & Pflege",
    "Kućanska kemija": "Haushaltschemie",
    "Kucanska kemija": "Haushaltschemie",
    "Dječja hrana": "Babynahrung",
    "Djecja hrana": "Babynahrung",
    "Mliječni proizvodi": "Milchprodukte",
    "Mljekarstvo": "Milchprodukte",
    "Mesni proizvodi": "Fleisch & Wurst",
    "Voće i povrće": "Obst & Gemüse",
    "Voce i povrce": "Obst & Gemüse",
    "Zamrznuta hrana": "Tiefkühlkost",
    "Slatkiši": "Süßwaren",
    "Slatkisi": "Süßwaren",
    "Pekarski proizvodi": "Backwaren",
    "Zdravlje i ljepota": "Gesundheit & Beauty",
    "Zdravlje": "Gesundheit",
    "Čišćenje": "Reinigung",
    "Ciscenje": "Reinigung",
    "Food": "Lebensmittel",
    "Non-Food": "Non-Food",
    "Other": "Sonstiges",
    "Храна": "Lebensmittel",
    "Пијалоци": "Getränke",
}

COUNTRY_LOCAL: dict[str, str] = {
    "HR": "Hrvatska 🇭🇷",
    "SI": "Slovenija 🇸🇮",
    "BA": "Bosna i Hercegovina 🇧🇦",
    "RS": "Srbija 🇷🇸",
    "MK": "Makedonija 🇲🇰",
    "ME": "Crna Gora 🇲🇪",
}

COUNTRY_DE: dict[str, str] = {
    "HR": "Kroatien 🇭🇷",
    "SI": "Slowenien 🇸🇮",
    "BA": "Bosnien & Herzegowina 🇧🇦",
    "RS": "Serbien 🇷🇸",
    "MK": "Nordmazedonien 🇲🇰",
    "ME": "Montenegro 🇲🇪",
}


def cat_de(name: str | None) -> str:
    """Fallback-Übersetzer für Kategorien ohne category_de-Spalte."""
    if not name:
        return "—"
    return CATEGORY_DE.get(name, name)


MAX_CALENDAR_ROWS = 500


def _apply_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Führt Quality Pipeline aus und gibt bereinigte Daten + Stats zurück."""
    if df.empty:
        return df, {"n_total": 0, "n_brand_fixed": 0, "n_cat_fixed": 0,
                    "n_price_swapped": 0, "n_excluded": 0, "n_clean": 0}
    clean, report = run_quality_pipeline(df)
    return clean, report._asdict()


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

with st.sidebar:
    st.markdown(
        f"""
        <div style='text-align:center; padding:.4rem 0 .8rem;'>
          <span style='font-size:1.6rem;'>📊</span><br>
          <strong style='font-size:1.1rem; color:#111827;'>moj-marketino</strong><br>
          <span style='font-size:.78rem; color:#6B7280;'>B2B Promo Intelligence</span><br>
          <span style='font-size:.7rem; color:#9CA3AF;'>v2.1 · Balkan Edition</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    db_connected = settings.has_supabase and get_client() is not None
    if db_connected:
        st.markdown(
            '<span class="badge-ok">● Live-Daten aktiv</span>',
            unsafe_allow_html=True,
        )
        st.caption("MarketinoDATABASE · EU-Central-2")
    else:
        st.markdown(
            '<span class="badge-warn">● Demo-Modus</span>',
            unsafe_allow_html=True,
        )
        st.caption("Keine Datenbankverbindung – Demo-Daten aktiv")

    st.divider()
    st.markdown("### 🔍 Filter")

    country_keys   = list(COUNTRY_DE.keys())
    country_labels = ["Alle Märkte"] + [COUNTRY_DE[k] for k in country_keys]
    country_keys_extended = ["__all__"] + country_keys
    sel_country_idx = st.selectbox(
        "Markt / Land",
        range(len(country_keys_extended)),
        format_func=lambda i: country_labels[i],
    )
    sel_country = None if sel_country_idx == 0 else country_keys_extended[sel_country_idx]

    @st.cache_data(ttl=600, show_spinner=False)
    def _sidebar_cats(c):
        return ["Alle Kategorien"] + get_distinct_categories(c)

    sidebar_cats = _sidebar_cats(sel_country)
    sel_cat_idx = st.selectbox(
        "Kategorie",
        range(len(sidebar_cats)),
        format_func=lambda i: f"{sidebar_cats[i]}  ({cat_de(sidebar_cats[i])})"
        if i > 0 else sidebar_cats[i],
    )
    sel_cat = None if sel_cat_idx == 0 else sidebar_cats[sel_cat_idx]

    brand_filter = st.text_input(
        "Marke / Hersteller",
        placeholder="z.B. Podravka, Milka…",
        help="Filtert Produktnamen nach diesem Begriff",
    )

    with st.expander("⚙️ Erweiterte Einstellungen"):
        match_threshold = st.slider(
            "Match-Konfidenz (min. 85 %)",
            min_value=MIN_MATCH_SCORE,
            max_value=0.99,
            value=MIN_MATCH_SCORE,
            step=0.05,
            help="Unterhalb dieses Schwellenwerts gilt ein Produkt als 'nicht zugeordnet'",
        )
        kw_vorschau = st.slider("Vorschau-Wochen", 1, 8, 4)

    st.divider()
    st.caption("© 2026 moj-marketino GmbH")

# ── Header ────────────────────────────────────────────────────────────────────

market_label = COUNTRY_DE.get(sel_country, "Alle Märkte") if sel_country else "Alle Märkte"
st.markdown(
    f"""
    <h1 style='margin-bottom:.1rem; font-size:1.7rem; font-weight:800; color:#111827;'>
      📊 Promo Intelligence Platform
    </h1>
    <p style='color:#6B7280; margin-top:.1rem; font-size:.92rem;'>
      Intelligente Promotionsvorschau für den Balkanmarkt · Markt: <strong>{market_label}</strong>
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
    "🔮 Promotion Predictor",
    "📉 Preisanalyse & Wettbewerb",
    "⚙️ Datenqualität & Pipeline",
])

# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Promotion Predictor
# ╚══════════════════════════════════════════════════════════════════════════════

with tab1:
    days_ahead = kw_vorschau * 7

    # ── Bevorstehende Aktionen direkt aus DB ──────────────────────────────────
    st.markdown('<div class="section-header">📅 Bevorstehende Aktionen</div>', unsafe_allow_html=True)
    st.caption(f"Produkte mit Aktionsbeginn in den nächsten {kw_vorschau} Wochen · Markt: {market_label}")

    @st.cache_data(ttl=180, show_spinner=False)
    def _upcoming(country, days, cat):
        df = load_upcoming_promos(country_code=country, days_ahead=days, limit=400)
        if cat and "category_l1" in df.columns:
            df = df[df["category_l1"] == cat]
        return df

    with st.spinner("Lade & bereinige Aktionsdaten…"):
        upcoming_raw = _upcoming(sel_country, days_ahead, sel_cat)
        upcoming_df, _uq = _apply_quality(upcoming_raw)

    if brand_filter and not upcoming_df.empty:
        _bcols = [c for c in ["brand", "manufacturer", "name", "master_product"] if c in upcoming_df.columns]
        if _bcols:
            _bmask = pd.Series(False, index=upcoming_df.index)
            for _bc in _bcols:
                _bmask |= upcoming_df[_bc].fillna("").str.contains(brand_filter, case=False, regex=False)
            upcoming_df = upcoming_df[_bmask]

    if not upcoming_df.empty:
        n_up = len(upcoming_df)
        n_retailers = upcoming_df["store_name"].nunique() if "store_name" in upcoming_df.columns else 0

        ua, ub, uc, ud = st.columns(4)
        ua.metric("Aktionen gefunden", str(n_up))
        ub.metric("Beteiligte Händler", str(n_retailers))
        if "discount_pct" in upcoming_df.columns:
            avg_disc = upcoming_df["discount_pct"].dropna().mean()
            uc.metric("Ø Rabatt", f"{avg_disc:.1f} %" if pd.notna(avg_disc) else "—")
        if _uq["n_excluded"] > 0:
            ud.metric("⚠️ Fehlerhafte Datensätze", str(_uq["n_excluded"]),
                      help="Ausgeschlossen wegen Preis-Fehler (Promo > Regulär)")

        st.markdown("<br>", unsafe_allow_html=True)

        # Bereinigte Tabelle — category_de aus Quality-Pipeline, Preise in EUR
        _up_cols = ["store_name", "name", "brand", "price_eur", "original_price_eur",
                    "discount_pct", "currency", "category_de", "country_code",
                    "valid_from", "valid_until", "discount_label"]
        up_show = upcoming_df[[c for c in _up_cols if c in upcoming_df.columns]].copy()

        # Fallback category_de wenn Spalte fehlt
        if "category_de" not in up_show.columns and "category_l1" in upcoming_df.columns:
            up_show["category_de"] = upcoming_df["category_l1"].apply(cat_de)

        up_show = up_show.rename(columns={
            "store_name": "Händler", "name": "Produkt", "brand": "Marke (bereinigt)",
            "price_eur": "Promo-Preis (€)", "original_price_eur": "Regulärpreis (€)",
            "discount_pct": "Rabatt %", "currency": "Währung",
            "category_de": "Kategorie", "country_code": "Markt",
            "valid_from": "Start", "valid_until": "Ende", "discount_label": "Rabatt-Label",
        })

        col_cfg: dict = {}
        if "Promo-Preis (€)" in up_show.columns:
            col_cfg["Promo-Preis (€)"] = st.column_config.NumberColumn("Promo-Preis (€)", format="%.2f €")
        if "Regulärpreis (€)" in up_show.columns:
            col_cfg["Regulärpreis (€)"] = st.column_config.NumberColumn("Regulärpreis (€)", format="%.2f €")
        if "Rabatt %" in up_show.columns:
            col_cfg["Rabatt %"] = st.column_config.ProgressColumn(
                "Rabatt %", min_value=0, max_value=70, format="%.1f %%",
            )

        st.dataframe(up_show, column_config=col_cfg, use_container_width=True, hide_index=True, height=380)

        # Treemap: Aktionen nach Händler & bereinigter Kategorie
        _tree_cat = "category_de" if "category_de" in upcoming_df.columns else "category_l1"
        if "store_name" in upcoming_df.columns and _tree_cat in upcoming_df.columns:
            treemap_df = (
                upcoming_df.groupby(["store_name", _tree_cat])
                .size().reset_index(name="Anzahl")
            )
            if _tree_cat != "category_de":
                treemap_df["category_de"] = treemap_df[_tree_cat].apply(cat_de)
                _tree_cat = "category_de"
            fig_tree = px.treemap(
                treemap_df,
                path=["store_name", _tree_cat],
                values="Anzahl",
                title=f"Aktionsverteilung – nächste {kw_vorschau} Wochen",
                color="Anzahl",
                color_continuous_scale="Blues",
            )
            fig_tree.update_layout(margin=dict(t=50, b=10))
            st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("Keine bevorstehenden Aktionen für diesen Filter gefunden.")

    st.divider()

    # ── ML-Vorhersage (LightGBM) ──────────────────────────────────────────────
    st.markdown('<div class="section-header">🤖 KI-Aktionsvorhersage</div>', unsafe_allow_html=True)
    st.caption("LightGBM-Modell erkennt Aktionsmuster aus historischen Zyklen – Wahrscheinlichkeit für nächste Promowelle.")

    ml_col1, ml_col2 = st.columns([4, 1])
    with ml_col2:
        train_btn = st.button("🔁 Modell neu trainieren", help="Trainiert auf aktuellen DB-Daten")

    trained_model = None
    if train_btn:
        with st.spinner("Trainiere auf Supabase-Daten…"):
            raw_train = load_products(country_code=sel_country, limit=2000)
            if len(raw_train) > 50:
                feat_df = to_feature_df(raw_train)
                trained_model = train_lgbm(feat_df)
                st.success(f"Modell trainiert · {len(feat_df):,} Datenpunkte")
            else:
                st.warning("Zu wenig Daten – Demo-Modell aktiv.")

    with st.spinner("Generiere Vorhersagen…"):
        pred_df = predict(model=trained_model)

    if not pred_df.empty:
        pred_show = pred_df.copy()
        pred_show["confidence_pct"] = pred_show["confidence"] * 100

        # Ampel-Signal
        def _signal(c: float) -> str:
            if c >= 0.85: return "🔴 Kritisch"
            if c >= 0.70: return "🟡 Wahrscheinlich"
            return "🟢 Möglich"

        pred_show["Signal"] = pred_show["confidence"].apply(_signal)

        pred_cfg = {
            "product": st.column_config.TextColumn("Produkt"),
            "retailer": st.column_config.TextColumn("Händler"),
            "predicted_promo_start": st.column_config.DateColumn("Vorhergesagter Start", format="DD.MM.YYYY"),
            "confidence_pct": st.column_config.ProgressColumn(
                "Wahrscheinlichkeit", min_value=0, max_value=100, format="%.0f %%",
            ),
            "expected_price": st.column_config.NumberColumn("Erw. Preis", format="%.2f €"),
            "Signal": st.column_config.TextColumn("Signal"),
        }

        display_cols = [c for c in ["product", "retailer", "predicted_promo_start", "confidence_pct", "expected_price", "Signal"] if c in pred_show.columns]
        st.dataframe(
            pred_show[display_cols],
            column_config=pred_cfg,
            use_container_width=True,
            hide_index=True,
        )


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Preisanalyse & Wettbewerb
# ╚══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">🔍 Produktsuche</div>', unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns([3, 2, 1])
    with pc1:
        search_term = st.text_input(
            "Produkt suchen",
            value=brand_filter or "",
            placeholder="z.B. Milka, Coca-Cola, Podravka Vegeta…",
        )
    with pc2:
        @st.cache_data(ttl=600, show_spinner=False)
        def _p_stores(c):
            return ["Alle Händler"] + get_distinct_stores(c)
        p_stores = _p_stores(sel_country)
        sel_price_store_idx = st.selectbox(
            "Händler", range(len(p_stores)),
            format_func=lambda i: p_stores[i], key="price_store",
        )
        price_store_val = None if sel_price_store_idx == 0 else p_stores[sel_price_store_idx]
    with pc3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔍 Suchen", type="primary", use_container_width=True, key="search_btn2")

    st.divider()

    # ── Preishistorie ─────────────────────────────────────────────────────────
    @st.cache_data(ttl=120, show_spinner=False)
    def _hist(term, store, country):
        return load_price_history(product_name=term or None, retailer=store, country_code=country)

    with st.spinner("Lade Preishistorie…"):
        hist_df = _hist(search_term, price_store_val, sel_country)

    # Preisspalte ermitteln (price_eur wenn vorhanden, sonst price)
    _hist_price_col = "price_eur" if not hist_df.empty and "price_eur" in hist_df.columns else "price"

    if not hist_df.empty and "recorded_at" in hist_df.columns and _hist_price_col in hist_df.columns:
        prod_label = search_term or "Alle Produkte"
        st.markdown(f'<div class="section-header">📈 Preisverlauf – {prod_label}</div>', unsafe_allow_html=True)
        st.caption(f"{len(hist_df):,} Messpunkte · Preise normiert auf EUR")

        retailers_hist = hist_df["retailer"].dropna().unique().tolist() if "retailer" in hist_df.columns else []

        if len(retailers_hist) > 1:
            fig_hist = px.line(
                hist_df.sort_values("recorded_at"),
                x="recorded_at", y=_hist_price_col,
                color="retailer",
                markers=True,
                labels={"recorded_at": "Datum", _hist_price_col: "Preis (€)", "retailer": "Händler"},
            )
        else:
            fig_hist = px.area(
                hist_df.sort_values("recorded_at"),
                x="recorded_at", y=_hist_price_col,
                markers=True,
                color_discrete_sequence=[PRIMARY],
                labels={"recorded_at": "Datum", _hist_price_col: "Preis (€)"},
            )

        fig_hist.update_layout(
            hovermode="x unified",
            margin=dict(t=20, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
            yaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Preis-KPIs
        h1, h2, h3, h4 = st.columns(4)
        _pseries = hist_df[_hist_price_col]
        avg_p = _pseries.mean()
        min_p = _pseries.min()
        max_p = _pseries.max()
        last_p = hist_df.sort_values("recorded_at")[_hist_price_col].iloc[-1]
        trend = last_p - hist_df.sort_values("recorded_at")[_hist_price_col].iloc[0]
        trend_str = f"{'▲' if trend > 0 else '▼'} {abs(trend):.2f} € Trend"

        for col, label, val, delta, css in [
            (h1, "Historischer Tiefstpreis", f"{min_p:.2f} €", "Günstigstes beobachtet", "kpi-card-green"),
            (h2, "Ø Durchschnittspreis",     f"{avg_p:.2f} €", f"{len(hist_df)} Messpunkte", "kpi-card"),
            (h3, "Historischer Höchstpreis", f"{max_p:.2f} €", "Teuerstes beobachtet", "kpi-card-red"),
            (h4, "Aktueller Preis",          f"{last_p:.2f} €", trend_str, "kpi-card-amber"),
        ]:
            col.markdown(
                f"""<div class="kpi-card {css}">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value">{val}</div>
                  <div class="kpi-delta">{delta}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

    else:
        st.info("Keine Preishistorie gefunden – gib einen Produktnamen ein (z.B. Milka, Coca-Cola).")

    # ── Prophet Preistrend-Prognose ───────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">🔮 Preistrend-Prognose (KI-Forecast)</div>', unsafe_allow_html=True)
    st.caption(
        "KI-Modell (Prophet) erkennt Saisonalität & Trend. "
        "Erwartete Preissturz-Punkte werden automatisch markiert."
    )

    pf1, pf2 = st.columns([2, 1])
    with pf1:
        if not hist_df.empty and "product_name" in hist_df.columns:
            available_products = sorted(hist_df["product_name"].dropna().unique().tolist())
        else:
            available_products = []

        if available_products:
            forecast_product = st.selectbox(
                "Produkt für Prognose",
                ["— Alle Daten verwenden —"] + available_products,
                key="prophet_product",
            )
            forecast_product_val = None if forecast_product.startswith("—") else forecast_product
        else:
            st.caption("Suche oben nach einem Produkt, um eine spezifische Auswahl zu erhalten.")
            forecast_product_val = search_term or None

    with pf2:
        forecast_periods = st.slider("Prognose-Horizont (Tage)", 7, 90, 30, key="prophet_periods")

    run_forecast = st.button("▶ Prognose berechnen", type="primary")

    if run_forecast or True:
        with st.spinner("KI-Modell rechnet…"):
            try:
                prophet_fig, forecast_df = forecast_price_trend(
                    df=hist_df if not hist_df.empty else pd.DataFrame(),
                    product_id=forecast_product_val,
                    periods=forecast_periods,
                )
                st.plotly_chart(prophet_fig, use_container_width=True)

                if forecast_df.empty:
                    st.warning("Keine ausreichenden Vorhersagedaten für diesen Zeitraum verfügbar.")
                else:
                    # Sicher auf Index zugreifen — reset_index verhindert iloc-Fehler
                    fc = forecast_df.reset_index(drop=True)
                    _price_col = "price_eur" if "price_eur" in hist_df.columns else "price"
                    last_hist_price = (
                        hist_df[_price_col].iloc[-1]
                        if not hist_df.empty and _price_col in hist_df.columns
                        else fc["yhat"].iloc[0]
                    )

                    min_pos    = fc["yhat"].idxmin()
                    max_pos    = fc["yhat"].idxmax()
                    end_price  = fc["yhat"].iloc[-1]
                    min_price  = fc["yhat"].min()
                    max_price  = fc["yhat"].max()
                    drop_pct   = (last_hist_price - min_price) / last_hist_price * 100 if last_hist_price else 0

                    try:
                        min_date = fc["ds"].iloc[min_pos].strftime("%d.%m.%Y")
                        max_date = fc["ds"].iloc[max_pos].strftime("%d.%m.%Y")
                    except (IndexError, AttributeError):
                        min_date = "—"
                        max_date = "—"

                    fi1, fi2, fi3, fi4 = st.columns(4)
                    for col, label, val, delta, css in [
                        (fi1, f"Preis in {forecast_periods} Tagen",
                         f"{end_price:.2f} €",
                         f"{'▼' if end_price < last_hist_price else '▲'} {abs(end_price - last_hist_price):.2f} € vs. heute",
                         "kpi-card"),
                        (fi2, "Erwarteter Tiefstpreis",
                         f"{min_price:.2f} €",
                         f"am {min_date}",
                         "kpi-card-green"),
                        (fi3, "Erwarteter Höchstpreis",
                         f"{max_price:.2f} €",
                         f"am {max_date}",
                         "kpi-card-red"),
                        (fi4, "Max. erw. Preissturz",
                         f"{drop_pct:.1f} %",
                         "Ø vs. Tiefpunkt",
                         "kpi-card-amber"),
                    ]:
                        col.markdown(
                            f"""<div class="kpi-card {css}">
                              <div class="kpi-label">{label}</div>
                              <div class="kpi-value">{val}</div>
                              <div class="kpi-delta">{delta}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    st.markdown("<br>", unsafe_allow_html=True)

                    with st.expander("📋 Prognosedaten anzeigen"):
                        fc_display = fc.copy()
                        fc_display["ds"] = fc_display["ds"].dt.strftime("%d.%m.%Y")
                        st.dataframe(
                            fc_display.rename(columns={
                                "ds": "Datum",
                                "yhat": "Prognose (€)",
                                "yhat_lower": "Untergrenze (€)",
                                "yhat_upper": "Obergrenze (€)",
                            }).style.format({
                                "Prognose (€)": "{:.3f}",
                                "Untergrenze (€)": "{:.3f}",
                                "Obergrenze (€)": "{:.3f}",
                            }),
                            use_container_width=True, hide_index=True,
                        )
            except Exception as e:
                st.warning(
                    "Keine ausreichenden Vorhersagedaten für diesen Zeitraum verfügbar. "
                    "Bitte wähle ein anderes Produkt oder erweitere den Zeitraum."
                )

    # ── Preisverteilung ───────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">📦 Preisverteilung nach Kategorie</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=180, show_spinner=False)
    def _price_dist(country, store, cat):
        return load_products(country_code=country, store_name=store, category_l1=cat, limit=500)

    price_dist_df = _price_dist(sel_country, price_store_val, sel_cat)

    _box_price_col = "price_eur" if not price_dist_df.empty and "price_eur" in price_dist_df.columns else "price"
    if not price_dist_df.empty and _box_price_col in price_dist_df.columns and "category_l1" in price_dist_df.columns:
        price_clean = price_dist_df.dropna(subset=[_box_price_col, "category_l1"])
        if not price_clean.empty:
            top_cats = price_clean["category_l1"].value_counts().head(8).index.tolist()
            price_filtered = price_clean[price_clean["category_l1"].isin(top_cats)].copy()
            price_filtered["Kategorie"] = price_filtered["category_l1"].apply(
                lambda x: f"{cat_de(x)} ({x})" if cat_de(x) != x else x
            )
            fig_box = px.box(
                price_filtered,
                x="Kategorie", y=_box_price_col,
                color="Kategorie",
                labels={_box_price_col: "Preis (€)"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_box.update_layout(
                showlegend=False,
                margin=dict(t=20, b=10),
                xaxis_tickangle=-30,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_box, use_container_width=True)


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
            q_show["category_de"] = q_df["category_l1"].apply(cat_de)
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
        cat_df["Kategorie DE"] = cat_df["category_l1"].apply(cat_de)
        cat_df["label"] = cat_df.apply(
            lambda r: f"{r['Kategorie DE']} ({r['category_l1']})" if r["Kategorie DE"] != r["category_l1"] else r["category_l1"],
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
                cal_clean["Kategorie"] = cal_clean["category_l1"].apply(cat_de)
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
                st.info("Keine Händler-Informationen für den Kalender verfügbar.")
