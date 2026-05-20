"""moj-marketino B2B Promo Intelligence Platform – Streamlit Dashboard.

Datenquelle: Supabase MarketinoDATABASE (997k+ Produkte, 6 Länder).

Tabs:
  1. Data Pipeline   – Live-Produktdaten aus Supabase + Entity Matching
  2. Prediction Engine – ML-Aktionsvorhersagen auf echten Daten
  3. Price Analytics  – Historischer Preisverlauf
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
from src.matching import ProductMatcher
from src.model import forecast_price_trend, get_feature_importance, predict, train_lgbm

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="moj-marketino | B2B Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#1A56DB"
ACCENT  = "#F05252"
GREEN   = "#10B981"

st.markdown(
    f"""
    <style>
      html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
      .block-container {{ padding-top: 1.2rem; }}
      .kpi-card {{
        background: #F9FAFB; border-radius: 10px;
        padding: 1.1rem 1.4rem;
        border-left: 4px solid {PRIMARY};
        box-shadow: 0 1px 4px rgba(0,0,0,.07);
        height: 100%;
      }}
      .kpi-label {{ font-size:.76rem; color:#6B7280; font-weight:600;
                    text-transform:uppercase; letter-spacing:.06em; }}
      .kpi-value {{ font-size:1.9rem; font-weight:700; color:#111827; line-height:1.15; }}
      .kpi-delta {{ font-size:.8rem; color:{GREEN}; }}
      .badge-ok   {{ background:#D1FAE5; color:#065F46; border-radius:4px;
                     padding:2px 8px; font-size:.75rem; font-weight:600; }}
      .badge-warn {{ background:#FEF3C7; color:#92400E; border-radius:4px;
                     padding:2px 8px; font-size:.75rem; font-weight:600; }}
      .badge-err  {{ background:#FEE2E2; color:#991B1B; border-radius:4px;
                     padding:2px 8px; font-size:.75rem; font-weight:600; }}
      .badge-live {{ background:#DBEAFE; color:#1E40AF; border-radius:4px;
                     padding:2px 8px; font-size:.75rem; font-weight:600; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 moj-marketino")
    st.markdown("**B2B Promo Intelligence**")
    st.caption("Version 2.0 · Supabase Edition")
    st.divider()

    # Connection status
    db_connected = settings.has_supabase and get_client() is not None
    if db_connected:
        st.markdown('<span class="badge-ok">🟢 Supabase verbunden</span>', unsafe_allow_html=True)
        st.caption("MarketinoDATABASE · EU-Central-2")
    else:
        st.markdown('<span class="badge-warn">⚠️ Demo-Modus (kein Supabase)</span>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    if settings.has_gemini:
        st.markdown('<span class="badge-ok">🟢 Gemini API aktiv</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">⚠️ Gemini: kein API-Key</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🌍 Filter")

    country_options = ["Alle"] + list(COUNTRY_NAMES.keys())
    country_labels  = ["Alle Länder"] + list(COUNTRY_NAMES.values())
    sel_country_idx = st.selectbox(
        "Land", range(len(country_options)),
        format_func=lambda i: country_labels[i],
    )
    sel_country = None if sel_country_idx == 0 else country_options[sel_country_idx]

    match_threshold = st.slider("Match-Konfidenz", 0.50, 0.95, 0.70, 0.05)
    st.divider()
    st.caption("© 2025 moj-marketino GmbH")

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <h1 style='margin-bottom:0;font-size:1.75rem;'>
      📊 Promo Intelligence Platform
    </h1>
    <p style='color:#6B7280;margin-top:.2rem;'>
      997k+ Produkte · 6 Länder · 212 Händler · Live-Daten aus MarketinoDATABASE
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
kpi_data = [
    (k1, "Produkte gesamt",    f"{stats['total_products']:,}".replace(",", "."),  "MarketinoDB"),
    (k2, "Aktive Händler",     str(stats["total_retailers"]),                      "212 Stores"),
    (k3, "Kataloge",           str(stats["total_catalogs"]),                       "1.769 Prospekte"),
    (k4, "Länder",             str(stats["total_countries"]),                      "HR · MK · SI · RS · BA · ME"),
    (k5, "Aktionen heute",     str(stats["active_promos"]),                        "Laufende Aktionen"),
]
for col, label, val, sub in kpi_data:
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
    "🔍 Data Pipeline",
    "🤖 Prediction Engine",
    "📈 Price Analytics",
])

# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Data Pipeline
# ╚══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Live-Produktdaten aus Supabase")

    # Filter-Zeile
    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 1])
    with fc1:
        @st.cache_data(ttl=600, show_spinner=False)
        def _stores(c): return ["Alle Händler"] + get_distinct_stores(c)
        stores = _stores(sel_country)
        sel_store_idx = st.selectbox("Händler", range(len(stores)), format_func=lambda i: stores[i])
        sel_store = None if sel_store_idx == 0 else stores[sel_store_idx]

    with fc2:
        @st.cache_data(ttl=600, show_spinner=False)
        def _cats(c): return ["Alle Kategorien"] + get_distinct_categories(c)
        cats = _cats(sel_country)
        sel_cat_idx = st.selectbox("Kategorie", range(len(cats)), format_func=lambda i: cats[i])
        sel_cat = None if sel_cat_idx == 0 else cats[sel_cat_idx]

    with fc3:
        n_products = st.select_slider("Datensätze laden", [50, 100, 250, 500], value=250)

    with fc4:
        st.markdown("<br>", unsafe_allow_html=True)
        load_btn = st.button("🔄 Laden", type="primary", use_container_width=True)

    st.divider()

    # ── Produkttabelle ────────────────────────────────────────────────────────
    @st.cache_data(ttl=120, show_spinner=False)
    def _load(country, store, cat, n):
        return load_products(country_code=country, store_name=store, category_l1=cat, limit=n)

    with st.spinner("Lade Produkte aus Supabase…"):
        prod_df = _load(sel_country, sel_store, sel_cat, n_products)

    left, right = st.columns([3, 2])

    with left:
        st.markdown(f"#### {len(prod_df):,} Produkte geladen")

        display_cols = {
            "store_name": "Händler",
            "name": "Produkt",
            "brand": "Marke",
            "price": "Preis",
            "original_price": "Orig.-Preis",
            "category_l1": "Kategorie",
            "country_code": "Land",
            "valid_from": "Von",
            "valid_until": "Bis",
            "discount_label": "Rabatt",
        }
        show_df = prod_df[[c for c in display_cols if c in prod_df.columns]].rename(columns=display_cols)

        # Preis-Formatierung
        if "Preis" in show_df.columns:
            show_df["Preis"] = show_df["Preis"].apply(
                lambda x: f"{x:.2f} €" if pd.notna(x) else "—"
            )
        if "Orig.-Preis" in show_df.columns:
            show_df["Orig.-Preis"] = show_df["Orig.-Preis"].apply(
                lambda x: f"{x:.2f} €" if pd.notna(x) else "—"
            )

        st.dataframe(show_df, use_container_width=True, hide_index=True, height=400)

    with right:
        # Händler-Donut
        @st.cache_data(ttl=300, show_spinner=False)
        def _retailer_dist(c): return get_retailer_distribution(c)
        dist_df = _retailer_dist(sel_country)

        if not dist_df.empty:
            top10 = dist_df.head(10)
            fig_pie = px.pie(
                top10, names="store_name", values="cnt",
                title="Top-10 Händler (Produktanzahl)",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.42,
            )
            fig_pie.update_layout(
                margin=dict(t=40, b=10, l=10, r=10),
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── Kategorie-Balken ──────────────────────────────────────────────────────
    @st.cache_data(ttl=300, show_spinner=False)
    def _cat_dist(c): return get_category_distribution(c)
    cat_df = _cat_dist(sel_country)

    if not cat_df.empty:
        st.markdown("#### Kategorie-Verteilung")
        fig_cat = px.bar(
            cat_df.head(12),
            x="cnt", y="category_l1", orientation="h",
            color="cnt", color_continuous_scale="Blues",
            labels={"cnt": "Anzahl Produkte", "category_l1": "Kategorie"},
        )
        fig_cat.update_layout(
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    st.divider()

    # ── Entity Matching ───────────────────────────────────────────────────────
    st.markdown("#### Entity Matching – Normierung auf Master-Produktliste")
    st.caption("Sentence-Transformer ordnet rohe Produktnamen der normierten Master-DB zu.")

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

    sample = prod_df.head(20)
    matcher = ProductMatcher(threshold=match_threshold)

    with st.spinner("Entity Resolution…"):
        results = matcher.batch_match(sample["name"].tolist(), MASTER_LIST)

    match_df = sample[["store_name", "name", "price", "country_code"]].copy()
    match_df["master_produkt"] = [r.master_product for r in results]
    match_df["score"] = [r.score for r in results]
    match_df["konfident"] = ["✅" if r.is_confident else "⚠️" for r in results]

    st.dataframe(
        match_df.rename(columns={
            "store_name": "Händler", "name": "Roh-Name",
            "price": "Preis", "country_code": "Land",
            "master_produkt": "→ Master", "score": "Score", "konfident": "OK",
        }),
        use_container_width=True, hide_index=True,
    )
    conf_rate = sum(1 for r in results if r.is_confident) / max(len(results), 1) * 100
    st.metric("Match-Rate (Sample)", f"{conf_rate:.1f} %")


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Prediction Engine
# ╚══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Aktionsvorhersage & Upcoming Promos")

    col_a, col_b, col_c = st.columns([1, 1, 2])

    with col_a:
        days_ahead = st.slider("Vorschau (Tage)", 3, 30, 14)
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        train_btn = st.button("🔁 ML-Modell trainieren", help="Trainiert LightGBM auf DB-Daten")

    trained_model = None

    # ── Upcoming Promos (direkt aus DB) ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📅 Bevorstehende Aktionen (direkt aus Supabase)")
    st.caption(f"Produkte mit Aktionsbeginn in den nächsten {days_ahead} Tagen")

    @st.cache_data(ttl=180, show_spinner=False)
    def _upcoming(country, days):
        return load_upcoming_promos(country_code=country, days_ahead=days, limit=300)

    with st.spinner("Lade bevorstehende Aktionen…"):
        upcoming_df = _upcoming(sel_country, days_ahead)

    if not upcoming_df.empty:
        # Gruppiert nach Händler
        up_display = upcoming_df[[
            c for c in ["store_name", "name", "brand", "price", "original_price",
                         "category_l1", "country_code", "valid_from", "valid_until",
                         "discount_label", "discount_depth"]
            if c in upcoming_df.columns
        ]].copy()

        if "discount_depth" in up_display.columns:
            up_display["Rabatt %"] = (up_display["discount_depth"] * 100).round(1).astype(str) + " %"
            up_display = up_display.drop(columns=["discount_depth"])

        st.dataframe(
            up_display.rename(columns={
                "store_name": "Händler", "name": "Produkt", "brand": "Marke",
                "price": "Aktionspreis", "original_price": "Regulärpreis",
                "category_l1": "Kategorie", "country_code": "Land",
                "valid_from": "Aktionsbeginn", "valid_until": "Aktionsende",
                "discount_label": "Rabatt-Label",
            }),
            use_container_width=True, hide_index=True, height=350,
        )

        # Treemap: Upcoming Promos nach Händler & Kategorie
        if "store_name" in upcoming_df.columns and "category_l1" in upcoming_df.columns:
            treemap_df = (
                upcoming_df.groupby(["store_name", "category_l1"])
                .size()
                .reset_index(name="cnt")
            )
            fig_tree = px.treemap(
                treemap_df, path=["store_name", "category_l1"], values="cnt",
                title="Bevorstehende Aktionen nach Händler & Kategorie",
                color="cnt", color_continuous_scale="Blues",
            )
            fig_tree.update_layout(margin=dict(t=40, b=10))
            st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("Keine bevorstehenden Aktionen für diesen Filter gefunden.")

    st.divider()

    # ── ML-Vorhersage ─────────────────────────────────────────────────────────
    st.markdown("#### 🤖 ML-Vorhersage (LightGBM)")
    st.caption("Trainiert auf historischen Aktionsmustern – zeigt Wahrscheinlichkeit für nächste Aktionswelle.")

    if train_btn:
        with st.spinner("Lade Trainingsdaten aus Supabase…"):
            raw_train = load_products(
                country_code=sel_country,
                limit=2000,
            )
            if len(raw_train) > 50:
                feat_df = to_feature_df(raw_train)
                trained_model = train_lgbm(feat_df)
                st.success(f"Modell trainiert auf {len(feat_df):,} Datenpunkten.")
            else:
                st.warning("Zu wenig Daten für Training – nutze Mock-Modell.")

    with st.spinner("Generiere Vorhersagen…"):
        pred_df = predict(model=trained_model)

    st.dataframe(
        pred_df.rename(columns={
            "product": "Produkt", "retailer": "Händler",
            "predicted_promo_start": "Vorhergesagter Aktionsbeginn",
            "expected_price": "Erw. Preis (€)",
            "confidence": "Konfidenz",
        }),
        use_container_width=True, hide_index=True,
    )

    # Feature Importance
    fi_df = get_feature_importance(trained_model)
    fig_fi = px.bar(
        fi_df, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Blues",
        title="Feature Importance",
        labels={"importance": "Wichtigkeit", "feature": "Feature"},
    )
    fig_fi.update_layout(
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── Aktuelle Aktionen nach Store ──────────────────────────────────────────
    st.divider()
    st.markdown("#### 🔥 Heutige Top-Aktionen")

    @st.cache_data(ttl=120, show_spinner=False)
    def _active(country): return load_active_promos(country_code=country, limit=100)

    active_df = _active(sel_country)

    if not active_df.empty and "price" in active_df.columns and "store_name" in active_df.columns:
        # Durchschnittspreis pro Händler heute
        avg_price = (
            active_df.groupby("store_name")["price"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "Ø Preis", "count": "Anzahl Aktionen", "store_name": "Händler"})
        )
        fig_bar = px.bar(
            avg_price.sort_values("Anzahl Aktionen", ascending=False).head(15),
            x="Händler", y="Anzahl Aktionen",
            color="Ø Preis", color_continuous_scale="RdYlGn_r",
            title="Aktuelle Aktionen heute – Top Händler",
            labels={"Ø Preis": "Ø Aktionspreis (€)"},
        )
        fig_bar.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Price Analytics
# ╚══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Preisanalyse & Historischer Verlauf")

    # ── Suche & Filter ────────────────────────────────────────────────────────
    pc1, pc2, pc3 = st.columns([3, 2, 1])
    with pc1:
        search_term = st.text_input("Produkt suchen", placeholder="z.B. Milka, Coca-Cola, Ariel…")
    with pc2:
        @st.cache_data(ttl=600, show_spinner=False)
        def _p_stores(c): return ["Alle"] + get_distinct_stores(c)
        p_stores = _p_stores(sel_country)
        sel_price_store = st.selectbox(
            "Händler", range(len(p_stores)),
            format_func=lambda i: p_stores[i], key="price_store",
        )
        price_store_val = None if sel_price_store == 0 else p_stores[sel_price_store]
    with pc3:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("🔍 Suchen", type="primary", use_container_width=True)

    st.divider()

    # ── Preishistorie ─────────────────────────────────────────────────────────
    @st.cache_data(ttl=120, show_spinner=False)
    def _hist(term, store, country):
        return load_price_history(product_name=term or None, retailer=store, country_code=country)

    with st.spinner("Lade Preishistorie…"):
        hist_df = _hist(search_term, price_store_val, sel_country)

    if not hist_df.empty and "recorded_at" in hist_df.columns and "price" in hist_df.columns:
        st.markdown(f"#### Preisverlauf – {search_term or 'Alle Produkte'}")
        st.caption(f"{len(hist_df):,} Datenpunkte aus `product_price_history`")

        # Einer oder mehrere Händler
        retailers_in_hist = hist_df["retailer"].dropna().unique().tolist() if "retailer" in hist_df.columns else []

        if len(retailers_in_hist) > 1:
            fig_hist = px.line(
                hist_df.sort_values("recorded_at"),
                x="recorded_at", y="price",
                color="retailer",
                markers=True,
                labels={"recorded_at": "Datum", "price": "Preis (€)", "retailer": "Händler"},
                title=f"Preisverlauf: {search_term or 'Alle'}",
            )
        else:
            fig_hist = px.area(
                hist_df.sort_values("recorded_at"),
                x="recorded_at", y="price",
                markers=True,
                color_discrete_sequence=[PRIMARY],
                labels={"recorded_at": "Datum", "price": "Preis (€)"},
                title=f"Preisverlauf: {search_term or 'Alle'}",
            )

        fig_hist.update_layout(hovermode="x unified", margin=dict(t=50, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

        # Stats
        s1, s2, s3, s4 = st.columns(4)
        for col, label, val in [
            (s1, "Ø Preis",    f"{hist_df['price'].mean():.2f} €"),
            (s2, "Min-Preis",  f"{hist_df['price'].min():.2f} €"),
            (s3, "Max-Preis",  f"{hist_df['price'].max():.2f} €"),
            (s4, "Messpunkte", str(len(hist_df))),
        ]:
            col.markdown(
                f"""<div class="kpi-card">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value">{val}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.info("Keine Preishistorie gefunden. Versuche einen anderen Suchbegriff.")

    # ── Prophet Preistrend-Prognose ───────────────────────────────────────────
    st.divider()
    st.markdown("#### 🔮 Prophet Preistrend-Prognose")
    st.caption(
        "Meta Prophet modelliert Saisonalität & Trend und prognostiziert den Preisverlauf. "
        "Erwartete Preissturz-Punkte werden automatisch annotiert."
    )

    pf1, pf2, pf3 = st.columns([3, 1, 1])
    with pf1:
        # Produktauswahl aus der Preishistorie
        if not hist_df.empty and "product_name" in hist_df.columns:
            available_products = sorted(hist_df["product_name"].dropna().unique().tolist())
        else:
            available_products = []

        if available_products:
            forecast_product = st.selectbox(
                "Produkt für Prognose",
                ["— automatisch (alle Daten) —"] + available_products,
                key="prophet_product",
            )
            forecast_product_val = None if forecast_product.startswith("—") else forecast_product
        else:
            st.caption("Suche zuerst ein Produkt oben, um eine Auswahl zu erhalten.")
            forecast_product_val = search_term or None

    with pf2:
        forecast_periods = st.slider("Prognose-Tage", 7, 90, 30, key="prophet_periods")

    with pf3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_forecast = st.button("▶ Prognose starten", type="primary", use_container_width=True)

    # Prognose ausführen (sofort beim ersten Laden oder auf Button-Klick)
    if run_forecast or True:
        with st.spinner("Prophet-Modell trainiert…"):
            prophet_fig, forecast_df = forecast_price_trend(
                df=hist_df if not hist_df.empty else pd.DataFrame(),
                product_id=forecast_product_val,
                periods=forecast_periods,
            )

        st.plotly_chart(prophet_fig, use_container_width=True)

        # Forecast-Insights
        if not forecast_df.empty:
            last_hist_price = hist_df["price"].iloc[-1] if not hist_df.empty and "price" in hist_df.columns else forecast_df["yhat"].iloc[0]
            min_idx     = forecast_df["yhat"].idxmin()
            max_idx     = forecast_df["yhat"].idxmax()
            end_price   = forecast_df["yhat"].iloc[-1]
            min_price   = forecast_df["yhat"].min()
            drop_pct    = (last_hist_price - min_price) / last_hist_price * 100 if last_hist_price else 0

            fi1, fi2, fi3, fi4 = st.columns(4)
            for col, label, val, delta in [
                (fi1, f"Preis in {forecast_periods} Tagen", f"{end_price:.2f} €",
                 f"{'▼' if end_price < last_hist_price else '▲'} {abs(end_price - last_hist_price):.2f} € vs. heute"),
                (fi2, "Erwarteter Tiefpunkt",
                 f"{min_price:.2f} €",
                 f"{forecast_df['ds'].iloc[min_idx].strftime('%d.%m.%Y')}"),
                (fi3, "Erwarteter Höchstpreis",
                 f"{forecast_df['yhat'].max():.2f} €",
                 f"{forecast_df['ds'].iloc[max_idx].strftime('%d.%m.%Y')}"),
                (fi4, "Max. erwarteter Preissturz",
                 f"{drop_pct:.1f} %",
                 "Ø vs. Tiefpunkt"),
            ]:
                col.markdown(
                    f"""<div class="kpi-card">
                      <div class="kpi-label">{label}</div>
                      <div class="kpi-value">{val}</div>
                      <div class="kpi-delta">{delta}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Forecast-Tabelle (kompakt, zusammenklappbar)
            with st.expander("📋 Rohdaten der Prognose anzeigen"):
                fc_display = forecast_df.copy()
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

    st.divider()

    # ── Preisverteilung aus Produkttabelle ────────────────────────────────────
    st.markdown("#### Preisverteilung nach Kategorie (aktuelle Produkte)")

    @st.cache_data(ttl=180, show_spinner=False)
    def _price_dist(country, store, cat):
        df = load_products(country_code=country, store_name=store, category_l1=cat, limit=500)
        return df

    price_dist_df = _price_dist(sel_country, price_store_val, None)

    if not price_dist_df.empty and "price" in price_dist_df.columns:
        price_clean = price_dist_df.dropna(subset=["price", "category_l1"]) if "category_l1" in price_dist_df.columns else price_dist_df.dropna(subset=["price"])

        if not price_clean.empty and "category_l1" in price_clean.columns:
            top_cats_price = price_clean["category_l1"].value_counts().head(8).index.tolist()
            price_filtered = price_clean[price_clean["category_l1"].isin(top_cats_price)]

            fig_box = px.box(
                price_filtered,
                x="category_l1", y="price",
                color="category_l1",
                title="Preisverteilung nach Top-Kategorien",
                labels={"category_l1": "Kategorie", "price": "Preis (€)"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_box.update_layout(
                showlegend=False,
                margin=dict(t=50, b=10),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Scatter: Preis vs. Menge
        if "amount" in price_clean.columns and "store_name" in price_clean.columns:
            scatter_df = price_clean.dropna(subset=["amount"]).head(300)
            if not scatter_df.empty:
                fig_scatter = px.scatter(
                    scatter_df,
                    x="amount", y="price",
                    color="store_name",
                    hover_data=["name"] if "name" in scatter_df.columns else None,
                    title="Preis vs. Menge (nach Händler)",
                    labels={"amount": "Menge", "price": "Preis (€)", "store_name": "Händler"},
                    opacity=0.7,
                )
                fig_scatter.update_layout(margin=dict(t=50, b=10))
                st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Aktionskalender ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📆 Aktionskalender – Gültigkeitsdaten")

    @st.cache_data(ttl=180, show_spinner=False)
    def _calendar_data(country, store):
        return load_products(country_code=country, store_name=store, limit=300)

    cal_df = _calendar_data(sel_country, price_store_val)

    if not cal_df.empty and "valid_from" in cal_df.columns and "valid_until" in cal_df.columns:
        cal_clean = cal_df.dropna(subset=["valid_from", "valid_until", "price"])
        if not cal_clean.empty and "store_name" in cal_clean.columns:
            cal_clean = cal_clean.sort_values("valid_from").head(50)
            fig_gantt = px.timeline(
                cal_clean,
                x_start="valid_from",
                x_end="valid_until",
                y="store_name",
                color="category_l1" if "category_l1" in cal_clean.columns else "store_name",
                hover_name="name" if "name" in cal_clean.columns else None,
                title="Aktionszeiträume (Top-50 nach Ladedatum)",
                labels={"store_name": "Händler", "category_l1": "Kategorie"},
            )
            fig_gantt.update_layout(margin=dict(t=50, b=10))
            st.plotly_chart(fig_gantt, use_container_width=True)
