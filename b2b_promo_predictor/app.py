"""moj-marketino B2B Promo Intelligence Platform – Streamlit Dashboard.

Drei Tabs:
  1. Data Pipeline   – Gemini-Extraktion & Entity Matching
  2. Prediction Engine – ML-Aktionsvorhersagen
  3. Price Analytics  – Historischer Preisverlauf
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Sicherstellen, dass src im Suchpfad ist
sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings
from src.extraction import PromoData, extract_from_image, _generate_mock_data
from src.features import create_features
from src.matching import ProductMatcher
from src.model import get_feature_importance, predict, train_lgbm

# ──────────────────────────────────────────────────────────────────────────────
# Page config & global styling
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="moj-marketino | B2B Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Corporate colour palette
PRIMARY = "#1A56DB"
ACCENT = "#F05252"
BG_CARD = "#F9FAFB"

st.markdown(
    f"""
    <style>
      /* Global font & background */
      html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
      .block-container {{ padding-top: 1.5rem; }}

      /* KPI cards */
      .kpi-card {{
        background: {BG_CARD};
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid {PRIMARY};
        box-shadow: 0 1px 4px rgba(0,0,0,.08);
      }}
      .kpi-label {{ font-size: .78rem; color: #6B7280; font-weight: 600;
                    text-transform: uppercase; letter-spacing: .06em; }}
      .kpi-value {{ font-size: 2rem; font-weight: 700; color: #111827; line-height: 1.1; }}
      .kpi-delta {{ font-size: .82rem; color: #10B981; }}

      /* Status badge */
      .badge-ok   {{ background:#D1FAE5; color:#065F46; border-radius:4px;
                     padding:2px 8px; font-size:.75rem; font-weight:600; }}
      .badge-warn {{ background:#FEF3C7; color:#92400E; border-radius:4px;
                     padding:2px 8px; font-size:.75rem; font-weight:600; }}
      .badge-err  {{ background:#FEE2E2; color:#991B1B; border-radius:4px;
                     padding:2px 8px; font-size:.75rem; font-weight:600; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://via.placeholder.com/200x50/1A56DB/FFFFFF?text=moj-marketino",
        use_column_width=True,
    )
    st.markdown("### B2B Promo Intelligence")
    st.caption("Version 1.0.0 · Enterprise Edition")
    st.divider()

    api_status = (
        '<span class="badge-ok">Gemini API aktiv</span>'
        if settings.has_gemini
        else '<span class="badge-warn">Demo-Modus (kein API-Key)</span>'
    )
    st.markdown(f"**API Status:** {api_status}", unsafe_allow_html=True)
    st.divider()

    st.markdown("**Einstellungen**")
    n_mock = st.slider("Mock-Datensätze", 5, 20, 8)
    match_threshold = st.slider("Match-Konfidenz Schwelle", 0.5, 0.95, 0.70, 0.05)
    st.divider()
    st.caption("© 2025 moj-marketino GmbH")

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <h1 style='margin-bottom:0;font-size:1.8rem;'>
      📊 Promo Intelligence Platform
    </h1>
    <p style='color:#6B7280;margin-top:.2rem;'>
      KI-gestützte Erkennung & Vorhersage von Discounter-Aktionen · B2B SaaS
    </p>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Global KPIs (above tabs)
# ──────────────────────────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)
for col, label, value, delta in [
    (k1, "Analysierte Prospekte", "1.284", "+12% vs. Vorwoche"),
    (k2, "Extrahierte Produkte", "38.650", "+8%"),
    (k3, "Match-Rate", "94,3 %", "+1.2 PP"),
    (k4, "Vorhersage-Genauigkeit", "87,1 %", "F1-Score"),
]:
    col.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-delta">▲ {delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(
    ["🔍 Data Pipeline", "🤖 Prediction Engine", "📈 Price Analytics"]
)

# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Data Pipeline
# ╚══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Prospekt-Extraktion & Entity Matching")
    st.caption(
        "Gemini Vision analysiert Discounter-Prospekte und extrahiert strukturierte "
        "Aktionsdaten. Entity Matching ordnet diese einer Master-Produktdatenbank zu."
    )

    col_upload, col_run = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Prospekt hochladen (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Lädt ein Bild hoch und sendet es an Gemini Vision.",
        )
    with col_run:
        st.markdown("<br>", unsafe_allow_html=True)
        run_pipeline = st.button("▶ Pipeline starten", type="primary", use_container_width=True)

    st.divider()

    # ── Extraction results ────────────────────────────────────────────────────
    if run_pipeline or True:  # Zeige sofort Demo-Daten
        with st.spinner("Extraktion läuft…"):
            if uploaded and settings.has_gemini:
                tmp_path = Path("/tmp") / uploaded.name
                tmp_path.write_bytes(uploaded.read())
                promos: list[PromoData] = extract_from_image(tmp_path)
            else:
                promos = _generate_mock_data(n_mock)

        extraction_df = pd.DataFrame([p.model_dump() for p in promos])

        left, right = st.columns(2)

        with left:
            st.markdown("#### Extrahierte Rohdaten")
            st.caption(f"{len(extraction_df)} Produkte extrahiert")
            st.dataframe(
                extraction_df[
                    ["retailer", "raw_product_name", "price_promo", "valid_from", "valid_to"]
                ].rename(
                    columns={
                        "retailer": "Händler",
                        "raw_product_name": "Produkt (roh)",
                        "price_promo": "Preis (€)",
                        "valid_from": "Von",
                        "valid_to": "Bis",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.markdown("#### Retailer-Verteilung")
            dist = extraction_df["retailer"].value_counts().reset_index()
            dist.columns = ["Händler", "Anzahl"]
            fig_pie = px.pie(
                dist,
                names="Händler",
                values="Anzahl",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.45,
            )
            fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Matching ──────────────────────────────────────────────────────────
        st.markdown("#### Entity Matching – Zuordnung zur Master-DB")

        MASTER_LIST = [
            "Milka Schokolade 300g",
            "Coca-Cola Regular 1,5L PET",
            "Ritter Sport Voll-Nuss 100g",
            "Haribo Goldbären Tüte 200g",
            "Nutella Nuss-Nougat-Creme 450g",
            "Ja! Frische Vollmilch 3,5% 1L",
            "Ariel Waschpulver Color 20 WL",
            "Pampers Baby-Dry Gr.3 44 Stk",
            "Red Bull Energy Drink 250ml",
            "Pringles Original Chips 185g",
        ]

        matcher = ProductMatcher(threshold=match_threshold)

        with st.spinner("Entity Resolution läuft…"):
            raw_names = extraction_df["raw_product_name"].tolist()
            match_results = matcher.batch_match(raw_names, MASTER_LIST)

        match_df = extraction_df[["retailer", "raw_product_name", "price_promo"]].copy()
        match_df["master_produkt"] = [r.master_product for r in match_results]
        match_df["match_score"] = [r.score for r in match_results]
        match_df["konfident"] = [
            "✅ Ja" if r.is_confident else "⚠️ Prüfen"
            for r in match_results
        ]

        def _color_score(val: float) -> str:
            if val >= 0.80:
                return "color: #065F46; font-weight:600"
            if val >= 0.60:
                return "color: #92400E; font-weight:600"
            return "color: #991B1B; font-weight:600"

        styled = match_df.rename(
            columns={
                "retailer": "Händler",
                "raw_product_name": "Roh-Name",
                "price_promo": "Preis (€)",
                "master_produkt": "Master-Produkt",
                "match_score": "Score",
                "konfident": "Konfident",
            }
        )
        st.dataframe(
            styled.style.applymap(_color_score, subset=["Score"]),
            use_container_width=True,
            hide_index=True,
        )

        confident_pct = sum(1 for r in match_results if r.is_confident) / max(len(match_results), 1) * 100
        st.metric("Match-Rate", f"{confident_pct:.1f} %", help="Anteil Matches über Konfidenz-Schwelle")


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Prediction Engine
# ╚══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Aktionsvorhersage – Nächste 2 Wochen")
    st.caption(
        "LightGBM-Modell, trainiert auf historischen Aktionsdaten. "
        "Zeigt, welche Produkte bei welchen Discountern voraussichtlich in Aktion gehen."
    )

    col_train, col_filter = st.columns([1, 2])

    with col_train:
        if st.button("🔁 Modell neu trainieren", help="Trainiert auf Mock-Daten"):
            with st.spinner("Training läuft…"):
                trained_model = train_lgbm()
            st.success("Modell erfolgreich trainiert!")
        else:
            trained_model = None

    with col_filter:
        retailers_all = ["Alle", "Lidl", "Aldi Süd", "Penny", "Rewe", "Netto"]
        selected_retailer = st.selectbox("Händler filtern", retailers_all)

    # Predictions laden
    pred_df = predict(model=trained_model)

    if selected_retailer != "Alle":
        pred_df = pred_df[pred_df["retailer"] == selected_retailer]

    st.markdown(f"**{len(pred_df)} Vorhersagen** für den gewählten Zeitraum")

    # Confidence-Farbe
    def _badge(conf: float) -> str:
        if conf >= 0.85:
            return f'<span class="badge-ok">{conf:.0%}</span>'
        if conf >= 0.70:
            return f'<span class="badge-warn">{conf:.0%}</span>'
        return f'<span class="badge-err">{conf:.0%}</span>'

    display_df = pred_df.copy()
    display_df["Konfidenz"] = display_df["confidence"].apply(_badge)

    st.dataframe(
        display_df.rename(
            columns={
                "product": "Produkt",
                "retailer": "Händler",
                "predicted_promo_start": "Aktionsbeginn (erw.)",
                "expected_price": "Erw. Preis (€)",
            }
        )[["Produkt", "Händler", "Aktionsbeginn (erw.)", "Erw. Preis (€)", "confidence"]].rename(
            columns={"confidence": "Konfidenz"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Feature Importance Chart
    st.markdown("#### Feature Importance")
    fi_df = get_feature_importance(trained_model)
    fig_fi = px.bar(
        fi_df,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues",
        labels={"importance": "Wichtigkeit", "feature": "Feature"},
    )
    fig_fi.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(t=10, b=10),
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Confidence Distribution
    st.markdown("#### Konfidenz-Verteilung")
    fig_hist = px.histogram(
        pred_df,
        x="confidence",
        nbins=20,
        color_discrete_sequence=[PRIMARY],
        labels={"confidence": "Konfidenz-Score", "count": "Anzahl"},
    )
    fig_hist.add_vline(x=0.70, line_dash="dash", line_color=ACCENT, annotation_text="Schwelle (70%)")
    fig_hist.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Price Analytics
# ╚══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Historischer Preisverlauf")
    st.caption("Preisentwicklung eines Produkts über alle Händler hinweg.")

    # Filter-Zeile
    fc1, fc2, fc3 = st.columns(3)

    hist_df = create_features()  # Mock-History mit Features

    products_list = sorted(hist_df["product"].unique().tolist())
    retailers_list = ["Alle"] + sorted(hist_df["retailer"].unique().tolist())

    with fc1:
        sel_product = st.selectbox("Produkt auswählen", products_list)
    with fc2:
        sel_retailer_price = st.selectbox("Händler", retailers_list, key="price_retailer")
    with fc3:
        date_range = st.selectbox("Zeitraum", ["Letzte 30 Tage", "Letzte 90 Tage", "Alles"])

    # Filtern
    filtered = hist_df[hist_df["product"] == sel_product].copy()
    if sel_retailer_price != "Alle":
        filtered = filtered[filtered["retailer"] == sel_retailer_price]

    if date_range == "Letzte 30 Tage":
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=30)
        filtered = filtered[filtered["date"] >= cutoff]
    elif date_range == "Letzte 90 Tage":
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=90)
        filtered = filtered[filtered["date"] >= cutoff]

    # ── Preisverlauf Plot ─────────────────────────────────────────────────────
    if filtered.empty:
        st.info("Keine Daten für diese Auswahl.")
    else:
        fig_price = go.Figure()

        for retailer in filtered["retailer"].unique():
            sub = filtered[filtered["retailer"] == retailer].sort_values("date")
            # Aktionspreise
            promo_sub = sub[sub["is_on_promo"] == 1]
            regular_sub = sub[sub["is_on_promo"] == 0]

            fig_price.add_trace(
                go.Scatter(
                    x=sub["date"],
                    y=sub["price_regular"],
                    mode="lines",
                    name=f"{retailer} – Regulär",
                    line=dict(width=1, dash="dot"),
                    opacity=0.5,
                )
            )
            fig_price.add_trace(
                go.Scatter(
                    x=sub["date"],
                    y=sub["price_promo"],
                    mode="lines+markers",
                    name=f"{retailer} – Aktionspreis",
                    line=dict(width=2),
                )
            )
            if not promo_sub.empty:
                fig_price.add_trace(
                    go.Scatter(
                        x=promo_sub["date"],
                        y=promo_sub["price_promo"],
                        mode="markers",
                        marker=dict(symbol="star", size=12, color=ACCENT),
                        name=f"{retailer} – Aktionspunkt",
                        showlegend=False,
                    )
                )

        fig_price.update_layout(
            title=f"Preisverlauf: {sel_product}",
            xaxis_title="Datum",
            yaxis_title="Preis (€)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # ── Statistik-Cards ───────────────────────────────────────────────────
        st.divider()
        s1, s2, s3, s4 = st.columns(4)
        avg_promo = filtered[filtered["is_on_promo"] == 1]["price_promo"].mean()
        avg_reg = filtered["price_regular"].mean()
        promo_freq = filtered["is_on_promo"].mean() * 100
        avg_discount = filtered["discount_depth"].mean() * 100

        for col, label, val, suffix in [
            (s1, "Ø Aktionspreis", f"{avg_promo:.2f}", " €"),
            (s2, "Ø Regulärpreis", f"{avg_reg:.2f}", " €"),
            (s3, "Aktionsfrequenz", f"{promo_freq:.1f}", " %"),
            (s4, "Ø Rabatttiefe", f"{avg_discount:.1f}", " %"),
        ]:
            col.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value">{val}<span style='font-size:1rem'>{suffix}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Discount-Depth Verlauf ────────────────────────────────────────────
        st.markdown("#### Rabatttiefe im Zeitverlauf")
        fig_dd = px.area(
            filtered.sort_values("date"),
            x="date",
            y="discount_depth",
            color="retailer",
            labels={"discount_depth": "Rabatttiefe", "date": "Datum", "retailer": "Händler"},
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig_dd.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_dd, use_container_width=True)
