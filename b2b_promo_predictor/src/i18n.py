"""Internationalisation (i18n) for moj-marketino.

Supported languages: EN, HR, SL, BS, SR, MK, ME.
Default language is determined by market (country code).
"""

from __future__ import annotations

SUPPORTED_LANGS: list[str] = ["EN", "HR", "SL", "BS", "SR", "MK", "ME"]

DEFAULT_LANGUAGE_BY_MARKET: dict[str, str] = {
    "HR": "HR",
    "SI": "SL",
    "BA": "BS",
    "RS": "SR",
    "MK": "MK",
    "ME": "ME",
}

LANG_LABELS: dict[str, str] = {
    "EN": "English",
    "HR": "Hrvatski",
    "SL": "Slovenščina",
    "BS": "Bosanski",
    "SR": "Srpski",
    "MK": "Македонски",
    "ME": "Crnogorski",
}

# ── Category translations ──────────────────────────────────────────────────────

CATEGORY_TRANSLATIONS: dict[str, dict[str, str]] = {
    "Hrana": {
        "EN": "Food", "HR": "Hrana", "SL": "Hrana", "BS": "Hrana",
        "SR": "Hrana", "MK": "Храна", "ME": "Hrana",
    },
    "Food": {
        "EN": "Food", "HR": "Hrana", "SL": "Hrana", "BS": "Hrana",
        "SR": "Hrana", "MK": "Храна", "ME": "Hrana",
    },
    "Piće": {
        "EN": "Beverages", "HR": "Piće", "SL": "Pijače", "BS": "Piće",
        "SR": "Piće", "MK": "Пијалоци", "ME": "Piće",
    },
    "Pice": {
        "EN": "Beverages", "HR": "Piće", "SL": "Pijače", "BS": "Piće",
        "SR": "Piće", "MK": "Пијалоци", "ME": "Piće",
    },
    "Kozmetika": {
        "EN": "Cosmetics & Care", "HR": "Kozmetika i njega", "SL": "Kozmetika in nega",
        "BS": "Kozmetika i njega", "SR": "Kozmetika i nega", "MK": "Козметика", "ME": "Kozmetika",
    },
    "Kućanska kemija": {
        "EN": "Household Chemicals", "HR": "Kućanska kemija", "SL": "Gospodinjska kemija",
        "BS": "Kućanska hemija", "SR": "Hemija za domaćinstvo", "MK": "Домаќинска хемија", "ME": "Kućna hemija",
    },
    "Kucanska kemija": {
        "EN": "Household Chemicals", "HR": "Kućanska kemija", "SL": "Gospodinjska kemija",
        "BS": "Kućanska hemija", "SR": "Hemija za domaćinstvo", "MK": "Домаќинска хемија", "ME": "Kućna hemija",
    },
    "Dječja hrana": {
        "EN": "Baby Food", "HR": "Dječja hrana", "SL": "Otroška hrana",
        "BS": "Dječija hrana", "SR": "Dječja hrana", "MK": "Детска храна", "ME": "Dječija hrana",
    },
    "Mliječni proizvodi": {
        "EN": "Dairy Products", "HR": "Mliječni proizvodi", "SL": "Mlečni izdelki",
        "BS": "Mliječni proizvodi", "SR": "Mlečni proizvodi", "MK": "Млечни производи", "ME": "Mliječni proizvodi",
    },
    "Mesni proizvodi": {
        "EN": "Meat & Deli", "HR": "Mesni proizvodi", "SL": "Mesni izdelki",
        "BS": "Mesni proizvodi", "SR": "Mesni proizvodi", "MK": "Месни производи", "ME": "Mesni proizvodi",
    },
    "Voće i povrće": {
        "EN": "Fruits & Vegetables", "HR": "Voće i povrće", "SL": "Sadje in zelenjava",
        "BS": "Voće i povrće", "SR": "Voće i povrće", "MK": "Овошје и зеленчук", "ME": "Voće i povrće",
    },
    "Zamrznuta hrana": {
        "EN": "Frozen Food", "HR": "Zamrznuta hrana", "SL": "Zamrznjena hrana",
        "BS": "Zamrznuta hrana", "SR": "Zamrznuta hrana", "MK": "Замрзната храна", "ME": "Zamrznuta hrana",
    },
    "Slatkiši": {
        "EN": "Confectionery", "HR": "Slatkiši", "SL": "Sladkarije",
        "BS": "Slatkiši", "SR": "Slatkiši", "MK": "Слатки", "ME": "Slatkiši",
    },
    "Pekarski proizvodi": {
        "EN": "Bakery", "HR": "Pekarski proizvodi", "SL": "Pekovski izdelki",
        "BS": "Pekarski proizvodi", "SR": "Pekarski proizvodi", "MK": "Пекарски производи", "ME": "Pekarski proizvodi",
    },
    "Zdravlje i ljepota": {
        "EN": "Health & Beauty", "HR": "Zdravlje i ljepota", "SL": "Zdravje in lepota",
        "BS": "Zdravlje i ljepota", "SR": "Zdravlje i lepota", "MK": "Здравје и убавина", "ME": "Zdravlje i ljepota",
    },
    "Čišćenje": {
        "EN": "Cleaning", "HR": "Čišćenje", "SL": "Čiščenje",
        "BS": "Čišćenje", "SR": "Čišćenje", "MK": "Чистење", "ME": "Čišćenje",
    },
    "Non-Food": {
        "EN": "Non-Food", "HR": "Non-Food", "SL": "Non-Food",
        "BS": "Non-Food", "SR": "Non-Food", "MK": "Non-Food", "ME": "Non-Food",
    },
    "Other": {
        "EN": "Other", "HR": "Ostalo", "SL": "Ostalo",
        "BS": "Ostalo", "SR": "Ostalo", "MK": "Друго", "ME": "Ostalo",
    },
    "Храна": {
        "EN": "Food", "HR": "Hrana", "SL": "Hrana", "BS": "Hrana",
        "SR": "Hrana", "MK": "Храна", "ME": "Hrana",
    },
    "Пијалоци": {
        "EN": "Beverages", "HR": "Piće", "SL": "Pijače", "BS": "Piće",
        "SR": "Piće", "MK": "Пијалоци", "ME": "Piće",
    },
}

# ── UI translations ────────────────────────────────────────────────────────────

TRANSLATIONS: dict[str, dict[str, str]] = {
    # ── App header ──
    "app.title": {
        "EN": "Promo Intelligence Platform",
        "HR": "Promo Intelligence Platforma",
        "SL": "Platforma za obveščanje o promocijah",
        "BS": "Platforma za praćenje promocija",
        "SR": "Platforma za praćenje promocija",
        "MK": "Платформа за промо разузнавање",
        "ME": "Platforma za praćenje promocija",
    },
    "app.subtitle": {
        "EN": "Smart promotion tracking for the Balkan market",
        "HR": "Pametno praćenje promotivnih cijena za balkansko tržište",
        "SL": "Pametno sledenje akcij za balkanski trg",
        "BS": "Pametno praćenje promotivnih cijena za balkansko tržište",
        "SR": "Pametno praćenje promotivnih cijena za balkansko tržište",
        "MK": "Паметно следење на промоции за балкански пазар",
        "ME": "Pametno praćenje promotivnih cijena za balkansko tržište",
    },
    # ── Sidebar ──
    "sidebar.filter": {
        "EN": "Filters", "HR": "Filtri", "SL": "Filtri",
        "BS": "Filtri", "SR": "Filtri", "MK": "Филтри", "ME": "Filtri",
    },
    "sidebar.market": {
        "EN": "Market / Country", "HR": "Tržište / Zemlja", "SL": "Trg / Država",
        "BS": "Tržište / Zemlja", "SR": "Tržište / Zemlja", "MK": "Пазар / Земја", "ME": "Tržište / Zemlja",
    },
    "sidebar.all_markets": {
        "EN": "All markets", "HR": "Sva tržišta", "SL": "Vsi trgi",
        "BS": "Sva tržišta", "SR": "Sva tržišta", "MK": "Сите пазари", "ME": "Sva tržišta",
    },
    "sidebar.category": {
        "EN": "Category", "HR": "Kategorija", "SL": "Kategorija",
        "BS": "Kategorija", "SR": "Kategorija", "MK": "Категорија", "ME": "Kategorija",
    },
    "sidebar.all_categories": {
        "EN": "All categories", "HR": "Sve kategorije", "SL": "Vse kategorije",
        "BS": "Sve kategorije", "SR": "Sve kategorije", "MK": "Сите категории", "ME": "Sve kategorije",
    },
    "sidebar.brand": {
        "EN": "Brand / Manufacturer", "HR": "Marka / Proizvođač", "SL": "Znamka / Proizvajalec",
        "BS": "Marka / Proizvođač", "SR": "Marka / Proizvođač", "MK": "Бренд / Производител", "ME": "Marka / Proizvođač",
    },
    "sidebar.language": {
        "EN": "Language", "HR": "Jezik", "SL": "Jezik",
        "BS": "Jezik", "SR": "Jezik", "MK": "Јазик", "ME": "Jezik",
    },
    "sidebar.demo_mode": {
        "EN": "DEMO MODE – no database connection", "HR": "DEMO NAČIN RADA – bez veze s bazom",
        "SL": "DEMO NAČIN – brez povezave z bazo", "BS": "DEMO NAČIN – bez veze s bazom",
        "SR": "DEMO NAČIN – bez veze s bazom", "MK": "ДЕМО РЕЖИМ – без врска со база",
        "ME": "DEMO NAČIN – bez veze sa bazom",
    },
    "sidebar.live_data": {
        "EN": "● Live data active", "HR": "● Live podaci aktivni", "SL": "● Živi podatki aktivni",
        "BS": "● Live podaci aktivni", "SR": "● Live podaci aktivni",
        "MK": "● Живи податоци активни", "ME": "● Live podaci aktivni",
    },
    # ── Tab labels ──
    "tab.predictor": {
        "EN": "🔮 Promotion Predictor", "HR": "🔮 Predviđanje akcija", "SL": "🔮 Napoved akcij",
        "BS": "🔮 Predviđanje akcija", "SR": "🔮 Predviđanje akcija",
        "MK": "🔮 Предвидување акции", "ME": "🔮 Predviđanje akcija",
    },
    "tab.price": {
        "EN": "📉 Price Analysis", "HR": "📉 Analiza cijena", "SL": "📉 Analiza cen",
        "BS": "📉 Analiza cijena", "SR": "📉 Analiza cena",
        "MK": "📉 Анализа на цени", "ME": "📉 Analiza cijena",
    },
    "tab.quality": {
        "EN": "⚙️ Data Quality", "HR": "⚙️ Kvaliteta podataka", "SL": "⚙️ Kakovost podatkov",
        "BS": "⚙️ Kvalitet podataka", "SR": "⚙️ Kvalitet podataka",
        "MK": "⚙️ Квалитет на податоци", "ME": "⚙️ Kvalitet podataka",
    },
    # ── KAM forecast section ──
    "kam.title": {
        "EN": "KAM Promotion Forecast", "HR": "KAM prognoza akcija", "SL": "KAM napoved akcij",
        "BS": "KAM prognoza akcija", "SR": "KAM prognoza akcija",
        "MK": "KAM прогноза на акции", "ME": "KAM prognoza akcija",
    },
    "kam.min_req": {
        "EN": "Min. {n} promotions and {d} days of history required per product–retailer pair.",
        "HR": "Potrebno min. {n} akcija i {d} dana povijesti po kombinaciji proizvod–prodavaonica.",
        "SL": "Potrebno min. {n} akcij in {d} dni zgodovine na kombinacijo izdelek–trgovec.",
        "BS": "Potrebno min. {n} akcija i {d} dana historije po kombinaciji proizvod–prodavnica.",
        "SR": "Potrebno min. {n} akcija i {d} dana istorije po kombinaciji proizvod–prodavnica.",
        "MK": "Потребни мин. {n} акции и {d} дена историја по комбинација производ–продавница.",
        "ME": "Potrebno min. {n} akcija i {d} dana istorije po kombinaciji proizvod–prodavnica.",
    },
    "kam.product_search": {
        "EN": "Product / search term", "HR": "Proizvod / pojam za pretragu", "SL": "Izdelek / iskalni izraz",
        "BS": "Proizvod / pojam za pretragu", "SR": "Proizvod / pojam za pretragu",
        "MK": "Производ / термин за пребарување", "ME": "Proizvod / pojam za pretragu",
    },
    "kam.retailer": {
        "EN": "Retailer", "HR": "Prodavaonica", "SL": "Trgovec",
        "BS": "Prodavnica", "SR": "Prodavnica", "MK": "Продавница", "ME": "Prodavnica",
    },
    "kam.all_retailers": {
        "EN": "All retailers", "HR": "Sve prodavaonice", "SL": "Vse trgovine",
        "BS": "Sve prodavnice", "SR": "Sve prodavnice", "MK": "Сите продавници", "ME": "Sve prodavnice",
    },
    "kam.min_probability": {
        "EN": "Min. probability", "HR": "Min. vjerojatnost", "SL": "Min. verjetnost",
        "BS": "Min. vjerovatnoća", "SR": "Min. verovatnoća", "MK": "Мин. веројатност", "ME": "Min. vjerovatnoća",
    },
    "kam.prediction_window": {
        "EN": "Forecast window (days)", "HR": "Prognozni period (dani)", "SL": "Napovedni horizont (dni)",
        "BS": "Period prognoze (dani)", "SR": "Period prognoze (dani)", "MK": "Период на прогноза (дена)", "ME": "Period prognoze (dani)",
    },
    "kam.signal": {
        "EN": "Signal", "HR": "Signal", "SL": "Signal",
        "BS": "Signal", "SR": "Signal", "MK": "Сигнал", "ME": "Signal",
    },
    "kam.all_signals": {
        "EN": "All", "HR": "Svi", "SL": "Vsi",
        "BS": "Svi", "SR": "Svi", "MK": "Сите", "ME": "Svi",
    },
    "kam.price_trend": {
        "EN": "Price trend", "HR": "Cjenovni trend", "SL": "Cenovni trend",
        "BS": "Cjenovni trend", "SR": "Cenovni trend", "MK": "Ценовен тренд", "ME": "Cjenovni trend",
    },
    "kam.only_future": {
        "EN": "Only future promotions", "HR": "Samo buduće akcije", "SL": "Samo prihodnje akcije",
        "BS": "Samo buduće akcije", "SR": "Samo buduće akcije", "MK": "Само идни акции", "ME": "Samo buduće akcije",
    },
    "kam.only_sufficient": {
        "EN": "Only products with sufficient history", "HR": "Samo proizvodi s dovoljno povijesti",
        "SL": "Samo izdelki z zadostno zgodovino", "BS": "Samo proizvodi s dovoljnom historijom",
        "SR": "Samo proizvodi sa dovoljno istorije", "MK": "Само производи со доволна историја",
        "ME": "Samo proizvodi sa dovoljno istorije",
    },
    # ── Signal labels ──
    "signal.high": {
        "EN": "🔴 High relevance", "HR": "🔴 Visoka relevantnost", "SL": "🔴 Visoka ustreznost",
        "BS": "🔴 Visoka relevantnost", "SR": "🔴 Visoka relevantnost",
        "MK": "🔴 Висока релевантност", "ME": "🔴 Visoka relevantnost",
    },
    "signal.watch": {
        "EN": "🟡 Watch", "HR": "🟡 Pratiti", "SL": "🟡 Opazovati",
        "BS": "🟡 Pratiti", "SR": "🟡 Pratiti", "MK": "🟡 Следи", "ME": "🟡 Pratiti",
    },
    "signal.normal": {
        "EN": "🟢 Normal", "HR": "🟢 Normalno", "SL": "🟢 Normalno",
        "BS": "🟢 Normalno", "SR": "🟢 Normalno", "MK": "🟢 Нормално", "ME": "🟢 Normalno",
    },
    "signal.unreliable": {
        "EN": "⚪ Not reliable", "HR": "⚪ Nije pouzdano", "SL": "⚪ Ni zanesljivo",
        "BS": "⚪ Nije pouzdano", "SR": "⚪ Nije pouzdano", "MK": "⚪ Не е веродостоен", "ME": "⚪ Nije pouzdano",
    },
    "signal.invalid": {
        "EN": "⚫ Invalid", "HR": "⚫ Nevažeće", "SL": "⚫ Neveljavno",
        "BS": "⚫ Nevažeće", "SR": "⚫ Nevažeće", "MK": "⚫ Невалидно", "ME": "⚫ Nevažeće",
    },
    # ── Trust levels ──
    "trust.reliable": {
        "EN": "✅ Reliable", "HR": "✅ Pouzdano", "SL": "✅ Zanesljivo",
        "BS": "✅ Pouzdano", "SR": "✅ Pouzdano", "MK": "✅ Веродостоен", "ME": "✅ Pouzdano",
    },
    "trust.limited": {
        "EN": "⚠️ Limited reliability", "HR": "⚠️ Ograničena pouzdanost", "SL": "⚠️ Omejena zanesljivost",
        "BS": "⚠️ Ograničena pouzdanost", "SR": "⚠️ Ograničena pouzdanost",
        "MK": "⚠️ Ограничена веродостојност", "ME": "⚠️ Ograničena pouzdanost",
    },
    "trust.unreliable": {
        "EN": "❌ Not reliable", "HR": "❌ Nije pouzdano", "SL": "❌ Ni zanesljivo",
        "BS": "❌ Nije pouzdano", "SR": "❌ Nije pouzdano", "MK": "❌ Не е веродостоен", "ME": "❌ Nije pouzdano",
    },
    # ── Price trend labels ──
    "trend.rising": {
        "EN": "rising", "HR": "raste", "SL": "narašča",
        "BS": "raste", "SR": "raste", "MK": "расте", "ME": "raste",
    },
    "trend.falling": {
        "EN": "falling", "HR": "pada", "SL": "pada",
        "BS": "pada", "SR": "pada", "MK": "паѓа", "ME": "pada",
    },
    "trend.stable": {
        "EN": "stable", "HR": "stabilno", "SL": "stabilno",
        "BS": "stabilno", "SR": "stabilno", "MK": "стабилно", "ME": "stabilno",
    },
    "trend.unknown": {
        "EN": "unknown", "HR": "nepoznato", "SL": "neznano",
        "BS": "nepoznato", "SR": "nepoznato", "MK": "непознато", "ME": "nepoznato",
    },
    # ── KPI cards ──
    "kpi.high_signal": {
        "EN": "High-relevance forecasts", "HR": "Visoko relevantne prognoze", "SL": "Visoko ustrezne napovedi",
        "BS": "Visoko relevantne prognoze", "SR": "Visoko relevantne prognoze",
        "MK": "Високо релевантни прогнози", "ME": "Visoko relevantne prognoze",
    },
    "kpi.rising_price": {
        "EN": "Rising price trend", "HR": "Rastuće cijene", "SL": "Naraščajoče cene",
        "BS": "Rastuće cijene", "SR": "Rastuće cene", "MK": "Растечки цени", "ME": "Rastuće cijene",
    },
    "kpi.overdue": {
        "EN": "Overdue promotions", "HR": "Zakašnjele akcije", "SL": "Zamujene akcije",
        "BS": "Zakašnjele akcije", "SR": "Zakašnjele akcije", "MK": "Задоцнети акции", "ME": "Zakašnjele akcije",
    },
    "kpi.forecast_base": {
        "EN": "Forecast base", "HR": "Osnova prognoza", "SL": "Osnova napovedi",
        "BS": "Osnova prognoza", "SR": "Osnova prognoza", "MK": "Основа за прогноза", "ME": "Osnova prognoza",
    },
    "kpi.avg_discount": {
        "EN": "Avg. expected discount", "HR": "Prosj. očekivani popust", "SL": "Povpr. pričakovani popust",
        "BS": "Prosj. očekivani popust", "SR": "Prosj. očekivani popust",
        "MK": "Просечен очекуван попуст", "ME": "Prosj. očekivani popust",
    },
    # ── Table column headers ──
    "col.priority": {
        "EN": "Priority", "HR": "Prioritet", "SL": "Prioriteta",
        "BS": "Prioritet", "SR": "Prioritet", "MK": "Приоритет", "ME": "Prioritet",
    },
    "col.product": {
        "EN": "Product", "HR": "Proizvod", "SL": "Izdelek",
        "BS": "Proizvod", "SR": "Proizvod", "MK": "Производ", "ME": "Proizvod",
    },
    "col.brand": {
        "EN": "Brand", "HR": "Marka", "SL": "Znamka",
        "BS": "Marka", "SR": "Marka", "MK": "Бренд", "ME": "Marka",
    },
    "col.retailer": {
        "EN": "Retailer", "HR": "Prodavaonica", "SL": "Trgovec",
        "BS": "Prodavnica", "SR": "Prodavnica", "MK": "Продавница", "ME": "Prodavnica",
    },
    "col.market": {
        "EN": "Market", "HR": "Tržište", "SL": "Trg",
        "BS": "Tržište", "SR": "Tržište", "MK": "Пазар", "ME": "Tržište",
    },
    "col.category": {
        "EN": "Category", "HR": "Kategorija", "SL": "Kategorija",
        "BS": "Kategorija", "SR": "Kategorija", "MK": "Категорија", "ME": "Kategorija",
    },
    "col.period": {
        "EN": "Promotion period", "HR": "Period akcije", "SL": "Obdobje akcije",
        "BS": "Period akcije", "SR": "Period akcije", "MK": "Период на акција", "ME": "Period akcije",
    },
    "col.probability": {
        "EN": "Probability", "HR": "Vjerojatnost", "SL": "Verjetnost",
        "BS": "Vjerovatnoća", "SR": "Verovatnoća", "MK": "Веројатност", "ME": "Vjerovatnoća",
    },
    "col.expected_price": {
        "EN": "Exp. price", "HR": "Očekivana cijena", "SL": "Pričakovana cena",
        "BS": "Očekivana cijena", "SR": "Očekivana cena", "MK": "Очекувана цена", "ME": "Očekivana cijena",
    },
    "col.last_price": {
        "EN": "Last promo price", "HR": "Zadnja promo cijena", "SL": "Zadnja promo cena",
        "BS": "Zadnja promo cijena", "SR": "Zadnja promo cena", "MK": "Последна промо цена", "ME": "Zadnja promo cijena",
    },
    "col.cycle": {
        "EN": "Avg. cycle", "HR": "Prosj. ciklus", "SL": "Povpr. cikel",
        "BS": "Prosj. ciklus", "SR": "Prosj. ciklus", "MK": "Просечен циклус", "ME": "Prosj. ciklus",
    },
    "col.last_promo": {
        "EN": "Last promotion", "HR": "Zadnja akcija", "SL": "Zadnja akcija",
        "BS": "Zadnja akcija", "SR": "Zadnja akcija", "MK": "Последна акција", "ME": "Zadnja akcija",
    },
    "col.history_count": {
        "EN": "Hist. promotions", "HR": "Promo u historiji", "SL": "Promo v zgodovini",
        "BS": "Promo u historiji", "SR": "Promo u istoriji", "MK": "Прomo во историјата", "ME": "Promo u istoriji",
    },
    "col.justification": {
        "EN": "Justification", "HR": "Objašnjenje", "SL": "Utemeljitev",
        "BS": "Objašnjenje", "SR": "Objašnjenje", "MK": "Образложение", "ME": "Objašnjenje",
    },
    # ── Upcoming promotions ──
    "upcoming.title": {
        "EN": "Upcoming promotions", "HR": "Nadolazeće akcije", "SL": "Prihodnje akcije",
        "BS": "Nadolazeće akcije", "SR": "Nadolazeće akcije", "MK": "Претстојни акции", "ME": "Nadolazeće akcije",
    },
    # ── Data quality ──
    "dq.complete": {
        "EN": "✓ Complete", "HR": "✓ Potpuno", "SL": "✓ Popolno",
        "BS": "✓ Potpuno", "SR": "✓ Potpuno", "MK": "✓ Комплетно", "ME": "✓ Potpuno",
    },
    "dq.brand_missing": {
        "EN": "Brand missing", "HR": "Marka nedostaje", "SL": "Znamka manjka",
        "BS": "Marka nedostaje", "SR": "Marka nedostaje", "MK": "Бренд недостасува", "ME": "Marka nedostaje",
    },
    "dq.orig_price_missing": {
        "EN": "Regular price missing", "HR": "Regularna cijena nedostaje", "SL": "Redna cena manjka",
        "BS": "Regularna cijena nedostaje", "SR": "Regularna cena nedostaje",
        "MK": "Редовна цена недостасува", "ME": "Regularna cijena nedostaje",
    },
    "dq.no_real_discount": {
        "EN": "No real discount", "HR": "Bez pravog popusta", "SL": "Brez pravega popusta",
        "BS": "Bez pravog popusta", "SR": "Bez pravog popusta", "MK": "Без вистински попуст", "ME": "Bez pravog popusta",
    },
    "dq.category_uncertain": {
        "EN": "Category uncertain", "HR": "Kategorija nesigurna", "SL": "Kategorija negotova",
        "BS": "Kategorija nesigurna", "SR": "Kategorija nesigurna", "MK": "Категорија несигурна", "ME": "Kategorija nesigurna",
    },
    # ── General ──
    "general.days": {
        "EN": "days", "HR": "dana", "SL": "dni",
        "BS": "dana", "SR": "dana", "MK": "дена", "ME": "dana",
    },
    "general.days_ago": {
        "EN": "{d} days ago", "HR": "prije {d} dana", "SL": "pred {d} dnevi",
        "BS": "prije {d} dana", "SR": "prije {d} dana", "MK": "пред {d} дена", "ME": "prije {d} dana",
    },
    "general.median": {
        "EN": "median", "HR": "medijan", "SL": "mediana",
        "BS": "medijan", "SR": "medijan", "MK": "медијана", "ME": "medijan",
    },
    "general.no_data": {
        "EN": "No data", "HR": "Nema podataka", "SL": "Ni podatkov",
        "BS": "Nema podataka", "SR": "Nema podataka", "MK": "Нема податоци", "ME": "Nema podataka",
    },
    "general.loading": {
        "EN": "Loading…", "HR": "Učitavanje…", "SL": "Nalaganje…",
        "BS": "Učitavanje…", "SR": "Učitavanje…", "MK": "Вчитување…", "ME": "Učitavanje…",
    },
    # ── Prediction status ──
    "status.ok": {
        "EN": "OK", "HR": "OK", "SL": "OK",
        "BS": "OK", "SR": "OK", "MK": "ОК", "ME": "OK",
    },
    "status.insufficient_history": {
        "EN": "Insufficient history", "HR": "Nedovoljna historija", "SL": "Nezadostna zgodovina",
        "BS": "Nedovoljna historija", "SR": "Nedovoljna istorija",
        "MK": "Недоволна историја", "ME": "Nedovoljna istorija",
    },
    "status.no_data": {
        "EN": "No data", "HR": "Nema podataka", "SL": "Ni podatkov",
        "BS": "Nema podataka", "SR": "Nema podataka", "MK": "Нема податоци", "ME": "Nema podataka",
    },
    # ── Country names in local language ──
    "country.HR": {
        "EN": "Croatia 🇭🇷", "HR": "Hrvatska 🇭🇷", "SL": "Hrvaška 🇭🇷",
        "BS": "Hrvatska 🇭🇷", "SR": "Hrvatska 🇭🇷", "MK": "Хрватска 🇭🇷", "ME": "Hrvatska 🇭🇷",
    },
    "country.SI": {
        "EN": "Slovenia 🇸🇮", "HR": "Slovenija 🇸🇮", "SL": "Slovenija 🇸🇮",
        "BS": "Slovenija 🇸🇮", "SR": "Slovenija 🇸🇮", "MK": "Словенија 🇸🇮", "ME": "Slovenija 🇸🇮",
    },
    "country.BA": {
        "EN": "Bosnia & Herzegovina 🇧🇦", "HR": "Bosna i Hercegovina 🇧🇦", "SL": "Bosna in Hercegovina 🇧🇦",
        "BS": "Bosna i Hercegovina 🇧🇦", "SR": "Bosna i Hercegovina 🇧🇦",
        "MK": "Босна и Херцеговина 🇧🇦", "ME": "Bosna i Hercegovina 🇧🇦",
    },
    "country.RS": {
        "EN": "Serbia 🇷🇸", "HR": "Srbija 🇷🇸", "SL": "Srbija 🇷🇸",
        "BS": "Srbija 🇷🇸", "SR": "Srbija 🇷🇸", "MK": "Србија 🇷🇸", "ME": "Srbija 🇷🇸",
    },
    "country.MK": {
        "EN": "North Macedonia 🇲🇰", "HR": "Sjeverna Makedonija 🇲🇰", "SL": "Severna Makedonija 🇲🇰",
        "BS": "Sjeverna Makedonija 🇲🇰", "SR": "Severna Makedonija 🇲🇰",
        "MK": "Македонија 🇲🇰", "ME": "Sjeverna Makedonija 🇲🇰",
    },
    "country.ME": {
        "EN": "Montenegro 🇲🇪", "HR": "Crna Gora 🇲🇪", "SL": "Črna gora 🇲🇪",
        "BS": "Crna Gora 🇲🇪", "SR": "Crna Gora 🇲🇪", "MK": "Црна Гора 🇲🇪", "ME": "Crna Gora 🇲🇪",
    },
}

# ── Currency by market ─────────────────────────────────────────────────────────

MARKET_CURRENCY: dict[str, str] = {
    "HR": "EUR",
    "SI": "EUR",
    "ME": "EUR",
    "BA": "BAM",
    "RS": "RSD",
    "MK": "MKD",
}

CURRENCY_SYMBOLS: dict[str, str] = {
    "EUR": "€",
    "BAM": "KM",
    "RSD": "din",
    "MKD": "ден",
}


# ── Translation functions ──────────────────────────────────────────────────────

def t(key: str, lang: str = "EN", **kwargs) -> str:
    """Translate a UI key to the requested language, falling back to EN.

    Args:
        key: Translation key (e.g. "sidebar.market").
        lang: Two-letter language code (EN/HR/SL/BS/SR/MK/ME).
        **kwargs: Optional format arguments substituted into the string.

    Returns:
        Localised string, or the key itself if not found.
    """
    lang = lang.upper()
    if lang not in SUPPORTED_LANGS:
        lang = "EN"

    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key

    text = entry.get(lang) or entry.get("EN") or key
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text


def translate_category(category: str, lang: str = "EN") -> str:
    """Translate a category name to the requested language.

    Falls back to original name if not found.
    """
    lang = lang.upper()
    if lang not in SUPPORTED_LANGS:
        lang = "EN"
    entry = CATEGORY_TRANSLATIONS.get(category)
    if entry is None:
        return category
    return entry.get(lang) or entry.get("EN") or category


def format_price(amount: float | None, currency: str = "EUR") -> str:
    """Format a price with the correct currency symbol for the market."""
    if amount is None:
        return "—"
    symbol = CURRENCY_SYMBOLS.get(currency, currency)
    if currency in ("BAM",):
        return f"{amount:.2f} {symbol}"
    if currency == "EUR":
        return f"{amount:.2f} {symbol}"
    if currency in ("RSD", "MKD"):
        return f"{amount:.0f} {symbol}"
    return f"{amount:.2f} {currency}"


def get_market_currency(country_code: str | None) -> str:
    """Return the currency code for a given market (country code)."""
    if not country_code:
        return "EUR"
    return MARKET_CURRENCY.get(country_code.upper(), "EUR")
