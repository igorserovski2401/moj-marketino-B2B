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
    "kam.no_history": {
        "EN": "🔍 **No product–retailer combination has sufficient history for a reliable forecast.**\nRequirement: at least {n} historical promotions and {d} days of history. Expand the market filter or wait for more data points.",
        "HR": "🔍 **Nijedna kombinacija proizvod–prodavaonica nema dovoljno povijesti za pouzdanu prognozu.**\nUvjet: min. {n} historijskih akcija i {d} dana povijesti. Proširi filter tržišta ili pričekaj više podataka.",
        "SL": "🔍 **Nobena kombinacija izdelek–trgovec nima zadostne zgodovine za zanesljivo napoved.**\nPogoj: min. {n} zgodovinskih akcij in {d} dni zgodovine. Razširi filter trga ali počakaj na več podatkov.",
        "BS": "🔍 **Nijedna kombinacija proizvod–prodavnica nema dovoljnu historiju za pouzdanu prognozu.**\nUvjet: min. {n} historijskih akcija i {d} dana historije. Proširi filter tržišta ili sačekaj više podataka.",
        "SR": "🔍 **Nijedna kombinacija proizvod–prodavnica nema dovoljnu istoriju za pouzdanu prognozu.**\nUslov: min. {n} istorijskih akcija i {d} dana istorije. Proširi filter tržišta ili sačekaj više podataka.",
        "MK": "🔍 **Ниедна комбинација производ–продавница нема доволна историја за сигурна прогноза.**\nУслов: мин. {n} историски акции и {d} дена историја. Прошири го филтерот за пазар или почекај повеќе податоци.",
        "ME": "🔍 **Nijedna kombinacija proizvod–prodavnica nema dovoljno istorije za pouzdanu prognozu.**\nUslov: min. {n} istorijskih akcija i {d} dana istorije. Proširi filter tržišta ili sačekaj više podataka.",
    },
    "kam.no_filter_match": {
        "EN": "No forecasts match the active filters. Reset or widen the filters.",
        "HR": "Nijedna prognoza ne odgovara aktivnim filterima. Poništi ili proširi filtere.",
        "SL": "Nobena napoved ne ustreza aktivnim filtrom. Ponastavi ali razširi filtre.",
        "BS": "Nijedna prognoza ne odgovara aktivnim filterima. Poništi ili proširi filtere.",
        "SR": "Nijedna prognoza ne odgovara aktivnim filterima. Poništi ili proširi filtere.",
        "MK": "Ниедна прогноза не одговара на активните филтри. Ресетирај ги или прошири ги филтрите.",
        "ME": "Nijedna prognoza ne odgovara aktivnim filterima. Poništi ili proširi filtere.",
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
    # ── Forecast tab (price trend / Prophet) ──
    "forecast.tab_title": {
        "EN": "Price trend forecast", "HR": "Prognoza trenda cijene",
        "SL": "Napoved cenovnega trenda", "BS": "Prognoza trenda cijene",
        "SR": "Prognoza trenda cene", "MK": "Прогноза на ценовен тренд",
        "ME": "Prognoza trenda cijene",
    },
    "forecast.section_select": {
        "EN": "Forecast selection (all required)", "HR": "Odabir za prognozu (sve obvezno)",
        "SL": "Izbor za napoved (vse obvezno)", "BS": "Odabir za prognozu (sve obavezno)",
        "SR": "Odabir za prognozu (sve obavezno)", "MK": "Избор за прогноза (сите задолжителни)",
        "ME": "Odabir za prognozu (sve obavezno)",
    },
    "forecast.product": {
        "EN": "Product", "HR": "Proizvod", "SL": "Izdelek",
        "BS": "Proizvod", "SR": "Proizvod", "MK": "Производ", "ME": "Proizvod",
    },
    "forecast.retailer": {
        "EN": "Retailer", "HR": "Prodavaonica", "SL": "Trgovec",
        "BS": "Prodavnica", "SR": "Prodavnica", "MK": "Продавница", "ME": "Prodavnica",
    },
    "forecast.market": {
        "EN": "Market", "HR": "Tržište", "SL": "Trg",
        "BS": "Tržište", "SR": "Tržište", "MK": "Пазар", "ME": "Tržište",
    },
    "forecast.currency": {
        "EN": "Currency", "HR": "Valuta", "SL": "Valuta",
        "BS": "Valuta", "SR": "Valuta", "MK": "Валута", "ME": "Valuta",
    },
    "forecast.horizon": {
        "EN": "Forecast horizon (days)", "HR": "Horizont prognoze (dani)",
        "SL": "Horizont napovedi (dni)", "BS": "Horizont prognoze (dani)",
        "SR": "Horizont prognoze (dani)", "MK": "Хоризонт на прогноза (дена)",
        "ME": "Horizont prognoze (dani)",
    },
    "forecast.run": {
        "EN": "▶ Run forecast", "HR": "▶ Pokreni prognozu", "SL": "▶ Zaženi napoved",
        "BS": "▶ Pokreni prognozu", "SR": "▶ Pokreni prognozu",
        "MK": "▶ Стартувај прогноза", "ME": "▶ Pokreni prognozu",
    },
    "forecast.missing_selection": {
        "EN": "Select product, retailer and market to run a reliable forecast.",
        "HR": "Odaberite proizvod, prodavaonicu i tržište za pouzdanu prognozu.",
        "SL": "Izberite izdelek, trgovca in trg za zanesljivo napoved.",
        "BS": "Odaberite proizvod, prodavnicu i tržište za pouzdanu prognozu.",
        "SR": "Odaberite proizvod, prodavnicu i tržište za pouzdanu prognozu.",
        "MK": "Изберете производ, продавница и пазар за веродостојна прогноза.",
        "ME": "Odaberite proizvod, prodavnicu i tržište za pouzdanu prognozu.",
    },
    "forecast.no_data_columns": {
        "EN": "No price/date columns in data – forecast not possible.",
        "HR": "Nedostaju stupci cijena/datuma – prognoza nije moguća.",
        "SL": "Manjkajo stolpci cene/datuma – napoved ni mogoča.",
        "BS": "Nedostaju kolone cijena/datuma – prognoza nije moguća.",
        "SR": "Nedostaju kolone cena/datuma – prognoza nije moguća.",
        "MK": "Недостасуваат колони цена/датум – прогнозата не е можна.",
        "ME": "Nedostaju kolone cijena/datuma – prognoza nije moguća.",
    },
    "forecast.too_few_points": {
        "EN": "Only {n} valid observations – forecast blocked (need ≥ 12).",
        "HR": "Samo {n} valjanih opažanja – prognoza blokirana (potrebno ≥ 12).",
        "SL": "Le {n} veljavnih opazovanj – napoved blokirana (potrebno ≥ 12).",
        "BS": "Samo {n} valjanih opservacija – prognoza blokirana (potrebno ≥ 12).",
        "SR": "Samo {n} validnih opservacija – prognoza blokirana (potrebno ≥ 12).",
        "MK": "Само {n} валидни опсервации – прогнозата е блокирана (потребни ≥ 12).",
        "ME": "Samo {n} validnih opservacija – prognoza blokirana (potrebno ≥ 12).",
    },
    "forecast.prophet_failed": {
        "EN": "Forecast model failed – not enough reliable history.",
        "HR": "Model prognoze nije uspio – premalo pouzdane povijesti.",
        "SL": "Model napovedi je odpovedal – premalo zanesljive zgodovine.",
        "BS": "Model prognoze nije uspio – premalo pouzdane historije.",
        "SR": "Model prognoze nije uspeo – premalo pouzdane istorije.",
        "MK": "Моделот на прогноза не успеа – недоволна веродостојна историја.",
        "ME": "Model prognoze nije uspio – premalo pouzdane istorije.",
    },
    "forecast.confidence_band": {
        "EN": "80% confidence band", "HR": "80% pojas pouzdanosti",
        "SL": "80% pas zaupanja", "BS": "80% pojas pouzdanosti",
        "SR": "80% pojas pouzdanosti", "MK": "80% појас на доверба",
        "ME": "80% pojas pouzdanosti",
    },
    "forecast.prophet_forecast": {
        "EN": "Forecast", "HR": "Prognoza", "SL": "Napoved",
        "BS": "Prognoza", "SR": "Prognoza", "MK": "Прогноза", "ME": "Prognoza",
    },
    "forecast.historical_price": {
        "EN": "Historical price", "HR": "Povijesna cijena", "SL": "Zgodovinska cena",
        "BS": "Historijska cijena", "SR": "Istorijska cena",
        "MK": "Историска цена", "ME": "Istorijska cijena",
    },
    "forecast.today": {
        "EN": "Today", "HR": "Danas", "SL": "Danes",
        "BS": "Danas", "SR": "Danas", "MK": "Денес", "ME": "Danas",
    },
    "forecast.forecast_label": {
        "EN": "Forecast", "HR": "Prognoza", "SL": "Napoved",
        "BS": "Prognoza", "SR": "Prognoza", "MK": "Прогноза", "ME": "Prognoza",
    },
    "forecast.price_label": {
        "EN": "Price", "HR": "Cijena", "SL": "Cena",
        "BS": "Cijena", "SR": "Cena", "MK": "Цена", "ME": "Cijena",
    },
    "forecast.axis_date": {
        "EN": "Date", "HR": "Datum", "SL": "Datum",
        "BS": "Datum", "SR": "Datum", "MK": "Датум", "ME": "Datum",
    },
    "forecast.axis_price": {
        "EN": "Price ({currency})", "HR": "Cijena ({currency})",
        "SL": "Cena ({currency})", "BS": "Cijena ({currency})",
        "SR": "Cena ({currency})", "MK": "Цена ({currency})",
        "ME": "Cijena ({currency})",
    },
    "forecast.chart_title": {
        "EN": "Price forecast: {product} (next {days} days)",
        "HR": "Prognoza cijene: {product} (sljedećih {days} dana)",
        "SL": "Napoved cene: {product} (naslednjih {days} dni)",
        "BS": "Prognoza cijene: {product} (sljedećih {days} dana)",
        "SR": "Prognoza cene: {product} (sledećih {days} dana)",
        "MK": "Прогноза на цена: {product} (следните {days} дена)",
        "ME": "Prognoza cijene: {product} (sljedećih {days} dana)",
    },
    "forecast.lowest_drop": {
        "EN": "⬇ Lowest price<br>{val:.2f} {sym}<br>−{pct:.1f}% vs current",
        "HR": "⬇ Najniža cijena<br>{val:.2f} {sym}<br>−{pct:.1f}% vs trenutno",
        "SL": "⬇ Najnižja cena<br>{val:.2f} {sym}<br>−{pct:.1f}% vs trenutno",
        "BS": "⬇ Najniža cijena<br>{val:.2f} {sym}<br>−{pct:.1f}% vs trenutno",
        "SR": "⬇ Najniža cena<br>{val:.2f} {sym}<br>−{pct:.1f}% vs trenutno",
        "MK": "⬇ Најниска цена<br>{val:.2f} {sym}<br>−{pct:.1f}% vs тековно",
        "ME": "⬇ Najniža cijena<br>{val:.2f} {sym}<br>−{pct:.1f}% vs trenutno",
    },
    "forecast.no_drop_expected": {
        "EN": "Lowest forecast price<br>{val:.2f} {sym}<br>+{pct:.1f}% above current – no drop expected",
        "HR": "Najniža prognozirana cijena<br>{val:.2f} {sym}<br>+{pct:.1f}% iznad trenutne – nema pada",
        "SL": "Najnižja napovedana cena<br>{val:.2f} {sym}<br>+{pct:.1f}% nad trenutno – brez padca",
        "BS": "Najniža prognozirana cijena<br>{val:.2f} {sym}<br>+{pct:.1f}% iznad trenutne – nema pada",
        "SR": "Najniža prognozirana cena<br>{val:.2f} {sym}<br>+{pct:.1f}% iznad trenutne – nema pada",
        "MK": "Најниска прогнозирана цена<br>{val:.2f} {sym}<br>+{pct:.1f}% над тековна – без пад",
        "ME": "Najniža prognozirana cijena<br>{val:.2f} {sym}<br>+{pct:.1f}% iznad trenutne – nema pada",
    },
    "forecast.kpi.future_price": {
        "EN": "Price in {days} days", "HR": "Cijena za {days} dana",
        "SL": "Cena čez {days} dni", "BS": "Cijena za {days} dana",
        "SR": "Cena za {days} dana", "MK": "Цена за {days} дена",
        "ME": "Cijena za {days} dana",
    },
    "forecast.kpi.lowest": {
        "EN": "Expected lowest price", "HR": "Očekivana najniža cijena",
        "SL": "Pričakovana najnižja cena", "BS": "Očekivana najniža cijena",
        "SR": "Očekivana najniža cena", "MK": "Очекувана најниска цена",
        "ME": "Očekivana najniža cijena",
    },
    "forecast.kpi.highest": {
        "EN": "Expected highest price", "HR": "Očekivana najviša cijena",
        "SL": "Pričakovana najvišja cena", "BS": "Očekivana najviša cijena",
        "SR": "Očekivana najviša cena", "MK": "Очекувана највисока цена",
        "ME": "Očekivana najviša cijena",
    },
    "forecast.kpi.max_drop": {
        "EN": "Max. expected drop", "HR": "Maks. očekivani pad",
        "SL": "Maks. pričakovani padec", "BS": "Maks. očekivani pad",
        "SR": "Maks. očekivani pad", "MK": "Макс. очекуван пад",
        "ME": "Maks. očekivani pad",
    },
    "forecast.kpi.max_rise": {
        "EN": "Max. expected rise", "HR": "Maks. očekivani rast",
        "SL": "Maks. pričakovana rast", "BS": "Maks. očekivani rast",
        "SR": "Maks. očekivani rast", "MK": "Макс. очекуван раст",
        "ME": "Maks. očekivani rast",
    },
    "forecast.kpi.change": {
        "EN": "Expected price change", "HR": "Očekivana promjena cijene",
        "SL": "Pričakovana sprememba cene", "BS": "Očekivana promjena cijene",
        "SR": "Očekivana promena cene", "MK": "Очекувана промена на цена",
        "ME": "Očekivana promjena cijene",
    },
    "forecast.no_drop": {
        "EN": "No expected drop", "HR": "Nema očekivanog pada",
        "SL": "Brez pričakovanega padca", "BS": "Nema očekivanog pada",
        "SR": "Nema očekivanog pada", "MK": "Без очекуван пад",
        "ME": "Nema očekivanog pada",
    },
    "forecast.no_rise": {
        "EN": "No expected rise", "HR": "Nema očekivanog rasta",
        "SL": "Brez pričakovane rasti", "BS": "Nema očekivanog rasta",
        "SR": "Nema očekivanog rasta", "MK": "Без очекуван раст",
        "ME": "Nema očekivanog rasta",
    },
    "forecast.audit_title": {
        "EN": "🔬 Forecast audit", "HR": "🔬 Audit prognoze",
        "SL": "🔬 Revizija napovedi", "BS": "🔬 Audit prognoze",
        "SR": "🔬 Audit prognoze", "MK": "🔬 Ревизија на прогноза",
        "ME": "🔬 Audit prognoze",
    },
    "forecast.audit.status": {
        "EN": "Status", "HR": "Status", "SL": "Status",
        "BS": "Status", "SR": "Status", "MK": "Статус", "ME": "Status",
    },
    "forecast.audit.trust": {
        "EN": "Trust level", "HR": "Razina pouzdanosti",
        "SL": "Stopnja zaupanja", "BS": "Nivo pouzdanosti",
        "SR": "Nivo pouzdanosti", "MK": "Ниво на доверба",
        "ME": "Nivo pouzdanosti",
    },
    "forecast.audit.source": {
        "EN": "Data source", "HR": "Izvor podataka",
        "SL": "Vir podatkov", "BS": "Izvor podataka",
        "SR": "Izvor podataka", "MK": "Извор на податоци",
        "ME": "Izvor podataka",
    },
    "forecast.audit.observations": {
        "EN": "Observation count", "HR": "Broj opažanja",
        "SL": "Število opazovanj", "BS": "Broj opservacija",
        "SR": "Broj opservacija", "MK": "Број на опсервации",
        "ME": "Broj opservacija",
    },
    "forecast.audit.history_days": {
        "EN": "History (days)", "HR": "Povijest (dani)",
        "SL": "Zgodovina (dni)", "BS": "Historija (dani)",
        "SR": "Istorija (dani)", "MK": "Историја (дена)",
        "ME": "Istorija (dani)",
    },
    "forecast.audit.outliers": {
        "EN": "Outliers removed", "HR": "Uklonjenih outliera",
        "SL": "Odstranjeni osamelci", "BS": "Uklonjenih outliera",
        "SR": "Uklonjenih outliera", "MK": "Отстранети отстапувања",
        "ME": "Uklonjenih outliera",
    },
    "forecast.audit.mape": {
        "EN": "Backtest MAPE", "HR": "Backtest MAPE",
        "SL": "Backtest MAPE", "BS": "Backtest MAPE",
        "SR": "Backtest MAPE", "MK": "Backtest MAPE",
        "ME": "Backtest MAPE",
    },
    # ── Section headers (Tab 1 detail, upcoming, ml) ──
    "section.detail": {
        "EN": "🔎 Detail view & historical evidence", "HR": "🔎 Detaljan prikaz i povijesni dokazi",
        "SL": "🔎 Podroben pregled in zgodovinski dokazi",
        "BS": "🔎 Detaljan prikaz i historijski dokazi",
        "SR": "🔎 Detaljan prikaz i istorijski dokazi",
        "MK": "🔎 Детален преглед и историски докази",
        "ME": "🔎 Detaljan prikaz i istorijski dokazi",
    },
    "section.upcoming": {
        "EN": "📅 Upcoming promotions (detail)", "HR": "📅 Nadolazeće akcije (detalj)",
        "SL": "📅 Prihodnje akcije (podrobnosti)", "BS": "📅 Nadolazeće akcije (detalj)",
        "SR": "📅 Nadolazeće akcije (detalj)", "MK": "📅 Претстојни акции (детал)",
        "ME": "📅 Nadolazeće akcije (detalj)",
    },
    "section.ml_expander": {
        "EN": "🧪 Experimental: LightGBM classifier (not KAM-grade)",
        "HR": "🧪 Eksperimentalno: LightGBM klasifikator (nije KAM razine)",
        "SL": "🧪 Eksperimentalno: LightGBM klasifikator (ni za KAM)",
        "BS": "🧪 Eksperimentalno: LightGBM klasifikator (nije za KAM)",
        "SR": "🧪 Eksperimentalno: LightGBM klasifikator (nije za KAM)",
        "MK": "🧪 Експериментално: LightGBM класификатор (не за КАМ)",
        "ME": "🧪 Eksperimentalno: LightGBM klasifikator (nije za KAM)",
    },
    "section.dq_filter": {
        "EN": "⚙️ Data quality filters", "HR": "⚙️ Filtri kvalitete podataka",
        "SL": "⚙️ Filtri kakovosti podatkov", "BS": "⚙️ Filtri kvaliteta podataka",
        "SR": "⚙️ Filtri kvaliteta podataka", "MK": "⚙️ Филтри за квалитет на податоци",
        "ME": "⚙️ Filtri kvaliteta podataka",
    },
    "section.advanced_filters": {
        "EN": "⚙️ Advanced forecast filters", "HR": "⚙️ Napredni filtri prognoze",
        "SL": "⚙️ Napredni filtri napovedi", "BS": "⚙️ Napredni filtri prognoze",
        "SR": "⚙️ Napredni filtri prognoze", "MK": "⚙️ Напредни филтри за прогноза",
        "ME": "⚙️ Napredni filtri prognoze",
    },
    "section.advanced_settings": {
        "EN": "⚙️ Advanced settings", "HR": "⚙️ Napredne postavke",
        "SL": "⚙️ Napredne nastavitve", "BS": "⚙️ Napredne postavke",
        "SR": "⚙️ Napredne postavke", "MK": "⚙️ Напредни поставки",
        "ME": "⚙️ Napredne postavke",
    },
    "section.audit": {
        "EN": "🔬 Audit & filter trace", "HR": "🔬 Audit i pregled filtera",
        "SL": "🔬 Revizija in sled filtrov", "BS": "🔬 Audit i pregled filtera",
        "SR": "🔬 Audit i pregled filtera", "MK": "🔬 Ревизија и пресек на филтри",
        "ME": "🔬 Audit i pregled filtera",
    },
    # ── Detail pane (Tab 1) ──
    "detail.product": {
        "EN": "Product", "HR": "Proizvod", "SL": "Izdelek",
        "BS": "Proizvod", "SR": "Proizvod", "MK": "Производ", "ME": "Proizvod",
    },
    "detail.brand": {
        "EN": "Brand", "HR": "Marka", "SL": "Znamka",
        "BS": "Marka", "SR": "Marka", "MK": "Бренд", "ME": "Marka",
    },
    "detail.retailer": {
        "EN": "Retailer", "HR": "Prodavaonica", "SL": "Trgovec",
        "BS": "Prodavnica", "SR": "Prodavnica", "MK": "Продавница", "ME": "Prodavnica",
    },
    "detail.market": {
        "EN": "Market", "HR": "Tržište", "SL": "Trg",
        "BS": "Tržište", "SR": "Tržište", "MK": "Пазар", "ME": "Tržište",
    },
    "detail.price_module": {
        "EN": "Price module", "HR": "Cjenovni modul", "SL": "Cenovni modul",
        "BS": "Cjenovni modul", "SR": "Cenovni modul", "MK": "Ценовен модул",
        "ME": "Cjenovni modul",
    },
    "detail.expected": {
        "EN": "Expected", "HR": "Očekivano", "SL": "Pričakovano",
        "BS": "Očekivano", "SR": "Očekivano", "MK": "Очекувано", "ME": "Očekivano",
    },
    "detail.last_promo_price": {
        "EN": "Last promo price", "HR": "Posljednja promo cijena",
        "SL": "Zadnja promo cena", "BS": "Posljednja promo cijena",
        "SR": "Poslednja promo cena", "MK": "Последна промо цена",
        "ME": "Posljednja promo cijena",
    },
    "detail.avg_90d": {
        "EN": "Avg. 90 days", "HR": "Prosj. 90 dana",
        "SL": "Povp. 90 dni", "BS": "Prosj. 90 dana",
        "SR": "Prosj. 90 dana", "MK": "Просек 90 дена",
        "ME": "Prosj. 90 dana",
    },
    "detail.avg_180d": {
        "EN": "Avg. 180 days", "HR": "Prosj. 180 dana",
        "SL": "Povp. 180 dni", "BS": "Prosj. 180 dana",
        "SR": "Prosj. 180 dana", "MK": "Просек 180 дена",
        "ME": "Prosj. 180 dana",
    },
    "detail.minmax_12m": {
        "EN": "Min/Max 12 months", "HR": "Min/Maks 12 mjeseci",
        "SL": "Min/Maks 12 mesecev", "BS": "Min/Maks 12 mjeseci",
        "SR": "Min/Maks 12 meseci", "MK": "Мин/Макс 12 месеци",
        "ME": "Min/Maks 12 mjeseci",
    },
    "detail.change_vs_last": {
        "EN": "Δ vs. last promo", "HR": "Δ vs. zadnja akcija",
        "SL": "Δ vs. zadnja akcija", "BS": "Δ vs. zadnja akcija",
        "SR": "Δ vs. poslednja akcija", "MK": "Δ vs. последна акција",
        "ME": "Δ vs. zadnja akcija",
    },
    "detail.cycle_signal": {
        "EN": "Cycle & signal", "HR": "Ciklus i signal",
        "SL": "Cikel in signal", "BS": "Ciklus i signal",
        "SR": "Ciklus i signal", "MK": "Циклус и сигнал",
        "ME": "Ciklus i signal",
    },
    "detail.avg_cycle": {
        "EN": "Avg. cycle", "HR": "Prosj. ciklus",
        "SL": "Povp. cikel", "BS": "Prosj. ciklus",
        "SR": "Prosj. ciklus", "MK": "Просечен циклус",
        "ME": "Prosj. ciklus",
    },
    "detail.last_promo": {
        "EN": "Last promotion", "HR": "Zadnja akcija",
        "SL": "Zadnja akcija", "BS": "Zadnja akcija",
        "SR": "Poslednja akcija", "MK": "Последна акција",
        "ME": "Poslednja akcija",
    },
    "detail.typical_duration": {
        "EN": "Typical duration", "HR": "Tipično trajanje",
        "SL": "Tipično trajanje", "BS": "Tipično trajanje",
        "SR": "Tipično trajanje", "MK": "Типично траење",
        "ME": "Tipično trajanje",
    },
    "detail.typical_discount": {
        "EN": "Typical discount", "HR": "Tipičan popust",
        "SL": "Tipičen popust", "BS": "Tipičan popust",
        "SR": "Tipičan popust", "MK": "Типичен попуст",
        "ME": "Tipičan popust",
    },
    "detail.signal": {
        "EN": "Signal", "HR": "Signal", "SL": "Signal",
        "BS": "Signal", "SR": "Signal", "MK": "Сигнал", "ME": "Signal",
    },
    "detail.probability": {
        "EN": "Probability", "HR": "Vjerojatnost", "SL": "Verjetnost",
        "BS": "Vjerovatnoća", "SR": "Verovatnoća", "MK": "Веројатност", "ME": "Vjerovatnoća",
    },
    "detail.justification": {
        "EN": "Justification", "HR": "Obrazloženje", "SL": "Utemeljitev",
        "BS": "Obrazloženje", "SR": "Obrazloženje", "MK": "Образложение", "ME": "Obrazloženje",
    },
    "detail.history_5": {
        "EN": "Historical promotions (last 5):",
        "HR": "Povijesne akcije (zadnjih 5):",
        "SL": "Zgodovinske akcije (zadnjih 5):",
        "BS": "Historijske akcije (zadnjih 5):",
        "SR": "Istorijske akcije (poslednjih 5):",
        "MK": "Историски акции (последни 5):",
        "ME": "Istorijske akcije (poslednjih 5):",
    },
    "detail.select_combo": {
        "EN": "Select product–retailer combination",
        "HR": "Odaberite kombinaciju proizvod–prodavaonica",
        "SL": "Izberite kombinacijo izdelek–trgovec",
        "BS": "Odaberite kombinaciju proizvod–prodavnica",
        "SR": "Odaberite kombinaciju proizvod–prodavnica",
        "MK": "Изберете комбинација производ–продавница",
        "ME": "Odaberite kombinaciju proizvod–prodavnica",
    },
    # ── Upcoming promotions ──
    "upcoming.caption": {
        "EN": "All promotions starting in the next {w} weeks · Market: {market}",
        "HR": "Sve akcije koje počinju u sljedećih {w} tjedana · Tržište: {market}",
        "SL": "Vse akcije, ki se začnejo v naslednjih {w} tednih · Trg: {market}",
        "BS": "Sve akcije koje počinju u sljedećih {w} sedmica · Tržište: {market}",
        "SR": "Sve akcije koje počinju u sledećih {w} nedelja · Tržište: {market}",
        "MK": "Сите акции што почнуваат во следните {w} недели · Пазар: {market}",
        "ME": "Sve akcije koje počinju u sljedećih {w} sedmica · Tržište: {market}",
    },
    "upcoming.found": {
        "EN": "Promotions found", "HR": "Pronađene akcije",
        "SL": "Najdene akcije", "BS": "Pronađene akcije",
        "SR": "Pronađene akcije", "MK": "Пронајдени акции",
        "ME": "Pronađene akcije",
    },
    "upcoming.retailers_involved": {
        "EN": "Retailers involved", "HR": "Uključene prodavaonice",
        "SL": "Vključeni trgovci", "BS": "Uključene prodavnice",
        "SR": "Uključene prodavnice", "MK": "Вклучени продавници",
        "ME": "Uključene prodavnice",
    },
    "upcoming.avg_discount": {
        "EN": "Avg. discount", "HR": "Prosj. popust",
        "SL": "Povp. popust", "BS": "Prosj. popust",
        "SR": "Prosj. popust", "MK": "Просечен попуст",
        "ME": "Prosj. popust",
    },
    "upcoming.faulty": {
        "EN": "⚠️ Faulty rows", "HR": "⚠️ Neispravnih redaka",
        "SL": "⚠️ Napačnih vrstic", "BS": "⚠️ Neispravnih redaka",
        "SR": "⚠️ Neispravnih redova", "MK": "⚠️ Неправилни редови",
        "ME": "⚠️ Neispravnih redova",
    },
    "upcoming.none": {
        "EN": "No upcoming promotions found for this filter.",
        "HR": "Nema nadolazećih akcija za ovaj filter.",
        "SL": "Za ta filter ni prihodnjih akcij.",
        "BS": "Nema nadolazećih akcija za ovaj filter.",
        "SR": "Nema nadolazećih akcija za ovaj filter.",
        "MK": "Нема претстојни акции за овој филтер.",
        "ME": "Nema nadolazećih akcija za ovaj filter.",
    },
    # ── Treemap / charts ──
    "charts.treemap_disabled": {
        "EN": "Treemap disabled – filter narrower (too many rows or too many 'Other' categories).",
        "HR": "Treemap onemogućen – filtrirajte uže (previše redaka ili previše 'Ostalo').",
        "SL": "Treemap onemogočen – filtrirajte ožje (preveč vrstic ali preveč 'Ostalo').",
        "BS": "Treemap onemogućen – filtrirajte uže (previše redaka ili previše 'Ostalo').",
        "SR": "Treemap onemogućen – filtrirajte uže (previše redova ili previše 'Ostalo').",
        "MK": "Treemap е оневозможен – филтрирајте потесно (премногу редови или премногу 'Друго').",
        "ME": "Treemap onemogućen – filtrirajte uže (previše redova ili previše 'Ostalo').",
    },
    "charts.treemap_min_rows": {
        "EN": "Need at least {n} rows to render a treemap (currently {got}).",
        "HR": "Potrebno najmanje {n} redaka za treemap (trenutno {got}).",
        "SL": "Potrebno najmanj {n} vrstic za treemap (trenutno {got}).",
        "BS": "Potrebno najmanje {n} redaka za treemap (trenutno {got}).",
        "SR": "Potrebno najmanje {n} redova za treemap (trenutno {got}).",
        "MK": "Потребни најмалку {n} редови за treemap (тековно {got}).",
        "ME": "Potrebno najmanje {n} redova za treemap (trenutno {got}).",
    },
    "charts.distribution_title": {
        "EN": "Promotion distribution – next {w} weeks",
        "HR": "Distribucija akcija – sljedećih {w} tjedana",
        "SL": "Porazdelitev akcij – naslednjih {w} tednov",
        "BS": "Distribucija akcija – sljedećih {w} sedmica",
        "SR": "Distribucija akcija – sledećih {w} nedelja",
        "MK": "Распределба на акции – следните {w} недели",
        "ME": "Distribucija akcija – sljedećih {w} sedmica",
    },
    # ── ML expander ──
    "ml.caption": {
        "EN": "This model class is experimental and not suitable for operational decisions. It only trains with real data (Supabase + ≥ 50 examples).",
        "HR": "Ova klasa modela je eksperimentalna i nije pogodna za operativne odluke. Trenira se samo sa stvarnim podacima (Supabase + ≥ 50 primjera).",
        "SL": "Ta razred modela je eksperimentalen in ni primeren za operativne odločitve. Uči se le z resničnimi podatki (Supabase + ≥ 50 primerov).",
        "BS": "Ova klasa modela je eksperimentalna i nije pogodna za operativne odluke. Trenira se samo sa stvarnim podacima (Supabase + ≥ 50 primjera).",
        "SR": "Ova klasa modela je eksperimentalna i nije pogodna za operativne odluke. Trenira se samo sa stvarnim podacima (Supabase + ≥ 50 primera).",
        "MK": "Оваа модел класа е експериментална и не е погодна за оперативни одлуки. Се обучува само со вистински податоци (Supabase + ≥ 50 примери).",
        "ME": "Ova klasa modela je eksperimentalna i nije pogodna za operativne odluke. Trenira se samo sa stvarnim podacima (Supabase + ≥ 50 primjera).",
    },
    "ml.retrain": {
        "EN": "🔁 Retrain model", "HR": "🔁 Ponovno treniraj model",
        "SL": "🔁 Ponovno usposobi model", "BS": "🔁 Ponovo treniraj model",
        "SR": "🔁 Ponovo treniraj model", "MK": "🔁 Повторно обучи модел",
        "ME": "🔁 Ponovo treniraj model",
    },
    "ml.no_db": {
        "EN": "No live DB connection – ML training disabled.",
        "HR": "Nema live DB veze – ML trening onemogućen.",
        "SL": "Ni žive povezave z bazo – usposabljanje ML onemogočeno.",
        "BS": "Nema live DB veze – ML trening onemogućen.",
        "SR": "Nema live DB veze – ML trening onemogućen.",
        "MK": "Нема врска со DB – обуката е оневозможена.",
        "ME": "Nema live DB veze – ML trening onemogućen.",
    },
    # ── Audit / debug expander ──
    "audit.raw_rows": {
        "EN": "Raw rows", "HR": "Sirovi redci", "SL": "Surove vrstice",
        "BS": "Sirovi redovi", "SR": "Sirovi redovi", "MK": "Сурови редови",
        "ME": "Sirovi redovi",
    },
    "audit.after_market": {
        "EN": "After market filter", "HR": "Nakon filtra tržišta",
        "SL": "Po filtru trga", "BS": "Nakon filtra tržišta",
        "SR": "Nakon filtra tržišta", "MK": "По филтер на пазар",
        "ME": "Nakon filtra tržišta",
    },
    "audit.after_category": {
        "EN": "After category filter", "HR": "Nakon filtra kategorije",
        "SL": "Po filtru kategorije", "BS": "Nakon filtra kategorije",
        "SR": "Nakon filtra kategorije", "MK": "По филтер на категорија",
        "ME": "Nakon filtra kategorije",
    },
    "audit.after_brand": {
        "EN": "After brand search", "HR": "Nakon pretrage marke",
        "SL": "Po iskanju znamke", "BS": "Nakon pretrage marke",
        "SR": "Nakon pretrage marke", "MK": "По пребарување на бренд",
        "ME": "Nakon pretrage marke",
    },
    "audit.after_retailer": {
        "EN": "After retailer filter", "HR": "Nakon filtra prodavaonice",
        "SL": "Po filtru trgovca", "BS": "Nakon filtra prodavnice",
        "SR": "Nakon filtra prodavnice", "MK": "По филтер на продавница",
        "ME": "Nakon filtra prodavnice",
    },
    "audit.forecast_rows": {
        "EN": "Forecast rows", "HR": "Redovi prognoze",
        "SL": "Vrstice napovedi", "BS": "Redovi prognoze",
        "SR": "Redovi prognoze", "MK": "Редови на прогноза",
        "ME": "Redovi prognoze",
    },
    "audit.upcoming_rows": {
        "EN": "Upcoming rows", "HR": "Nadolazeći redovi",
        "SL": "Prihodnje vrstice", "BS": "Nadolazeći redovi",
        "SR": "Nadolazeći redovi", "MK": "Претстојни редови",
        "ME": "Nadolazeći redovi",
    },
    # ── Build info / Sidebar misc ──
    "build.version": {
        "EN": "Build", "HR": "Build", "SL": "Build",
        "BS": "Build", "SR": "Build", "MK": "Билд", "ME": "Build",
    },
    "build.mode": {
        "EN": "App mode", "HR": "Način rada", "SL": "Način dela",
        "BS": "Način rada", "SR": "Način rada", "MK": "Режим",
        "ME": "Način rada",
    },
    "build.data_source": {
        "EN": "Data source", "HR": "Izvor podataka", "SL": "Vir podatkov",
        "BS": "Izvor podataka", "SR": "Izvor podataka", "MK": "Извор на податоци",
        "ME": "Izvor podataka",
    },
    "build.language": {
        "EN": "Language", "HR": "Jezik", "SL": "Jezik",
        "BS": "Jezik", "SR": "Jezik", "MK": "Јазик", "ME": "Jezik",
    },
    "build.production": {
        "EN": "production", "HR": "produkcija", "SL": "produkcija",
        "BS": "produkcija", "SR": "produkcija", "MK": "продукција", "ME": "produkcija",
    },
    "build.demo": {
        "EN": "demo", "HR": "demo", "SL": "demo",
        "BS": "demo", "SR": "demo", "MK": "демо", "ME": "demo",
    },
    "build.supabase": {
        "EN": "supabase", "HR": "supabase", "SL": "supabase",
        "BS": "supabase", "SR": "supabase", "MK": "supabase", "ME": "supabase",
    },
    "build.mock": {
        "EN": "mock", "HR": "demo (mock)", "SL": "demo (mock)",
        "BS": "demo (mock)", "SR": "demo (mock)", "MK": "demo (mock)",
        "ME": "demo (mock)",
    },
    # ── Trust level descriptions ──
    "trust.belastbar": {
        "EN": "Reliable", "HR": "Pouzdano", "SL": "Zanesljivo",
        "BS": "Pouzdano", "SR": "Pouzdano", "MK": "Веродостоен",
        "ME": "Pouzdano",
    },
    "trust.eingeschr": {
        "EN": "Limited", "HR": "Ograničeno", "SL": "Omejeno",
        "BS": "Ograničeno", "SR": "Ograničeno", "MK": "Ограничено",
        "ME": "Ograničeno",
    },
    "trust.nicht_belastbar": {
        "EN": "Not reliable", "HR": "Nije pouzdano", "SL": "Ni zanesljivo",
        "BS": "Nije pouzdano", "SR": "Nije pouzdano", "MK": "Не е веродостоен",
        "ME": "Nije pouzdano",
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
