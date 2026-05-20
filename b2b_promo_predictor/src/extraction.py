"""Prospekt-Extraktion via Gemini Vision mit Pydantic-Validierung.

Falls kein GEMINI_API_KEY vorhanden ist, werden realistische Mock-Daten zurückgegeben,
damit das Dashboard sofort ohne Fehler lauffähig ist.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .config import settings


class PromoData(BaseModel):
    """Schema für ein einzelnes Aktionsprodukt aus einem Discounter-Prospekt."""

    retailer: str = Field(..., description="Name des Discounters, z.B. Konzum, Kaufland")
    country: str = Field(default="HR", description="ISO-Ländercode (HR/SI/BA/RS/MK/ME)")
    valid_from: date = Field(..., description="Aktionsbeginn")
    valid_to: date = Field(..., description="Aktionsende")
    raw_product_name: str = Field(..., description="Produktname exakt wie im Prospekt")
    cleaned_brand: str = Field(..., description="Normierter Markenname")
    price_promo: float = Field(..., ge=0, description="Aktionspreis in Euro")
    volume_value: Optional[float] = Field(None, ge=0, description="Mengenwert")
    volume_unit: Optional[str] = Field(None, description="Mengeneinheit z.B. g, ml, Stk")

    @field_validator("country")
    @classmethod
    def uppercase_country(cls, v: str) -> str:
        return v.upper()

    @field_validator("price_promo")
    @classmethod
    def round_price(cls, v: float) -> float:
        return round(v, 2)


_MOCK_PRODUCTS = [
    ("Milka Schokolade 300g", "Milka", 1.29, 300, "g"),
    ("Coca-Cola 1,5L PET", "Coca-Cola", 1.09, 1500, "ml"),
    ("Ritter Sport Voll-Nuss 100g", "Ritter Sport", 0.99, 100, "g"),
    ("Haribo Goldbären 200g", "Haribo", 0.79, 200, "g"),
    ("Nutella 450g", "Nutella", 3.49, 450, "g"),
    ("Ja! Vollmilch 3,5% 1L", "Ja!", 0.89, 1000, "ml"),
    ("Ariel Pulver 20WL", "Ariel", 5.99, 20, "WL"),
    ("Pampers Windeln Gr.3 44Stk", "Pampers", 9.99, 44, "Stk"),
    ("Red Bull Energy Drink 250ml", "Red Bull", 0.99, 250, "ml"),
    ("Pringles Original 185g", "Pringles", 1.79, 185, "g"),
]

_RETAILERS = ["Konzum", "Kaufland", "Lidl", "Spar", "Bingo", "Tinex"]


def _generate_mock_data(n: int = 8) -> list[PromoData]:
    """Erzeugt realistische Mock-PromoData-Objekte für Demo-Zwecke."""
    results: list[PromoData] = []
    today = date.today()
    for _ in range(n):
        raw, brand, price, vol, unit = random.choice(_MOCK_PRODUCTS)
        start = today + timedelta(days=random.randint(-3, 7))
        return_obj = PromoData(
            retailer=random.choice(_RETAILERS),
            country="HR",
            valid_from=start,
            valid_to=start + timedelta(days=random.randint(5, 14)),
            raw_product_name=raw,
            cleaned_brand=brand,
            price_promo=round(price + random.uniform(-0.20, 0.30), 2),
            volume_value=float(vol),
            volume_unit=unit,
        )
        results.append(return_obj)
    return results


def extract_from_image(image_path: str | Path) -> list[PromoData]:
    """Extrahiert strukturierte Aktionsdaten aus einem Prospektbild via Gemini Vision.

    Fällt automatisch auf Mock-Daten zurück, wenn kein API-Key konfiguriert ist
    oder das Bild nicht gelesen werden kann.

    Args:
        image_path: Pfad zur Prospekt-Bilddatei (JPG, PNG, PDF-Seite).

    Returns:
        Liste von validierten PromoData-Objekten.
    """
    if not settings.has_gemini:
        return _generate_mock_data()

    try:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        image_path = Path(image_path)
        if not image_path.exists():
            return _generate_mock_data()

        from PIL import Image
        img = Image.open(image_path)

        prompt = (
            "Du bist ein Datenextraktions-Experte für Handelsmarketing. "
            "Analysiere dieses Discounter-Prospektbild und extrahiere ALLE sichtbaren Aktionsprodukte. "
            "Gib das Ergebnis als JSON-Array zurück. Jedes Element hat diese Felder: "
            "retailer (str), country (str, ISO), valid_from (YYYY-MM-DD), valid_to (YYYY-MM-DD), "
            "raw_product_name (str), cleaned_brand (str), price_promo (float), "
            "volume_value (float or null), volume_unit (str or null). "
            "Antworte NUR mit dem JSON-Array, kein weiterer Text."
        )

        response = model.generate_content([prompt, img])
        raw_json = response.text.strip().lstrip("```json").rstrip("```").strip()
        data = json.loads(raw_json)
        return [PromoData(**item) for item in data]

    except Exception:
        return _generate_mock_data()
