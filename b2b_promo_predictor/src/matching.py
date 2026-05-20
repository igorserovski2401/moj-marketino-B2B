"""Entity Resolution via Sentence-Transformers + Cosine Similarity.

Mappt rohe Produktnamen auf eine normierte Master-Datenbank.
Pipeline: Keyword-Regeln → Embedding → Threshold-Guard → Kategorie-Konflikt-Guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import settings

# ── Konstanten ────────────────────────────────────────────────────────────────

MIN_MATCH_SCORE: float = 0.85
UNKNOWN_MASTER_PRODUCT: str = "Nicht zugeordnet / Unbekannt"

MatchStatus = Literal[
    "keyword_exact",
    "embedding_high_confidence",
    "unmatched_low_confidence",
    "unmatched_category_conflict",
    "unmatched_empty",
]

# Keyword → Master-Produkt (lowercase lookup, Transformer wird übersprungen).
# Keyword-Match liefert Score 1.0 und unterliegt keinem Kategorie-Konflikt-Check.
KEYWORD_MASTER_RULES: dict[str, str] = {
    "red bull":      "Red Bull Energy Drink 250ml",
    "redbull":       "Red Bull Energy Drink 250ml",
    "milka":         "Milka Schokolade 300g",
    "coca-cola":     "Coca-Cola 1,5L PET",
    "coca cola":     "Coca-Cola 1,5L PET",
    "cocacola":      "Coca-Cola 1,5L PET",
    "nutella":       "Nutella Nuss-Nougat-Creme 450g",
    "ariel":         "Ariel Waschmittel 20 WL",
    "pampers":       "Pampers Baby-Dry Gr.3 44 Stk",
    "haribo":        "Haribo Goldbären 200g",
    "pringles":      "Pringles Original 185g",
    "ritter sport":  "Ritter Sport Voll-Nuss 100g",
    "juha":          "Suppe / Brühe",
    "majoneza":      "Mayonnaise",
    "majonez":       "Mayonnaise",
    "čaj":           "Tee",
    "caj":           "Tee",
    "vegeta":        "Vegeta Gewürzmischung",
}

# Kategorie-Paare (raw_cat_lower, master_cat_lower) die niemals gemacht werden dürfen.
BLOCKED_CATEGORY_PAIRS: set[tuple[str, str]] = {
    ("food",      "household"),
    ("household", "food"),
    ("drinks",    "cosmetics"),
    ("cosmetics", "drinks"),
    ("hrana",     "kozmetika"),
    ("kozmetika", "hrana"),
    ("piće",      "kućanska kemija"),
    ("kućanska kemija", "piće"),
    ("pice",      "kozmetika"),
    ("kozmetika", "pice"),
}


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    """Ergebnis einer Produktzuordnung mit vollständigem Audit-Trail."""

    original_product_name: str
    normalized_product_name: str
    master_product: str
    match_score: float
    match_status: MatchStatus
    match_method: Literal["keyword", "embedding", "fallback", "none"]
    raw_candidate: str
    category: str = ""
    master_category: str = ""

    @property
    def score(self) -> float:
        """Rückwärtskompatibles Alias für match_score."""
        return self.match_score

    @property
    def is_confident(self) -> bool:
        """True wenn das Match zuverlässig genug für die Anzeige ist."""
        return self.match_status in ("keyword_exact", "embedding_high_confidence")


# ── Private Helpers ───────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _keyword_match(raw_name: str) -> MatchResult | None:
    """Prüft Keyword-Regeln vor dem Transformer (Score = 1.0 bei Treffer)."""
    normalized = _normalize_name(raw_name)
    for keyword, master in KEYWORD_MASTER_RULES.items():
        if keyword in normalized:
            return MatchResult(
                original_product_name=raw_name,
                normalized_product_name=normalized,
                master_product=master,
                match_score=1.0,
                match_status="keyword_exact",
                match_method="keyword",
                raw_candidate=raw_name,
            )
    return None


def _check_category_conflict(raw_cat: str, master_cat: str) -> bool:
    """True wenn das Kategorienpaar cross-category geblockt ist."""
    return (raw_cat.lower(), master_cat.lower()) in BLOCKED_CATEGORY_PAIRS


def _unmatched(raw_name: str, reason: MatchStatus) -> MatchResult:
    """Erstellt ein explizit nicht-zugeordnetes MatchResult."""
    return MatchResult(
        original_product_name=raw_name,
        normalized_product_name=_normalize_name(raw_name),
        master_product=UNKNOWN_MASTER_PRODUCT,
        match_score=0.0,
        match_status=reason,
        match_method="none",
        raw_candidate=raw_name,
    )


# ── ProductMatcher ────────────────────────────────────────────────────────────

class ProductMatcher:
    """Mappt freie Produktnamen auf eine Master-Produktliste.

    Matching-Pipeline (in Prioritätsreihenfolge):
    1. KEYWORD_MASTER_RULES: exakte Keyword-Suche → Score 1.0
    2. Sentence-Transformer Embeddings + Cosine Similarity
    3. Fallback: Substring-Matching (wenn Modell nicht verfügbar)
    4. Threshold-Guard: score < MIN_MATCH_SCORE → UNKNOWN_MASTER_PRODUCT
    5. Kategorie-Konflikt-Guard: BLOCKED_CATEGORY_PAIRS → UNKNOWN_MASTER_PRODUCT

    Args:
        model_name: HuggingFace-Modell-ID.
        threshold: Mindestscore (0–1). Kann nicht unter MIN_MATCH_SCORE gesetzt werden.
    """

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float = MIN_MATCH_SCORE,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.threshold = max(float(threshold), MIN_MATCH_SCORE)
        self._model = None
        self._master_embeddings: np.ndarray | None = None
        self._master_list: list[str] = []

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._model = None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norms = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return b_norms @ a_norm

    def _fallback_match(self, raw_name: str, master_list: list[str]) -> MatchResult:
        """Substring-Matching als Fallback ohne ML-Modell."""
        raw_lower = raw_name.lower()
        best_match = master_list[0]
        best_score = 0.0
        for candidate in master_list:
            overlap = sum(1 for w in candidate.lower().split() if w in raw_lower)
            score = overlap / max(len(candidate.split()), 1)
            if score > best_score:
                best_score = score
                best_match = candidate

        is_confident = best_score >= self.threshold
        return MatchResult(
            original_product_name=raw_name,
            normalized_product_name=_normalize_name(raw_name),
            master_product=best_match if is_confident else UNKNOWN_MASTER_PRODUCT,
            match_score=round(best_score, 4),
            match_status="embedding_high_confidence" if is_confident else "unmatched_low_confidence",
            match_method="fallback",
            raw_candidate=best_match,
        )

    def _embedding_result(
        self,
        raw_name: str,
        best_score: float,
        best_candidate: str,
        raw_cat: str,
        master_cat: str,
    ) -> MatchResult:
        """Erstellt MatchResult nach Threshold- und Konflikt-Prüfung."""
        if best_score < self.threshold:
            return MatchResult(
                original_product_name=raw_name,
                normalized_product_name=_normalize_name(raw_name),
                master_product=UNKNOWN_MASTER_PRODUCT,
                match_score=round(best_score, 4),
                match_status="unmatched_low_confidence",
                match_method="embedding",
                raw_candidate=best_candidate,
                category=raw_cat,
                master_category=master_cat,
            )
        if raw_cat and master_cat and _check_category_conflict(raw_cat, master_cat):
            return MatchResult(
                original_product_name=raw_name,
                normalized_product_name=_normalize_name(raw_name),
                master_product=UNKNOWN_MASTER_PRODUCT,
                match_score=round(best_score, 4),
                match_status="unmatched_category_conflict",
                match_method="embedding",
                raw_candidate=best_candidate,
                category=raw_cat,
                master_category=master_cat,
            )
        return MatchResult(
            original_product_name=raw_name,
            normalized_product_name=_normalize_name(raw_name),
            master_product=best_candidate,
            match_score=round(best_score, 4),
            match_status="embedding_high_confidence",
            match_method="embedding",
            raw_candidate=best_candidate,
            category=raw_cat,
            master_category=master_cat,
        )

    def match_product(
        self,
        raw_name: str,
        master_list: list[str],
        raw_category: str = "",
        master_categories: list[str] | None = None,
    ) -> MatchResult:
        """Findet den besten Master-Produktnamen für einen rohen Produktnamen.

        Args:
            raw_name: Freier Produktname.
            master_list: Normierte Masterliste.
            raw_category: Kategorie des Rohprodukts (für Konflikt-Guard).
            master_categories: Kategorien der Master-Produkte (parallel zu master_list).

        Returns:
            MatchResult mit Status-Information.
        """
        if not master_list:
            return _unmatched(raw_name, "unmatched_empty")

        # 1. Keyword-Regeln (kein Modell nötig, Score 1.0)
        kw = _keyword_match(raw_name)
        if kw is not None:
            kw.category = raw_category
            return kw

        # 2. Embedding-Matching
        self._load_model()
        if self._model is None:
            fb = self._fallback_match(raw_name, master_list)
            fb.category = raw_category
            return fb

        if self._master_list != master_list:
            self._master_list = master_list
            self._master_embeddings = self._model.encode(
                master_list, convert_to_numpy=True, show_progress_bar=False
            )

        query_emb = self._model.encode(
            [raw_name], convert_to_numpy=True, show_progress_bar=False
        )[0]
        scores = self._cosine_similarity(query_emb, self._master_embeddings)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_candidate = master_list[best_idx]
        master_cat = (
            master_categories[best_idx]
            if master_categories and len(master_categories) > best_idx
            else ""
        )
        return self._embedding_result(raw_name, best_score, best_candidate, raw_category, master_cat)

    def batch_match(
        self,
        raw_names: list[str],
        master_list: list[str],
        raw_categories: list[str] | None = None,
        master_categories: list[str] | None = None,
    ) -> list[MatchResult]:
        """Matcht mehrere Produktnamen auf einmal (effizienter als Einzelaufrufe).

        Args:
            raw_names: Liste roher Produktnamen.
            master_list: Normierte Masterliste.
            raw_categories: Kategorien der Rohprodukte (parallel zu raw_names).
            master_categories: Kategorien der Master-Produkte (parallel zu master_list).

        Returns:
            Liste von MatchResults in gleicher Reihenfolge wie raw_names.
        """
        if not master_list or not raw_names:
            return []

        # Phase 1: Keyword-Matching (kein Modell nötig)
        kw_results: dict[int, MatchResult] = {}
        remaining_indices: list[int] = []
        for i, name in enumerate(raw_names):
            kw = _keyword_match(name)
            if kw is not None:
                kw.category = raw_categories[i] if raw_categories else ""
                kw_results[i] = kw
            else:
                remaining_indices.append(i)

        if not remaining_indices:
            return [kw_results[i] for i in range(len(raw_names))]

        # Phase 2: Embedding für verbleibende Namen
        self._load_model()
        emb_results: dict[int, MatchResult] = {}

        if self._model is None:
            for orig_idx in remaining_indices:
                raw_cat = raw_categories[orig_idx] if raw_categories else ""
                fb = self._fallback_match(raw_names[orig_idx], master_list)
                fb.category = raw_cat
                emb_results[orig_idx] = fb
        else:
            if self._master_list != master_list:
                self._master_list = master_list
                self._master_embeddings = self._model.encode(
                    master_list, convert_to_numpy=True, show_progress_bar=False
                )

            remaining_names = [raw_names[i] for i in remaining_indices]
            query_embs = self._model.encode(
                remaining_names, convert_to_numpy=True, show_progress_bar=False
            )

            for pos, orig_idx in enumerate(remaining_indices):
                scores = self._cosine_similarity(query_embs[pos], self._master_embeddings)
                best_idx = int(np.argmax(scores))
                best_score = float(scores[best_idx])
                best_candidate = master_list[best_idx]
                raw_cat = raw_categories[orig_idx] if raw_categories else ""
                master_cat = (
                    master_categories[best_idx]
                    if master_categories and len(master_categories) > best_idx
                    else ""
                )
                emb_results[orig_idx] = self._embedding_result(
                    raw_names[orig_idx], best_score, best_candidate, raw_cat, master_cat
                )

        return [
            kw_results[i] if i in kw_results else emb_results[i]
            for i in range(len(raw_names))
        ]
