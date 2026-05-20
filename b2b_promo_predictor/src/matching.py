"""Entity Resolution via Sentence-Transformers + Cosine Similarity.

Mappt rohe Produktnamen auf eine normierte Master-Datenbank.
Das Embedding-Modell wird beim ersten Aufruf geladen und gecacht.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from functools import lru_cache

from .config import settings


@dataclass
class MatchResult:
    """Ergebnis einer Produktzuordnung."""

    master_product: str
    score: float
    is_confident: bool  # True wenn score >= threshold


class ProductMatcher:
    """Mappt freie Produktnamen auf eine Master-Produktliste.

    Nutzt Sentence-Transformer-Embeddings und Cosine Similarity.
    Fällt bei fehlenden Abhängigkeiten auf einfaches Substring-Matching zurück.

    Args:
        model_name: HuggingFace-Modell-ID für den Encoder.
        threshold: Mindestscore (0–1) für selbstsicheres Matching.
    """

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float = 0.70,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.threshold = threshold
        self._model = None
        self._master_embeddings: np.ndarray | None = None
        self._master_list: list[str] = []

    def _load_model(self) -> None:
        """Lazy-lädt das Sentence-Transformer-Modell."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._model = None  # Fallback-Modus

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Berechnet Cosine Similarity zwischen Vektor a und Matrix b."""
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norms = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return b_norms @ a_norm

    def _fallback_match(self, raw_name: str, master_list: list[str]) -> MatchResult:
        """Einfaches Substring-Matching als Fallback ohne ML-Modell."""
        raw_lower = raw_name.lower()
        best_match = master_list[0]
        best_score = 0.0
        for candidate in master_list:
            overlap = sum(
                1 for word in candidate.lower().split() if word in raw_lower
            )
            score = overlap / max(len(candidate.split()), 1)
            if score > best_score:
                best_score = score
                best_match = candidate
        return MatchResult(
            master_product=best_match,
            score=round(best_score, 4),
            is_confident=best_score >= self.threshold,
        )

    def match_product(self, raw_name: str, master_list: list[str]) -> MatchResult:
        """Findet den besten Master-Produktnamen für einen rohen Produktnamen.

        Args:
            raw_name: Freier Produktname wie im Prospekt.
            master_list: Normierte Masterliste von Produktnamen.

        Returns:
            MatchResult mit bestem Match und Konfidenz-Score.
        """
        if not master_list:
            return MatchResult(master_product="", score=0.0, is_confident=False)

        self._load_model()

        if self._model is None:
            return self._fallback_match(raw_name, master_list)

        # Embeddings für Master-Liste cachen wenn gleiche Liste
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

        return MatchResult(
            master_product=master_list[best_idx],
            score=round(best_score, 4),
            is_confident=best_score >= self.threshold,
        )

    def batch_match(
        self, raw_names: list[str], master_list: list[str]
    ) -> list[MatchResult]:
        """Matcht mehrere Produktnamen auf einmal (effizienter als Einzelaufrufe).

        Args:
            raw_names: Liste roher Produktnamen.
            master_list: Normierte Masterliste.

        Returns:
            Liste von MatchResults in gleicher Reihenfolge wie raw_names.
        """
        if not master_list or not raw_names:
            return []

        self._load_model()

        if self._model is None:
            return [self._fallback_match(n, master_list) for n in raw_names]

        if self._master_list != master_list:
            self._master_list = master_list
            self._master_embeddings = self._model.encode(
                master_list, convert_to_numpy=True, show_progress_bar=False
            )

        query_embs = self._model.encode(
            raw_names, convert_to_numpy=True, show_progress_bar=False
        )

        results: list[MatchResult] = []
        for emb in query_embs:
            scores = self._cosine_similarity(emb, self._master_embeddings)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            results.append(
                MatchResult(
                    master_product=master_list[best_idx],
                    score=round(best_score, 4),
                    is_confident=best_score >= self.threshold,
                )
            )
        return results
