"""
retriever.py — HybridRetriever untuk RAG Chatbot Perpustakaan PNJ
==================================================================
Arsitektur:
  1. BM25   : keyword matching (judul × 3, topik × 2, penulis, call_number)
  2. Dense  : semantic matching via bge-m3 (termasuk deskripsi)
  3. Fusion : weighted sum ATAU Reciprocal Rank Fusion (RRF)
  4. Rerank : bge-reranker-v2-m3 cross-encoder (OPSIONAL)

Catatan pemilihan konfigurasi (ablation study — scripts/06_ablation_study.py):
  Config A — BM25 only        : MRR@5=0.9021, NDCG@5=0.7377, Hit@5=95.0%
  Config B — Dense only       : MRR@5=0.7333, NDCG@5=0.4865, Hit@5=87.5%
  Config C — Hybrid no rerank : MRR@5=0.8917, NDCG@5=0.7682, Hit@5=95.0%  ← TERPILIH
  Config D — Hybrid + rerank  : MRR@5=0.7250, NDCG@5=0.4571, Hit@5=87.5%  (LEBIH BURUK)

Production api.py menggunakan Config C (search_no_rerank).
Reranker tetap tersedia di kelas ini untuk keperluan ablation/eksperimen.
"""

import os
import logging
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

log = logging.getLogger(__name__)


# ── Konstanta default (bisa di-override saat init) ────────────────────────────
DEFAULT_BM25_WEIGHT    = 0.45   # optimum NDCG@5 dari grid search (grid_search_weights.py); band 0.15-0.45 robust
DEFAULT_DENSE_WEIGHT   = 0.55   # 1 - bm25_weight
DEFAULT_TOP_K_RETRIEVE = 50     # jumlah kandidat sebelum reranking
DEFAULT_TOP_K_FINAL    = 5      # jumlah hasil akhir
DEFAULT_RERANKER_LEN   = 400    # max chars dokumen yang dikirim ke reranker
DEFAULT_THRESHOLD      = 0.0    # skor minimum reranker (override dari kalibrasi)


def build_bm25_text(row) -> str:
    """
    Bangun teks untuk BM25. Judul di-repeat 3x, topik 2x untuk meningkatkan
    bobot kata kunci paling penting.
    """
    parts = []
    if row.get('judul'):
        parts.extend([str(row['judul'])] * 3)          # bobot tinggi
    if row.get('topik'):
        parts.extend([str(row['topik'])] * 2)           # bobot sedang
    if row.get('penulis'):
        parts.append(str(row['penulis']))
    if row.get('call_number'):
        parts.append(str(row['call_number']))
    return ' '.join(p for p in parts if p.strip())


def build_dense_text(row, desc_max_chars: int = 1500) -> str:
    """
    Bangun teks untuk dense embedding. Termasuk deskripsi (truncated)
    karena dense embedding menangkap makna semantik dari konten buku.
    """
    parts = []
    for col in ['judul', 'penulis', 'topik', 'call_number', 'penerbit']:
        val = row.get(col, '')
        if val and str(val).strip():
            parts.append(str(val).strip())
    desc = row.get('deskripsi', '') or ''
    if desc.strip():
        parts.append(desc[:desc_max_chars])
    return ' '.join(parts)


class HybridRetriever:
    """
    Hybrid retriever: BM25 + Dense + Cross-encoder Reranker.

    Parameters
    ----------
    df : pd.DataFrame
        Katalog buku dengan kolom: biblio_id, judul, penulis, topik,
        call_number, penerbit, tahun, deskripsi
    embed_model : SentenceTransformer
        Model bge-m3 (sudah di-load sebelumnya)
    reranker : CrossEncoder
        Model bge-reranker-v2-m3 (sudah di-load sebelumnya)
    bm25_weight : float
        Bobot BM25 dalam fusion (default 0.3 jika koleksi kaya deskripsi)
    dense_weight : float
        Bobot dense dalam fusion (default 0.7)
    fusion_method : str
        'weighted' (default) atau 'rrf' (Reciprocal Rank Fusion)
    embed_cache : str | None
        Path file .npy untuk cache embeddings
    reranker_threshold : float
        Skor minimum reranker untuk buku dianggap relevan
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embed_model: SentenceTransformer,
        reranker: CrossEncoder | None = None,   # None = Config C (production default)
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        fusion_method: str = 'weighted',
        embed_cache: str | None = None,
        reranker_threshold: float = DEFAULT_THRESHOLD,
    ):
        self.df        = df.reset_index(drop=True)
        self.model     = embed_model
        self.reranker  = reranker
        self.bm25_w    = bm25_weight
        self.dense_w   = dense_weight
        self.fusion    = fusion_method
        self.threshold = reranker_threshold

        # ── BM25 ──────────────────────────────────────────────────────────────
        log.info('Membangun BM25 index...')
        bm25_corpus = df.apply(build_bm25_text, axis=1).tolist()
        tokenized   = [doc.lower().split() for doc in bm25_corpus]
        self.bm25   = BM25Okapi(tokenized, k1=1.5, b=0.75)

        # ── Dense embeddings ──────────────────────────────────────────────────
        if embed_cache and os.path.exists(embed_cache):
            self.embeddings = np.load(embed_cache)
            log.info(f'Embeddings dimuat dari cache: {self.embeddings.shape}')
        else:
            log.info('Membuat dense embeddings (bisa beberapa menit)...')
            dense_corpus    = df.apply(build_dense_text, axis=1).tolist()
            self.embeddings = embed_model.encode(
                dense_corpus,
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=True,
                device='cpu',
                convert_to_numpy=True,
            )
            if embed_cache:
                np.save(embed_cache, self.embeddings)
                log.info(f'Embeddings disimpan ke: {embed_cache} {self.embeddings.shape}')

        log.info(f'HybridRetriever siap — {len(df):,} buku | fusion={fusion_method}')

    # ── Internal: retrieve top-N candidates ───────────────────────────────────

    # Kata yang menunjukkan niat user, bukan konten buku — dibuang dari BM25 query
    _BM25_STOPWORDS = {
        'skripsi', 'tugas', 'akhir', 'laporan', 'makalah', 'tesis', 'disertasi',
        'buku', 'ada', 'cari', 'cariin', 'nyari', 'dong', 'nih', 'gak', 'ga',
        'yang', 'untuk', 'buat', 'tentang', 'mengenai', 'tolong', 'minta',
    }

    def _bm25_scores(self, query: str) -> np.ndarray:
        tokens = [t for t in query.lower().split() if t not in self._BM25_STOPWORDS]
        if not tokens:
            tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        return scores / (scores.max() + 1e-9)  # normalisasi 0-1

    def _dense_scores(self, query: str) -> np.ndarray:
        q_emb = self.model.encode([query], normalize_embeddings=True)
        return (self.embeddings @ q_emb.T).flatten()

    def _weighted_fusion(self, bm25_norm, dense_scores) -> np.ndarray:
        return self.bm25_w * bm25_norm + self.dense_w * dense_scores

    def _rrf_fusion(self, bm25_norm, dense_scores, k: int = 60) -> np.ndarray:
        """Reciprocal Rank Fusion — lebih robust, tidak butuh normalisasi."""
        bm25_ranks  = np.argsort(np.argsort(-bm25_norm))   # rank dari tinggi ke rendah
        dense_ranks = np.argsort(np.argsort(-dense_scores))
        rrf = 1.0 / (k + bm25_ranks) + 1.0 / (k + dense_ranks)
        return rrf

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K_RETRIEVE) -> pd.DataFrame:
        """Ambil top_k kandidat berdasarkan fusion score (sebelum reranking)."""
        bm25_norm    = self._bm25_scores(query)
        dense_scores = self._dense_scores(query)

        if self.fusion == 'rrf':
            combined = self._rrf_fusion(bm25_norm, dense_scores)
        else:
            combined = self._weighted_fusion(bm25_norm, dense_scores)

        top_idx    = np.argsort(combined)[::-1][:top_k]
        candidates = self.df.iloc[top_idx].copy()
        candidates['_score_fusion'] = combined[top_idx]
        candidates['_bm25_norm']    = bm25_norm[top_idx]
        candidates['_dense_score']  = dense_scores[top_idx]
        return candidates.reset_index(drop=True)

    def search(
        self,
        query: str,
        top_k_final: int = DEFAULT_TOP_K_FINAL,
        apply_threshold: bool = True,
    ) -> list[dict]:
        """
        Full pipeline: retrieve → rerank → threshold → return top_k_final.
        Ini adalah Config D (dengan reranker). Dari ablation study, Config D
        lebih buruk dari Config C. Gunakan search_no_rerank() untuk production.

        Jika reranker=None (production default), otomatis fallback ke search_no_rerank.

        Returns list of dicts dengan field:
            biblio_id, judul, penulis, call_number, penerbit, tahun, deskripsi, score
        """
        if self.reranker is None:
            log.warning('search() dipanggil tapi reranker=None → fallback ke search_no_rerank()')
            return self.search_no_rerank(query, top_k=top_k_final)

        candidates = self.retrieve(query)

        # Reranking
        dense_texts = candidates.apply(build_dense_text, axis=1).tolist()
        pairs       = [(query, t[:DEFAULT_RERANKER_LEN]) for t in dense_texts]
        rerank_scores = self.reranker.predict(pairs)
        candidates    = candidates.copy()
        candidates['score'] = rerank_scores

        # Threshold filter
        if apply_threshold:
            candidates = candidates[candidates['score'] >= self.threshold]

        # Ambil top_k_final
        top = candidates.nlargest(top_k_final, 'score')

        results = []
        for _, row in top.iterrows():
            results.append({
                'biblio_id'      : int(row['biblio_id']),
                'judul'          : str(row.get('judul', '') or ''),
                'penulis'        : str(row.get('penulis', '') or ''),
                'call_number'    : str(row.get('call_number', '') or ''),
                'penerbit'       : str(row.get('penerbit', '') or ''),
                'tahun'          : str(row.get('tahun', '') or ''),
                'deskripsi'      : str(row.get('deskripsi', '') or ''),
                'total_eksemplar': int(row.get('total_eksemplar', 0) or 0),
                'tersedia'       : int(row.get('tersedia', 0) or 0),
                'score'          : float(row['score']),
            })
        return results

    def search_no_rerank(self, query: str, top_k: int = DEFAULT_TOP_K_FINAL) -> list[dict]:
        """Retrieval tanpa reranker — Config C (production)."""
        candidates = self.retrieve(query, top_k=top_k * 10)
        top = candidates.nlargest(top_k, '_score_fusion')
        return [
            {
                'biblio_id'      : int(row['biblio_id']),
                'judul'          : str(row.get('judul', '') or ''),
                'penulis'        : str(row.get('penulis', '') or ''),
                'call_number'    : str(row.get('call_number', '') or ''),
                'penerbit'       : str(row.get('penerbit', '') or ''),
                'tahun'          : str(row.get('tahun', '') or ''),
                'deskripsi'      : str(row.get('deskripsi', '') or ''),
                'total_eksemplar': int(row.get('total_eksemplar', 0) or 0),
                'tersedia'       : int(row.get('tersedia', 0) or 0),
                'score'          : float(row['_score_fusion']),
            }
            for _, row in top.iterrows()
        ]

    def search_bm25_only(self, query: str, top_k: int = DEFAULT_TOP_K_FINAL) -> list[dict]:
        """BM25 only — untuk ablation study."""
        scores  = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {
                'biblio_id'  : int(self.df.iloc[i]['biblio_id']),
                'judul'      : str(self.df.iloc[i].get('judul', '') or ''),
                'penulis'    : str(self.df.iloc[i].get('penulis', '') or ''),
                'call_number': str(self.df.iloc[i].get('call_number', '') or ''),
                'score'      : float(scores[i]),
            }
            for i in top_idx
        ]

    def search_dense_only(self, query: str, top_k: int = DEFAULT_TOP_K_FINAL) -> list[dict]:
        """Dense only — untuk ablation study."""
        scores  = self._dense_scores(query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {
                'biblio_id'  : int(self.df.iloc[i]['biblio_id']),
                'judul'      : str(self.df.iloc[i].get('judul', '') or ''),
                'penulis'    : str(self.df.iloc[i].get('penulis', '') or ''),
                'call_number': str(self.df.iloc[i].get('call_number', '') or ''),
                'score'      : float(scores[i]),
            }
            for i in top_idx
        ]
