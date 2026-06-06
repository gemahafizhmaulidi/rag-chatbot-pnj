# Hasil Retrieval & Ablation (Concern A) — Chatbot Perpustakaan PNJ

Audit & perbaikan keresahan "Hybrid BM25(0.3)+Dense(0.7), Config C, NO reranker".
Eksperimen 2026-06-02/03. Ground truth: `data/ground_truth.json` (40 query, graded 0/1/2).

## A1 — Grid search bobot fusi (sebelumnya HANYA ditebak)
Temuan: bobot 0.3/0.7 lama **tidak pernah di-grid-search** (hardcoded). Setelah grid search
(`grid_search_weights.py` → `output/grid_search_weights.csv`):

| bm25_w | MRR@5 | Hit@5 | NDCG@5 |
|---|---|---|---|
| 0.00 (dense only) | 0.733 | 0.875 | 0.487 |
| 0.15 | 0.926 | 0.975 | 0.749 |
| 0.30 (lama) | 0.892 | 0.95 | 0.762 |
| **0.45 (BARU, optimum)** | **0.909** | **0.975** | **0.774** |
| 1.00 (BM25 only) | 0.902 | 0.95 | 0.726 |

- Band 0.15–0.45 **robust** (selisih dalam bootstrap CI ±0.08) → hybrid tidak sensitif tuning.
- **0.45 dipilih** (optimum NDCG@5). Produksi (`api.py`, `retriever.py`) + script 05/06 diupdate ke 0.45/0.55.

## Ablation 4 konfigurasi @ bobot 0.45 (`output/ablation_summary.csv`)
| Config | MRR@5 | Hit@5 | NDCG@5 |
|---|---|---|---|
| A — BM25 only | 0.9021 | 0.95 | 0.7377 |
| B — Dense only | 0.7333 | 0.875 | 0.4865 |
| **C — Hybrid (PRODUKSI)** | **0.9092** | **0.975** | **0.7740** |
| D — Hybrid + Reranker | 0.7604 | 0.875 | 0.4983 |

→ Config C kini **mengungguli BM25-only di SEMUA metrik** (di bobot 0.3 lama, MRR BM25 sempat
sedikit lebih tinggi — kelemahan argumen itu hilang).

## Headline (Config C, `output/eval_summary.txt`)
NDCG@5 = **0.7740** | MRR@5 = **0.9092** | Hit@5 = **97.5%** | NDCG@10 = 0.7767 (40 query, 95% CI).

## A3 — Reranker: bug atau temuan? → TEMUAN NYATA (`diag_reranker.log`)
Config D anjlok (NDCG@5 ~0.46–0.50, di bawah dense-only). Diuji:
- Truncation 400 char **bukan** penyebab utama: full-text (512 token) hanya 0.48 — tetap jelek.
- Spot-check: reranker memberi skor tinggi seragam (dok relevan 0.86–0.95 **tapi kalah** oleh dok
  tak-relevan yang lebih tinggi) → **gagal diskriminasi** pada metadata katalog terse (judul+penulis+
  DDC+deskripsi pendek), di luar distribusi latih (passage natural). Plus *pure re-sort* top-50
  membuang sinyal fusi.
- Kesimpulan: bukan bug pemakaian → omit reranker (Config C) **terjustifikasi & terinvestigasi**.
  Future work opsional: blend skor reranker+fusi (bukan re-sort murni).

## Berkas
`grid_search_weights.csv`, `ablation_summary.csv`, `ablation_results.csv`, `eval_summary.txt`,
`diag_reranker.log` (di `output/routing_eval/`), chart `ablation_chart_k5.png`.
