"""
07_evaluate.py — Full evaluation: MRR, Precision, Hit, NDCG + breakdown
========================================================================
Evaluasi sistem final menggunakan Config C (Hybrid BM25+Dense, TANPA reranker).

Config C dipilih berdasarkan ablation study (scripts/06_ablation_study.py):
  Config A — BM25 only        : NDCG@5=0.7377, Hit@5=95.0%
  Config B — Dense only       : NDCG@5=0.4865, Hit@5=87.5%
  Config C — Hybrid no rerank : NDCG@5=0.7682, Hit@5=95.0%  ← TERPILIH (ini script)
  Config D — Hybrid + rerank  : NDCG@5=0.4571, Hit@5=87.5%  (lebih buruk, TIDAK dipakai)

Output:
  output/eval_results.csv          — per-query scores
  output/eval_summary.txt          — ringkasan untuk BAB IV
  output/eval_by_topic.csv         — breakdown per topic
  output/eval_by_difficulty.csv    — breakdown per difficulty
  output/eval_error_analysis.csv   — query yang hit@5=0 (untuk error analysis)
  output/eval_chart_topic.png      — visualisasi breakdown topic

Jalankan:
    python scripts/07_evaluate.py
"""

import os, sys, json, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
from sentence_transformers import SentenceTransformer
# CrossEncoder tidak diimport — Config C tidak menggunakan reranker
from dotenv import load_dotenv

from retriever import HybridRetriever

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'output')
GT_PATH     = os.path.join(DATA_DIR, 'ground_truth.json')
CAL_PATH    = os.path.join(OUTPUT_DIR, 'calibration', 'thresholds.json')
EMBED_CACHE = os.path.join(OUTPUT_DIR, 'embeddings.npy')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DB_CONFIG = dict(
    host     = os.getenv('DB_HOST',     'localhost'),
    user     = os.getenv('DB_USER',     'root'),
    password = os.getenv('DB_PASSWORD', ''),
    database = os.getenv('DB_NAME2',    'opacv2'),
)


# ── Metrik ────────────────────────────────────────────────────────────────────

def compute_metrics(retrieved_ids, relevant_2, relevant_1, k_values=(1, 5, 10)):
    all_relevant = relevant_2 | relevant_1
    rel_scores   = {bid: 2 for bid in relevant_2}
    rel_scores.update({bid: 1 for bid in relevant_1 if bid not in relevant_2})

    metrics = {}
    for k in k_values:
        top_k = retrieved_ids[:k]
        hits  = sum(1 for bid in top_k if bid in all_relevant)

        # MRR
        mrr = 0.0
        for rank, bid in enumerate(top_k, 1):
            if bid in all_relevant:
                mrr = 1.0 / rank
                break

        # NDCG
        dcg       = sum(rel_scores.get(bid, 0) / np.log2(r + 2)
                        for r, bid in enumerate(top_k))
        ideal_rels = sorted(rel_scores.values(), reverse=True)[:k]
        ideal_dcg  = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rels))
        ndcg       = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        metrics[f'mrr@{k}']  = round(mrr, 4)
        metrics[f'p@{k}']    = round(hits / k, 4)
        metrics[f'hit@{k}']  = 1 if hits > 0 else 0
        metrics[f'ndcg@{k}'] = round(ndcg, 4)

    return metrics


def bootstrap_ci(values, n_boot=2000, ci=0.95):
    arr   = np.array(values)
    means = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, 100 - (1 - ci) / 2 * 100)
    return float(np.mean(arr)), float(lo), float(hi)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_catalog():
    conn = mysql.connector.connect(**DB_CONFIG)
    df   = pd.read_sql("""
        SELECT b.biblio_id, b.title AS judul, b.call_number,
               b.notes AS deskripsi, b.publish_year AS tahun,
               GROUP_CONCAT(DISTINCT a.author_name  SEPARATOR ', ') AS penulis,
               GROUP_CONCAT(DISTINCT t.topic        SEPARATOR ', ') AS topik,
               p.publisher_name AS penerbit
        FROM biblio b
        JOIN mst_gmd g ON b.gmd_id = g.gmd_id AND g.gmd_code = 'TE'
        LEFT JOIN biblio_author ba ON b.biblio_id = ba.biblio_id
        LEFT JOIN mst_author a    ON ba.author_id = a.author_id
        LEFT JOIN biblio_topic bt ON b.biblio_id = bt.biblio_id
        LEFT JOIN mst_topic t     ON bt.topic_id  = t.topic_id
        LEFT JOIN mst_publisher p ON b.publisher_id = p.publisher_id
        GROUP BY b.biblio_id
    """, conn)
    conn.close()
    for col in ['deskripsi', 'penulis', 'topik', 'penerbit']:
        df[col] = df[col].fillna('').astype(str)
    return df.reset_index(drop=True)


# ── Main evaluation ───────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("FULL EVALUATION — RAG Chatbot Perpustakaan PNJ")
    print("=" * 65)

    if not os.path.exists(GT_PATH):
        print(f"[ERROR] {GT_PATH} tidak ditemukan.")
        sys.exit(1)

    with open(GT_PATH, encoding='utf-8') as f:
        ground_truth = json.load(f)

    # Load threshold
    if os.path.exists(CAL_PATH):
        with open(CAL_PATH) as f:
            cal = json.load(f)
        threshold = cal['reranker_threshold']
        print(f"  Threshold dari kalibrasi: {threshold:.4f}")
    else:
        threshold = 0.0
        print(f"  [WARN] Threshold default: {threshold}")

    print(f"  Jumlah queries: {len(ground_truth)}")

    # Load
    print("\nLoading models...")
    embed_model = SentenceTransformer('BAAI/bge-m3')
    # reranker TIDAK dimuat — Config C tidak memerlukan CrossEncoder
    df          = load_catalog()
    retriever   = HybridRetriever(
        df, embed_model, reranker=None,   # Config C: tanpa reranker
        bm25_weight=0.45, dense_weight=0.55,   # optimum grid search (grid_search_weights.py)
        embed_cache=EMBED_CACHE,
        reranker_threshold=threshold,
    )

    # Evaluasi
    print("\nMenjalankan evaluasi...")
    rows = []
    for i, item in enumerate(ground_truth, 1):
        query   = item['query']
        rel_2   = set(item['relevant_ids']['2'])
        rel_1   = set(item['relevant_ids']['1'])
        # Config C: Hybrid BM25+Dense tanpa reranker (konfigurasi terpilih)
        results = retriever.search_no_rerank(query, top_k=10)
        ret_ids = [r['biblio_id'] for r in results]
        metrics = compute_metrics(ret_ids, rel_2, rel_1, k_values=(1, 5, 10))

        # Top hasil untuk error analysis
        top_titles = [r['judul'][:40] for r in results[:5]]

        rows.append({
            'query_id'   : item['query_id'],
            'query'      : query,
            'topic'      : item.get('topic', ''),
            'difficulty' : item.get('difficulty', ''),
            'n_relevant' : len(rel_2) + len(rel_1),
            'n_retrieved': len(results),
            **metrics,
            'top_results': ' | '.join(top_titles),
        })
        print(f"  [{i:02d}/{len(ground_truth)}] MRR@5={metrics['mrr@5']:.3f}  "
              f"NDCG@5={metrics['ndcg@5']:.3f}  Hit@5={metrics['hit@5']}  "
              f"  {query[:45]}")

    df_results = pd.DataFrame(rows)
    results_path = os.path.join(OUTPUT_DIR, 'eval_results.csv')
    df_results.to_csv(results_path, index=False)

    # ── Ringkasan keseluruhan ─────────────────────────────────────────────────
    metric_cols = ['mrr@1', 'mrr@5', 'p@5', 'hit@5', 'ndcg@5', 'hit@10', 'ndcg@10']
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("HASIL EVALUASI — Chatbot Perpustakaan PNJ (v3)")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Jumlah query    : {len(ground_truth)}")
    summary_lines.append(f"Threshold       : {threshold:.4f} (dari kalibrasi)")
    summary_lines.append(f"Embedding model : BAAI/bge-m3")
    summary_lines.append(f"Konfigurasi     : Config C — Hybrid BM25+Dense (tanpa reranker, terpilih dari ablation)")
    summary_lines.append(f"Fusion          : BM25({retriever.bm25_w}) + Dense({retriever.dense_w})")
    summary_lines.append("")
    summary_lines.append("METRIK (95% Bootstrap CI):")

    for col in metric_cols:
        mean, lo, hi = bootstrap_ci(df_results[col].tolist())
        summary_lines.append(f"  {col.upper():<12}: {mean:.4f}  (CI: {lo:.4f} – {hi:.4f})")

    # ── Breakdown per topic ───────────────────────────────────────────────────
    summary_lines.append("")
    summary_lines.append("BREAKDOWN PER TOPIC:")
    topic_group = df_results.groupby('topic')[metric_cols].mean().round(4)
    summary_lines.append(topic_group.to_string())

    df_topic = df_results.groupby('topic')[metric_cols].mean().round(4).reset_index()
    df_topic.to_csv(os.path.join(OUTPUT_DIR, 'eval_by_topic.csv'), index=False)

    # ── Breakdown per difficulty ──────────────────────────────────────────────
    summary_lines.append("")
    summary_lines.append("BREAKDOWN PER DIFFICULTY:")
    diff_group = df_results.groupby('difficulty')[metric_cols].mean().round(4)
    summary_lines.append(diff_group.to_string())

    df_diff = df_results.groupby('difficulty')[metric_cols].mean().round(4).reset_index()
    df_diff.to_csv(os.path.join(OUTPUT_DIR, 'eval_by_difficulty.csv'), index=False)

    # ── Error analysis: query yang gagal (hit@5 = 0) ─────────────────────────
    df_errors = df_results[df_results['hit@5'] == 0][
        ['query_id', 'query', 'topic', 'difficulty', 'n_relevant', 'top_results']
    ]
    if len(df_errors) > 0:
        summary_lines.append(f"\nERROR ANALYSIS ({len(df_errors)} query dengan Hit@5=0):")
        for _, row in df_errors.iterrows():
            summary_lines.append(f"  [{row['query_id']}] {row['query']}")
            summary_lines.append(f"    Top results: {row['top_results']}")
        df_errors.to_csv(os.path.join(OUTPUT_DIR, 'eval_error_analysis.csv'), index=False)

    summary_lines.append("=" * 60)

    summary_text = '\n'.join(summary_lines)
    print(f"\n{summary_text}")

    summary_path = os.path.join(OUTPUT_DIR, 'eval_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    # ── Visualisasi breakdown topic ───────────────────────────────────────────
    if len(df_topic) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        topics = df_topic['topic'].tolist()
        x      = np.arange(len(topics))

        for ax, (metric, label) in zip(axes, [('ndcg@5', 'NDCG@5'), ('hit@5', 'Hit@5')]):
            vals   = df_topic[metric].tolist()
            colors = ['#27ae60' if v >= 0.7 else '#f39c12' if v >= 0.4 else '#e74c3c'
                      for v in vals]
            bars   = ax.bar(x, vals, color=colors, alpha=0.85, zorder=3)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(topics, rotation=30, ha='right', fontsize=9)
            ax.set_ylabel(label, fontsize=12)
            ax.set_title(f'{label} per Topic', fontsize=12)
            ax.set_ylim(0, 1.15)
            ax.yaxis.grid(True, alpha=0.3, zorder=0)
            ax.set_axisbelow(True)

        plt.tight_layout()
        chart_path = os.path.join(OUTPUT_DIR, 'eval_chart_topic.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nChart: {chart_path}")

    print(f"\nSemua output disimpan di: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
