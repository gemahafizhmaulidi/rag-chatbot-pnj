"""
06_ablation_study.py — Ablation study: kontribusi tiap komponen retrieval
=========================================================================
Membandingkan 4 konfigurasi untuk menunjukkan kontribusi masing-masing komponen:

  Config A : BM25 Only
  Config B : Dense Only (bge-m3)
  Config C : BM25 + Dense Hybrid (tanpa reranker)  ← SISTEM FINAL (produksi)
  Config D : BM25 + Dense + Reranker  (terbukti menurunkan NDCG → TIDAK dipakai, lihat diag_reranker.py)

Metrik: MRR@5, P@5, Hit@5, NDCG@5, Hit@10, NDCG@10

Output:
  output/ablation_results.csv    — per-query per-config
  output/ablation_summary.csv    — ringkasan (tabel untuk BAB IV)
  output/ablation_chart.png      — bar chart untuk BAB IV

Jalankan:
    python scripts/06_ablation_study.py
"""

import os, sys, json, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
from sentence_transformers import SentenceTransformer, CrossEncoder
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

def compute_metrics(retrieved_ids: list, relevant_2: set, relevant_1: set,
                    k_values=(5, 10)) -> dict:
    """
    Hitung MRR, Precision, Hit, NDCG untuk berbagai nilai k.
    Relevansi: skor 2 = sangat relevan, skor 1 = cukup relevan, 0 = tidak relevan.
    """
    all_relevant = relevant_2 | relevant_1
    rel_scores   = {bid: 2 for bid in relevant_2}
    rel_scores.update({bid: 1 for bid in relevant_1 if bid not in relevant_2})

    metrics = {}
    for k in k_values:
        top_k = retrieved_ids[:k]

        # MRR@k
        mrr = 0.0
        for rank, bid in enumerate(top_k, 1):
            if bid in all_relevant:
                mrr = 1.0 / rank
                break

        # Precision@k
        hits   = sum(1 for bid in top_k if bid in all_relevant)
        prec   = hits / k

        # Hit@k (binary: ada minimal 1 relevan)
        hit    = 1 if hits > 0 else 0

        # NDCG@k
        dcg       = sum(rel_scores.get(bid, 0) / np.log2(r + 2)
                        for r, bid in enumerate(top_k))
        ideal_rels = sorted(rel_scores.values(), reverse=True)[:k]
        ideal_dcg  = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rels))
        ndcg       = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        metrics[f'mrr@{k}']    = round(mrr, 4)
        metrics[f'p@{k}']      = round(prec, 4)
        metrics[f'hit@{k}']    = hit
        metrics[f'ndcg@{k}']   = round(ndcg, 4)

    return metrics


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval."""
    means = [np.mean(np.random.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(np.mean(values)), float(lo), float(hi)


# ── Load data ─────────────────────────────────────────────────────────────────

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


# ── Run ablation ──────────────────────────────────────────────────────────────

def run_ablation(retriever, ground_truth, k_values=(5, 10)):
    configs = {
        'A_BM25_only'        : lambda q: retriever.search_bm25_only(q, top_k=max(k_values)),
        'B_Dense_only'       : lambda q: retriever.search_dense_only(q, top_k=max(k_values)),
        'C_Hybrid_no_rerank' : lambda q: retriever.search_no_rerank(q, top_k=max(k_values)),
        'D_Hybrid_rerank'    : lambda q: retriever.search(q, top_k_final=max(k_values),
                                                           apply_threshold=False),
    }

    rows = []
    n    = len(ground_truth)

    for cfg_name, search_fn in configs.items():
        print(f"\n  Config {cfg_name}:")
        for i, item in enumerate(ground_truth, 1):
            query      = item['query']
            rel_2      = set(item['relevant_ids']['2'])
            rel_1      = set(item['relevant_ids']['1'])
            results    = search_fn(query)
            ret_ids    = [r['biblio_id'] for r in results]
            metrics    = compute_metrics(ret_ids, rel_2, rel_1, k_values)

            rows.append({
                'config'    : cfg_name,
                'query_id'  : item['query_id'],
                'query'     : query,
                'topic'     : item.get('topic', ''),
                'difficulty': item.get('difficulty', ''),
                **metrics,
            })
            print(f"    [{i:02d}/{n}] {query[:50]:<50} "
                  f"MRR@5={metrics['mrr@5']:.3f}  NDCG@5={metrics['ndcg@5']:.3f}", end='\r')
        print()

    return pd.DataFrame(rows)


# ── Summary & visualization ───────────────────────────────────────────────────

def summarize(df_results, k_values=(5, 10)):
    metric_cols = [f'{m}@{k}' for m in ['mrr', 'p', 'hit', 'ndcg'] for k in k_values]
    summary_rows = []

    for cfg in df_results['config'].unique():
        sub  = df_results[df_results['config'] == cfg]
        row  = {'config': cfg, 'n_queries': len(sub)}
        for col in metric_cols:
            mean, lo, hi = bootstrap_ci(sub[col].tolist())
            row[col]         = round(mean, 4)
            row[f'{col}_lo'] = round(lo, 4)
            row[f'{col}_hi'] = round(hi, 4)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def plot_ablation(df_summary, k=5):
    metrics    = [f'mrr@{k}', f'p@{k}', f'hit@{k}', f'ndcg@{k}']
    labels     = [f'MRR@{k}', f'P@{k}', f'Hit@{k}', f'NDCG@{k}']
    configs    = df_summary['config'].tolist()
    cfg_labels = ['A: BM25 Only', 'B: Dense Only', 'C: Hybrid\n(no rerank)', 'D: Hybrid\n+Reranker']
    colors     = ['#95a5a6', '#3498db', '#f39c12', '#27ae60']

    x     = np.arange(len(metrics))
    width = 0.18
    n_cfg = len(configs)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (cfg, label, color) in enumerate(zip(configs, cfg_labels, colors)):
        row    = df_summary[df_summary['config'] == cfg].iloc[0]
        vals   = [row[m] for m in metrics]
        lo_err = [row[m] - row[f'{m}_lo'] for m in metrics]
        hi_err = [row[f'{m}_hi'] - row[m] for m in metrics]
        offsets = x + (i - n_cfg / 2 + 0.5) * width
        bars = ax.bar(offsets, vals, width, label=label, color=color, alpha=0.88,
                      zorder=3)
        ax.errorbar(offsets, vals, yerr=[lo_err, hi_err], fmt='none',
                    color='#2c3e50', capsize=3, linewidth=1.5, zorder=4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Ablation Study — Kontribusi Komponen Retrieval (k={k})\n'
                 f'Error bars: 95% Bootstrap CI', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f'ablation_chart_k{k}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart disimpan: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("ABLATION STUDY — RAG Chatbot Perpustakaan PNJ")
    print("=" * 65)

    if not os.path.exists(GT_PATH):
        print(f"[ERROR] {GT_PATH} tidak ditemukan. Jalankan 02_annotate_ground_truth.py dulu.")
        sys.exit(1)

    with open(GT_PATH, encoding='utf-8') as f:
        ground_truth = json.load(f)

    # Load threshold yang sudah dikalibrasi
    if os.path.exists(CAL_PATH):
        with open(CAL_PATH) as f:
            cal = json.load(f)
        reranker_threshold = cal['reranker_threshold']
        print(f"  Menggunakan threshold yang dikalibrasi: {reranker_threshold:.4f}")
    else:
        reranker_threshold = 0.0
        print(f"  [WARN] Threshold belum dikalibrasi. Gunakan default {reranker_threshold}")

    print(f"\n  {len(ground_truth)} queries dari ground truth")

    # Load models
    print("\nLoading models...")
    embed_model = SentenceTransformer('BAAI/bge-m3')
    reranker    = CrossEncoder('BAAI/bge-reranker-v2-m3')

    # Load catalog & build retriever
    print("Loading katalog dan membangun retriever...")
    df        = load_catalog()
    retriever = HybridRetriever(
        df, embed_model, reranker,
        bm25_weight=0.45, dense_weight=0.55,   # optimum grid search (grid_search_weights.py)
        embed_cache=EMBED_CACHE,
        reranker_threshold=reranker_threshold,
    )

    # Run ablation
    print("\nMenjalankan ablation study (4 konfigurasi × semua query)...")
    df_results = run_ablation(retriever, ground_truth)

    # Save per-query
    results_path = os.path.join(OUTPUT_DIR, 'ablation_results.csv')
    df_results.to_csv(results_path, index=False)
    print(f"\n  Per-query results: {results_path}")

    # Summary
    df_summary = summarize(df_results)
    summary_path = os.path.join(OUTPUT_DIR, 'ablation_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"  Summary: {summary_path}")

    # Print tabel ringkasan
    print(f"\n{'='*65}")
    print("TABEL ABLATION (untuk BAB IV)")
    print(f"{'─'*65}")
    display_cols = ['config', 'mrr@5', 'p@5', 'hit@5', 'ndcg@5', 'hit@10', 'ndcg@10']
    for col in display_cols:
        if col not in df_summary.columns:
            display_cols.remove(col)
    print(df_summary[display_cols].to_string(index=False))

    # Plot
    print("\nMembuat chart...")
    plot_ablation(df_summary, k=5)
    plot_ablation(df_summary, k=10)

    print(f"\n{'='*65}")
    print("Ablation study selesai.")
    print(f"  Results : {results_path}")
    print(f"  Summary : {summary_path}")
    print(f"  Charts  : {OUTPUT_DIR}/ablation_chart_k5.png")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
