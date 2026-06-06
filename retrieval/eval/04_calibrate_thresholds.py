"""
05_calibrate_thresholds.py — Kalibrasi threshold dari data nyata
================================================================
Mengukur distribusi skor reranker dan ChromaDB distance untuk menentukan
threshold yang optimal berdasarkan data — bukan ditebak.

Output:
  output/calibration/reranker_distribution.png
  output/calibration/hardstop_distribution.png
  output/calibration/thresholds.json   ← dibaca oleh api.py dan 06_ablation

Jalankan SETELAH:
  - 02_annotate_ground_truth.py (data/ground_truth.json harus ada)
  - 04_build_kb.py (chroma_kb harus ada)

Jalankan:
    python scripts/05_calibrate_thresholds.py
"""

import os, sys, json, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mysql.connector
import chromadb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

from retriever import HybridRetriever, build_dense_text

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'retrieval', 'output', 'calibration')
CHROMA_DIR = os.path.join(BASE_DIR, 'knowledge_base', 'chroma_db')
GT_PATH    = os.path.join(DATA_DIR, 'ground_truth.json')
RAGAS_PATH = os.path.join(DATA_DIR, 'ground_truth_ragas.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DB_CONFIG = dict(
    host     = os.getenv('DB_HOST',     'localhost'),
    user     = os.getenv('DB_USER',     'root'),
    password = os.getenv('DB_PASSWORD', ''),
    database = os.getenv('DB_NAME2',    'opacv2'),
)

TOP_K_RETRIEVE = 50   # kandidat sebelum reranking (sama dengan runtime)


# ── Load katalog ──────────────────────────────────────────────────────────────

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


# ── Kalibrasi Reranker ────────────────────────────────────────────────────────

def calibrate_reranker(retriever, ground_truth):
    """
    Kumpulkan skor reranker untuk semua (query, book) pairs dari ground truth.
    Label 1 = relevan (ada di relevant_ids), 0 = tidak relevan.
    """
    print("\n[1/2] Mengumpulkan skor reranker dari ground truth...")
    all_scores = []
    all_labels = []

    for item in ground_truth:
        query = item['query']
        rel_2 = set(item['relevant_ids']['2'])
        rel_1 = set(item['relevant_ids']['1'])
        all_relevant = rel_2 | rel_1

        # Retrieve candidates
        candidates = retriever.retrieve(query, top_k=TOP_K_RETRIEVE)
        dense_texts = candidates.apply(build_dense_text, axis=1).tolist()
        pairs       = [(query, t[:400]) for t in dense_texts]
        scores      = retriever.reranker.predict(pairs)

        for i, (_, row) in enumerate(candidates.iterrows()):
            bid = int(row['biblio_id'])
            all_scores.append(float(scores[i]))
            all_labels.append(1 if bid in all_relevant else 0)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Statistik
    rel_scores     = all_scores[all_labels == 1]
    not_rel_scores = all_scores[all_labels == 0]
    print(f"  Total pairs: {len(all_scores)} | Relevan: {all_labels.sum()} | Tidak: {(1-all_labels).sum()}")
    print(f"  Relevan    — mean: {rel_scores.mean():.3f}, std: {rel_scores.std():.3f}, "
          f"min: {rel_scores.min():.3f}, max: {rel_scores.max():.3f}")
    print(f"  Tidak rel  — mean: {not_rel_scores.mean():.3f}, std: {not_rel_scores.std():.3f}, "
          f"min: {not_rel_scores.min():.3f}, max: {not_rel_scores.max():.3f}")

    # Cari threshold optimal (maksimalkan F1)
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 500)
    f1_scores  = []
    for t in thresholds:
        preds = (all_scores >= t).astype(int)
        f1_scores.append(f1_score(all_labels, preds, zero_division=0))

    best_idx   = int(np.argmax(f1_scores))
    best_t     = float(thresholds[best_idx])
    best_f1    = float(f1_scores[best_idx])
    best_preds = (all_scores >= best_t).astype(int)
    best_prec  = float(precision_score(all_labels, best_preds, zero_division=0))
    best_rec   = float(recall_score(all_labels, best_preds, zero_division=0))

    try:
        auc = float(roc_auc_score(all_labels, all_scores))
    except Exception:
        auc = float('nan')

    print(f"\n  ✓ Threshold optimal : {best_t:.4f}")
    print(f"    F1               : {best_f1:.4f}")
    print(f"    Precision        : {best_prec:.4f}")
    print(f"    Recall           : {best_rec:.4f}")
    print(f"    ROC-AUC          : {auc:.4f}")

    # Plot distribusi
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(not_rel_scores, bins=60, alpha=0.6, color='#e74c3c', label='Tidak Relevan')
    ax.hist(rel_scores,     bins=60, alpha=0.7, color='#27ae60', label='Relevan')
    ax.axvline(best_t, color='#2c3e50', linestyle='--', linewidth=2,
               label=f'Threshold = {best_t:.3f}')
    ax.set_xlabel('Reranker Score', fontsize=12)
    ax.set_ylabel('Frekuensi', fontsize=12)
    ax.set_title('Distribusi Skor Reranker\n(Relevan vs Tidak Relevan)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # F1 vs threshold
    ax = axes[1]
    ax.plot(thresholds, f1_scores, color='#2980b9', linewidth=2)
    ax.axvline(best_t, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Optimal = {best_t:.3f} (F1={best_f1:.3f})')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score vs Threshold (Reranker)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'reranker_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot disimpan: {path}")

    return best_t, {
        'optimal_threshold': best_t,
        'f1': best_f1,
        'precision': best_prec,
        'recall': best_rec,
        'roc_auc': auc,
        'n_pairs': len(all_scores),
        'n_relevant': int(all_labels.sum()),
    }


# ── Kalibrasi Hard Stop ───────────────────────────────────────────────────────

def calibrate_hard_stop(embed_model, chroma_col, ragas_data):
    """
    Kalibrasi threshold ChromaDB distance untuk hard stop.
    In-scope: pertanyaan tentang perpustakaan (dari ragas golden)
    Out-of-scope: pertanyaan yang jelas tidak relevan
    """
    print("\n[2/2] Mengkalibrasi hard stop threshold...")

    in_scope_queries = [item['question'] for item in ragas_data
                        if item.get('query_type') != 'out_of_scope']
    oos_queries      = [item['question'] for item in ragas_data
                        if item.get('query_type') == 'out_of_scope']

    if not oos_queries:
        print("  [WARN] Tidak ada out-of-scope query di ground_truth_ragas.json")
        print("         Tambahkan query dengan query_type='out_of_scope'")
        return 0.55, {}

    all_distances = []
    all_labels    = []   # 1 = in-scope, 0 = out-of-scope

    for q in in_scope_queries:
        q_emb = embed_model.encode([q], normalize_embeddings=True).tolist()
        res   = chroma_col.query(query_embeddings=q_emb, n_results=1,
                                  include=['distances'])
        d = res['distances'][0][0] if res['distances'][0] else 1.0
        all_distances.append(d)
        all_labels.append(1)

    for q in oos_queries:
        q_emb = embed_model.encode([q], normalize_embeddings=True).tolist()
        res   = chroma_col.query(query_embeddings=q_emb, n_results=1,
                                  include=['distances'])
        d = res['distances'][0][0] if res['distances'][0] else 1.0
        all_distances.append(d)
        all_labels.append(0)

    all_distances = np.array(all_distances)
    all_labels    = np.array(all_labels)

    # Untuk hard stop: reject jika distance > threshold
    # Label: 1 = harus direject (OOS), 0 = harus diterima (in-scope)
    oos_labels = 1 - all_labels

    thresholds = np.linspace(0, 1, 500)
    f1_scores  = []
    for t in thresholds:
        preds = (all_distances >= t).astype(int)
        f1_scores.append(f1_score(oos_labels, preds, zero_division=0))

    best_idx = int(np.argmax(f1_scores))
    best_t   = float(thresholds[best_idx])
    best_f1  = float(f1_scores[best_idx])

    in_dists  = all_distances[all_labels == 1]
    oos_dists = all_distances[all_labels == 0]
    print(f"  In-scope  — mean dist: {in_dists.mean():.3f}, max: {in_dists.max():.3f}")
    print(f"  OOS       — mean dist: {oos_dists.mean():.3f}, min: {oos_dists.min():.3f}")
    print(f"  ✓ Threshold optimal: {best_t:.4f} (F1={best_f1:.4f})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(oos_dists, bins=30, alpha=0.6, color='#e74c3c', label='Out-of-Scope')
    ax.hist(in_dists,  bins=30, alpha=0.7, color='#27ae60', label='In-Scope')
    ax.axvline(best_t, color='#2c3e50', linestyle='--', linewidth=2,
               label=f'Threshold = {best_t:.3f}')
    ax.set_xlabel('ChromaDB Distance', fontsize=12)
    ax.set_ylabel('Frekuensi', fontsize=12)
    ax.set_title('Distribusi Distance: In-Scope vs Out-of-Scope', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(thresholds, f1_scores, color='#8e44ad', linewidth=2)
    ax.axvline(best_t, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Optimal = {best_t:.3f} (F1={best_f1:.3f})')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score vs Threshold (Hard Stop)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'hardstop_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot disimpan: {path}")

    return best_t, {
        'optimal_threshold': best_t,
        'f1': best_f1,
        'n_in_scope': int(all_labels.sum()),
        'n_out_of_scope': int((1 - all_labels).sum()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("KALIBRASI THRESHOLD — RAG Chatbot Perpustakaan PNJ")
    print("=" * 65)

    # Validasi files
    for path, name in [(GT_PATH, 'ground_truth.json'), (RAGAS_PATH, 'ground_truth_ragas.json')]:
        if not os.path.exists(path):
            print(f"\n[ERROR] {name} tidak ditemukan di {path}")
            print("  Jalankan 02_annotate_ground_truth.py terlebih dahulu.")
            sys.exit(1)

    with open(GT_PATH, encoding='utf-8') as f:
        ground_truth = json.load(f)
    with open(RAGAS_PATH, encoding='utf-8') as f:
        ragas_data = json.load(f)

    print(f"\nGround truth book: {len(ground_truth)} queries")
    print(f"Ground truth RAGAS: {len(ragas_data)} Q&A pairs")

    # Load models
    print("\nLoading models (bisa beberapa menit pertama kali)...")
    embed_model = SentenceTransformer('BAAI/bge-m3')
    reranker    = CrossEncoder('BAAI/bge-reranker-v2-m3')

    # Load catalog & build retriever
    df       = load_catalog()
    cache    = os.path.join(BASE_DIR, 'output', 'embeddings.npy')
    retriever = HybridRetriever(
        df, embed_model, reranker,
        embed_cache=cache,
        reranker_threshold=0.0,  # threshold sementara untuk kalibrasi
    )

    # Load ChromaDB
    if not os.path.exists(CHROMA_DIR):
        print("\n[ERROR] ChromaDB tidak ditemukan. Jalankan 04_build_kb.py dulu.")
        sys.exit(1)
    client    = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_col = client.get_collection('kb_perpustakaan_pnj')

    # Kalibrasi
    reranker_threshold, reranker_stats = calibrate_reranker(retriever, ground_truth)
    hardstop_threshold, hardstop_stats = calibrate_hard_stop(embed_model, chroma_col, ragas_data)

    # Simpan hasil
    result = {
        'reranker_threshold' : reranker_threshold,
        'hardstop_threshold' : hardstop_threshold,
        'reranker_stats'     : reranker_stats,
        'hardstop_stats'     : hardstop_stats,
    }
    out_path = os.path.join(OUTPUT_DIR, 'thresholds.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*65}")
    print("HASIL KALIBRASI:")
    print(f"  Reranker threshold : {reranker_threshold:.4f}")
    print(f"  Hard stop threshold: {hardstop_threshold:.4f}")
    print(f"\n  Disimpan ke: {out_path}")
    print(f"  Plot tersimpan di: {OUTPUT_DIR}/")
    print(f"\nUpdate api.py:")
    print(f"  RERANKER_THRESHOLD = {reranker_threshold:.4f}")
    print(f"  KB_DIST_THRESHOLD  = {hardstop_threshold:.4f}")
    print("=" * 65)


if __name__ == '__main__':
    main()
