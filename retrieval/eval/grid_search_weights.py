"""
grid_search_weights.py — Grid search bobot fusi BM25 vs Dense (yang SEHARUSNYA ada)
==================================================================================
Menjawab pertanyaan sidang: "kenapa BM25(0.3)+Dense(0.7), bukan 0.5/0.5?"
Sweep bm25_weight 0.0..1.0, ukur NDCG@5 / MRR@5 / Hit@5 pada ground_truth (graded).
Skor BM25 & dense di-precompute sekali per query lalu di-reweight (cepat).

Pakai: python3.10 scripts/grid_search_weights.py
Output: output/grid_search_weights.csv + ringkasan
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import mysql.connector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from retriever import HybridRetriever

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_PATH = os.path.join(BASE, 'data', 'ground_truth.json')
EMBED_CACHE = os.path.join(BASE, 'output', 'embeddings.npy')
OUT_CSV = os.path.join(BASE, 'output', 'grid_search_weights.csv')
DB = dict(host=os.getenv('DB_HOST', 'localhost'), user=os.getenv('DB_USER', 'root'),
          password=os.getenv('DB_PASSWORD', ''), database=os.getenv('DB_NAME2', 'opacv2'))


def load_catalog():
    conn = mysql.connector.connect(**DB)
    df = pd.read_sql("""
        SELECT b.biblio_id, b.title AS judul, b.call_number, b.notes AS deskripsi,
               b.publish_year AS tahun,
               GROUP_CONCAT(DISTINCT a.author_name SEPARATOR ', ') AS penulis,
               GROUP_CONCAT(DISTINCT t.topic SEPARATOR ', ') AS topik,
               p.publisher_name AS penerbit
        FROM biblio b JOIN mst_gmd g ON b.gmd_id=g.gmd_id AND g.gmd_code='TE'
        LEFT JOIN biblio_author ba ON b.biblio_id=ba.biblio_id
        LEFT JOIN mst_author a ON ba.author_id=a.author_id
        LEFT JOIN biblio_topic bt ON b.biblio_id=bt.biblio_id
        LEFT JOIN mst_topic t ON bt.topic_id=t.topic_id
        LEFT JOIN mst_publisher p ON b.publisher_id=p.publisher_id
        GROUP BY b.biblio_id""", conn)
    conn.close()
    for c in ['deskripsi', 'penulis', 'topik', 'penerbit']:
        df[c] = df[c].fillna('').astype(str)
    return df.reset_index(drop=True)


def metrics(ranked_ids, rel2, rel1, k=5):
    allrel = rel2 | rel1
    rs = {b: 2 for b in rel2}; rs.update({b: 1 for b in rel1 if b not in rel2})
    topk = ranked_ids[:k]
    mrr = next((1.0 / r for r, b in enumerate(topk, 1) if b in allrel), 0.0)
    hit = 1 if any(b in allrel for b in topk) else 0
    dcg = sum(rs.get(b, 0) / np.log2(r + 2) for r, b in enumerate(topk))
    idcg = sum(v / np.log2(i + 2) for i, v in enumerate(sorted(rs.values(), reverse=True)[:k]))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return mrr, hit, ndcg


print('Loading model + catalog + embeddings...')
model = SentenceTransformer('BAAI/bge-m3')
df = load_catalog()
r = HybridRetriever(df, model, reranker=None, embed_cache=EMBED_CACHE)
gt = json.load(open(GT_PATH, encoding='utf-8'))
ids = df['biblio_id'].values
print(f'{len(df)} buku, {len(gt)} query GT\n')

# Precompute skor BM25(norm) & dense per query (sekali saja)
pre = []
for item in gt:
    q = item['query']
    pre.append((r._bm25_scores(q), r._dense_scores(q),
                set(item['relevant_ids']['2']), set(item['relevant_ids']['1'])))

rows = []
for w in [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]:
    Ms, Hs, Ns = [], [], []
    for bm, dn, rel2, rel1 in pre:
        combined = w * bm + (1 - w) * dn
        order = np.argsort(combined)[::-1][:10]
        ranked = [int(ids[i]) for i in order]
        m, h, n = metrics(ranked, rel2, rel1, k=5)
        Ms.append(m); Hs.append(h); Ns.append(n)
    rows.append({'bm25_w': w, 'dense_w': round(1 - w, 2),
                 'mrr@5': round(np.mean(Ms), 4), 'hit@5': round(np.mean(Hs), 4),
                 'ndcg@5': round(np.mean(Ns), 4)})

res = pd.DataFrame(rows)
res.to_csv(OUT_CSV, index=False)

print('bm25_w dense_w  MRR@5  Hit@5  NDCG@5   <- baris ditandai = produksi (0.3/0.7) & optimal')
best_ndcg = res['ndcg@5'].idxmax(); best_mrr = res['mrr@5'].idxmax()
for i, row in res.iterrows():
    tag = ''
    if abs(row['bm25_w'] - 0.3) < 1e-9: tag += '  <PRODUKSI'
    if i == best_ndcg: tag += '  <BEST-NDCG'
    if i == best_mrr and best_mrr != best_ndcg: tag += '  <BEST-MRR'
    print(f"  {row['bm25_w']:.2f}   {row['dense_w']:.2f}   {row['mrr@5']:.4f} {row['hit@5']:.4f} {row['ndcg@5']:.4f}{tag}")

print(f"\nOptimal NDCG@5: bm25_w={res.loc[best_ndcg,'bm25_w']:.2f} -> NDCG@5={res.loc[best_ndcg,'ndcg@5']:.4f}")
print(f"Produksi (0.3): NDCG@5={res.loc[res['bm25_w'].sub(0.3).abs().idxmin(),'ndcg@5']:.4f}")
print(f"CSV: {OUT_CSV}")
