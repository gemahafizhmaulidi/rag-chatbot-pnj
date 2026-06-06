"""
09b_enrichment_eval_sound.py — Evaluasi dampak enrichment (METODOLOGI DIPERBAIKI)
=================================================================================
Mengganti 09_before_after_enrichment.py yang cacat:
  - GT lama SIRKULAR (relevan = deskripsi enrichment mengandung keyword).
  - Lama DENSE-ONLY, bukan Config C produksi.

Versi sound:
  - GT INDEPENDEN: data/ground_truth.json (40 query, relevansi anotasi manual 0/1/2,
    TIDAK diturunkan dari deskripsi).
  - Retrieval = Config C HYBRID produksi (BM25 0.45 + Dense 0.55). BM25 identik before/after
    (BM25 tak pakai deskripsi); HANYA matriks dense yang berganti (before=opac_original tanpa
    deskripsi, after=opacv2 dengan deskripsi enrichment).
  - Metrik: MRR@5, Hit@5, NDCG@5 (graded). Bandingkan before vs after pada query yang SAMA.

Pakai: python3.10 scripts/09b_enrichment_eval_sound.py
Output: output/enrichment_eval_sound.csv + ringkasan
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
EMB_BEFORE = os.path.join(BASE, 'output', 'embeddings_before.npy')
EMB_AFTER  = os.path.join(BASE, 'output', 'embeddings_after.npy')
OUT_CSV = os.path.join(BASE, 'output', 'enrichment_eval_sound.csv')
BM25_W, DENSE_W = 0.45, 0.55
DB = dict(host=os.getenv('DB_HOST', 'localhost'), user=os.getenv('DB_USER', 'root'),
          password=os.getenv('DB_PASSWORD', ''))


def load_catalog(dbname, order=True):
    conn = mysql.connector.connect(**DB, database=dbname)
    df = pd.read_sql(f"""
        SELECT b.biblio_id, b.title AS judul, b.call_number, b.notes AS deskripsi,
               GROUP_CONCAT(DISTINCT a.author_name SEPARATOR ', ') AS penulis,
               GROUP_CONCAT(DISTINCT t.topic SEPARATOR ', ') AS topik,
               p.publisher_name AS penerbit
        FROM biblio b JOIN mst_gmd g ON b.gmd_id=g.gmd_id AND g.gmd_code='TE'
        LEFT JOIN biblio_author ba ON b.biblio_id=ba.biblio_id
        LEFT JOIN mst_author a ON ba.author_id=a.author_id
        LEFT JOIN biblio_topic bt ON b.biblio_id=bt.biblio_id
        LEFT JOIN mst_topic t ON bt.topic_id=t.topic_id
        LEFT JOIN mst_publisher p ON b.publisher_id=p.publisher_id
        GROUP BY b.biblio_id {'ORDER BY b.biblio_id' if order else ''}""", conn)
    conn.close()
    for c in ['deskripsi', 'penulis', 'topik', 'penerbit']:
        df[c] = df[c].fillna('').astype(str)
    return df.sort_values('biblio_id').reset_index(drop=True)


def metrics(ranked, rel2, rel1, k=5):
    allrel = rel2 | rel1
    rs = {b: 2 for b in rel2}; rs.update({b: 1 for b in rel1 if b not in rel2})
    topk = ranked[:k]
    mrr = next((1.0 / r for r, b in enumerate(topk, 1) if b in allrel), 0.0)
    hit = 1 if any(b in allrel for b in topk) else 0
    dcg = sum(rs.get(b, 0) / np.log2(r + 2) for r, b in enumerate(topk))
    idcg = sum(v / np.log2(i + 2) for i, v in enumerate(sorted(rs.values(), reverse=True)[:k]))
    return mrr, hit, (dcg / idcg if idcg else 0.0)


print('Loading catalog (opacv2, sorted) + bge-m3 + embeddings...')
dbname_after = os.getenv('DB_NAME2', 'opacv2')
dbname_before = os.getenv('DB_NAME_ORIGINAL', 'opac_original')
df = load_catalog(dbname_after)          # untuk BM25 + id mapping (judul/topik sama before/after)
df_before = load_catalog(dbname_before)
assert (df['biblio_id'].values == df_before['biblio_id'].values).all(), 'biblio_id tidak sejajar!'

emb_before = np.load(EMB_BEFORE)
emb_after  = np.load(EMB_AFTER)
assert emb_before.shape[0] == emb_after.shape[0] == len(df), 'shape embeddings != katalog'

model = SentenceTransformer('BAAI/bge-m3')
# Pakai HybridRetriever hanya untuk BM25 (dense kita override manual dgn matriks before/after)
r = HybridRetriever(df, model, reranker=None, bm25_weight=BM25_W, dense_weight=DENSE_W,
                    embed_cache=EMB_AFTER)
ids = df['biblio_id'].values

# Konteks: berapa buku relevan yang SEBELUMNYA kosong deskripsi (hanya ini yang bisa terbantu)
empty_before = set(df_before.loc[df_before['deskripsi'].str.strip().str.len() <= 10, 'biblio_id'])

gt = json.load(open(GT_PATH, encoding='utf-8'))
rows = []
for item in gt:
    q = item['query']
    rel2, rel1 = set(item['relevant_ids']['2']), set(item['relevant_ids']['1'])
    bm = r._bm25_scores(q)                                   # identik before/after
    qe = model.encode([q], normalize_embeddings=True)
    d_before = (emb_before @ qe.T).flatten()
    d_after  = (emb_after  @ qe.T).flatten()
    rank_b = [int(ids[i]) for i in np.argsort(BM25_W * bm + DENSE_W * d_before)[::-1][:10]]
    rank_a = [int(ids[i]) for i in np.argsort(BM25_W * bm + DENSE_W * d_after)[::-1][:10]]
    mb, hb, nb = metrics(rank_b, rel2, rel1)
    ma, ha, na = metrics(rank_a, rel2, rel1)
    n_enriched = len((rel2 | rel1) & empty_before)
    rows.append({'query_id': item['query_id'], 'query': q, 'n_rel': len(rel2 | rel1),
                 'n_rel_enriched': n_enriched,
                 'mrr_before': mb, 'mrr_after': ma, 'hit5_before': hb, 'hit5_after': ha,
                 'ndcg5_before': nb, 'ndcg5_after': na})

res = pd.DataFrame(rows)
res.to_csv(OUT_CSV, index=False)


def agg(col_b, col_a, label):
    b, a = res[col_b].mean(), res[col_a].mean()
    d = a - b
    rel = (d / b * 100) if b else float('nan')
    print(f"  {label:<8} {b:.4f} -> {a:.4f}   Δ={d:+.4f} ({rel:+.1f}% relatif)")


print('\n' + '=' * 64)
print('DAMPAK ENRICHMENT (Config C Hybrid 0.45/0.55, GT independen 40 query)')
print('=' * 64)
agg('mrr_before', 'mrr_after', 'MRR@5')
agg('hit5_before', 'hit5_after', 'Hit@5')
agg('ndcg5_before', 'ndcg5_after', 'NDCG@5')
naik = (res['ndcg5_after'] - res['ndcg5_before'] > 1e-6).sum()
turun = (res['ndcg5_before'] - res['ndcg5_after'] > 1e-6).sum()
print(f"\nPer query (NDCG@5): naik={naik} | turun={turun} | sama={len(res)-naik-turun}")
print(f"Query dgn ≥1 buku relevan yang diperkaya (kosong sebelumnya): "
      f"{(res['n_rel_enriched']>0).sum()}/{len(res)}")
print(f"\nCSV: {OUT_CSV}")
