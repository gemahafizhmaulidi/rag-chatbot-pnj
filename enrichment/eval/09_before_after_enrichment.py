"""
11_before_after_enrichment.py — Evaluasi dampak pengayaan metadata terhadap retrieval
=======================================================================================
Eksperimen ini menjawab pertanyaan:
    "Apakah enrichment deskripsi buku via LLM meningkatkan kualitas dense retrieval?"

Metodologi:
  1. Buat 15 query semantik (mendeskripsikan isi, bukan judul)
  2. Untuk setiap query, cari buku relevan di opacv2 via keyword search di notes/deskripsi
     — HANYA buku yang di opac_original TIDAK punya deskripsi (kondisi fair)
  3. Jalankan dense retrieval menggunakan embeddings_before vs embeddings_after
     (bge-m3, cosine similarity)
  4. Hitung MRR@5, Hit@5, NDCG@5, Hit@10 untuk kedua kondisi
  5. Bandingkan dan simpulkan

Jalankan:
    cd rag-system-v3
    python scripts/11_before_after_enrichment.py
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import mysql.connector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ── 1. Koneksi database ───────────────────────────────────────────────────────
DB = dict(host=os.getenv('DB_HOST','localhost'),
          user=os.getenv('DB_USER','root'),
          password=os.getenv('DB_PASSWORD',''))

conn_before = mysql.connector.connect(**DB, database=os.getenv('DB_NAME_ORIGINAL', 'opac_original'))
conn_after  = mysql.connector.connect(**DB, database=os.getenv('DB_NAME2', 'opacv2'))

print(f"DB before (asli)  : {os.getenv('DB_NAME_ORIGINAL', 'opac_original')}")
print(f"DB after  (enrich): {os.getenv('DB_NAME2', 'opacv2')}")

CATALOG_SQL = """
    SELECT b.biblio_id, b.title AS judul, b.notes AS deskripsi
    FROM biblio b
    JOIN mst_gmd g ON b.gmd_id = g.gmd_id AND g.gmd_code = 'TE'
    ORDER BY b.biblio_id
"""

df_before = pd.read_sql(CATALOG_SQL, conn_before).fillna('')
df_after  = pd.read_sql(CATALOG_SQL, conn_after).fillna('')

conn_before.close()
conn_after.close()

# Pastikan urutan biblio_id sama (untuk alignment dengan embeddings.npy)
df_before = df_before.sort_values('biblio_id').reset_index(drop=True)
df_after  = df_after.sort_values('biblio_id').reset_index(drop=True)

assert (df_before['biblio_id'].values == df_after['biblio_id'].values).all(), \
    "biblio_id tidak sinkron antara opac_original dan opacv2!"

print(f"Katalog: {len(df_before):,} buku")
n_desc_before = (df_before['deskripsi'].str.strip().str.len() > 10).sum()
n_desc_after  = (df_after['deskripsi'].str.strip().str.len() > 10).sum()
print(f"Deskripsi sebelum enrichment : {n_desc_before:,} ({n_desc_before/len(df_before)*100:.1f}%)")
print(f"Deskripsi setelah enrichment : {n_desc_after:,}  ({n_desc_after/len(df_after)*100:.1f}%)")
print()

# ── 2. Load embeddings ────────────────────────────────────────────────────────
emb_before = np.load(os.path.join(OUTPUT_DIR, 'embeddings_before.npy'))
emb_after  = np.load(os.path.join(OUTPUT_DIR, 'embeddings_after.npy'))

assert emb_before.shape == emb_after.shape == (len(df_before), 1024), \
    f"Shape mismatch: {emb_before.shape} vs {emb_after.shape}"

print(f"Embeddings loaded: {emb_before.shape}")
print()

# ── 3. Load bge-m3 untuk encode query ────────────────────────────────────────
print("Loading bge-m3 untuk encode query...")
model = SentenceTransformer('BAAI/bge-m3')
print("Model loaded.\n")

# ── 4. Definisi query semantik + keyword pencarian ground truth ───────────────
# Setiap query punya 'keywords' yang digunakan untuk mencari buku relevan
# di deskripsi opacv2. Ini semi-otomatis: relevan = deskripsi mengandung keyword.
# Catatan: hanya buku yang SEBELUMNYA tidak punya deskripsi yang dihitung.

QUERIES = [
    {
        'query'   : 'buku yang menjelaskan cara kerja transistor dan rangkaian penguat sinyal listrik',
        'keywords': ['transistor', 'penguat', 'sinyal', 'elektronika analog'],
        'category': 'elektronika',
    },
    {
        'query'   : 'panduan membangun aplikasi berbasis web menggunakan teknologi server-side',
        'keywords': ['web', 'server', 'aplikasi web', 'php', 'javascript', 'html'],
        'category': 'pemrograman web',
    },
    {
        'query'   : 'buku tentang cara menghitung beban struktur dan perencanaan gedung bertingkat',
        'keywords': ['struktur', 'beban', 'beton', 'gedung', 'konstruksi', 'bangunan'],
        'category': 'teknik sipil',
    },
    {
        'query'   : 'pengantar konsep kecerdasan buatan dan jaringan syaraf tiruan untuk pemula',
        'keywords': ['kecerdasan buatan', 'neural network', 'machine learning', 'deep learning', 'jaringan syaraf'],
        'category': 'AI/ML',
    },
    {
        'query'   : 'buku akuntansi yang membahas pencatatan transaksi keuangan perusahaan',
        'keywords': ['akuntansi', 'jurnal', 'transaksi', 'laporan keuangan', 'neraca', 'pencatatan'],
        'category': 'akuntansi',
    },
    {
        'query'   : 'materi kuliah tentang analisis tegangan regangan dan kekuatan material',
        'keywords': ['tegangan', 'regangan', 'kekuatan material', 'mekanika', 'material teknik'],
        'category': 'teknik mesin',
    },
    {
        'query'   : 'buku yang membahas strategi pemasaran produk dan perilaku konsumen',
        'keywords': ['pemasaran', 'marketing', 'konsumen', 'strategi', 'produk', 'pasar'],
        'category': 'manajemen pemasaran',
    },
    {
        'query'   : 'panduan belajar pemrograman berorientasi objek untuk pemula',
        'keywords': ['pemrograman', 'objek', 'class', 'inheritance', 'oop', 'java', 'python'],
        'category': 'pemrograman',
    },
    {
        'query'   : 'buku tentang cara kerja protokol jaringan internet dan keamanan data',
        'keywords': ['protokol', 'jaringan', 'tcp', 'ip', 'keamanan', 'kriptografi', 'firewall'],
        'category': 'jaringan komputer',
    },
    {
        'query'   : 'materi tentang prinsip perpajakan dan cara menghitung pajak penghasilan',
        'keywords': ['pajak', 'perpajakan', 'penghasilan', 'pph', 'wajib pajak', 'fiskal'],
        'category': 'perpajakan',
    },
    {
        'query'   : 'buku tentang pengelolaan dan perawatan mesin produksi industri',
        'keywords': ['mesin', 'perawatan', 'maintenance', 'industri', 'produksi', 'manufaktur'],
        'category': 'teknik mesin industri',
    },
    {
        'query'   : 'panduan praktis instalasi dan pemrograman mikrokontroler arduino',
        'keywords': ['mikrokontroler', 'arduino', 'embedded', 'sensor', 'aktuator', 'sistem tertanam'],
        'category': 'embedded system',
    },
    {
        'query'   : 'buku statistika yang membahas uji hipotesis dan analisis data penelitian',
        'keywords': ['statistika', 'uji hipotesis', 'analisis data', 'distribusi', 'regresi', 'korelasi'],
        'category': 'statistika',
    },
    {
        'query'   : 'pengantar manajemen sumber daya manusia dan pengembangan organisasi',
        'keywords': ['sumber daya manusia', 'sdm', 'rekrutmen', 'pelatihan', 'organisasi', 'manajemen sdm'],
        'category': 'manajemen SDM',
    },
    {
        'query'   : 'buku tentang sistem basis data relasional dan bahasa query SQL',
        'keywords': ['basis data', 'database', 'sql', 'relasional', 'query', 'tabel', 'normalisasi'],
        'category': 'basis data',
    },
]

# ── 5. Fungsi pencarian ground truth ─────────────────────────────────────────
def find_relevant_books(df_after, df_before, keywords, min_desc_len=50, top_n=10):
    """
    Cari buku relevan di opacv2 berdasarkan keyword di deskripsi.
    Hanya buku yang SEBELUMNYA tidak punya deskripsi di opac_original.
    """
    # Buku yang sebelumnya tidak punya deskripsi
    no_desc_before = df_before['deskripsi'].str.strip().str.len() <= 10

    # Buku di opacv2 yang deskripsinya mengandung minimal satu keyword
    desc_after = df_after['deskripsi'].str.lower()
    keyword_match = pd.Series([False] * len(df_after))
    for kw in keywords:
        keyword_match |= desc_after.str.contains(kw.lower(), regex=False, na=False)

    # Hitung berapa keyword yang match (untuk ranking)
    match_count = pd.Series([0] * len(df_after))
    for kw in keywords:
        match_count += desc_after.str.contains(kw.lower(), regex=False, na=False).astype(int)

    # Gabung kondisi: tidak punya deskripsi sebelum + punya deskripsi yang relevan sesudah
    mask = no_desc_before & keyword_match & (df_after['deskripsi'].str.len() >= min_desc_len)

    if mask.sum() == 0:
        return []

    # Ambil top-N berdasarkan jumlah keyword yang match
    candidates = df_after[mask].copy()
    candidates['_match_count'] = match_count[mask].values
    candidates = candidates.sort_values('_match_count', ascending=False).head(top_n)

    return candidates['biblio_id'].tolist()

# ── 6. Fungsi retrieval dan metrik ────────────────────────────────────────────
def dense_retrieve(query_emb, corpus_emb, top_k=10):
    """Cosine similarity retrieval. Embeddings harus sudah normalized."""
    scores = corpus_emb @ query_emb  # dot product = cosine sim (normalized)
    top_idx = np.argsort(-scores)[:top_k]
    return top_idx.tolist(), scores[top_idx].tolist()

def compute_metrics(retrieved_ids, relevant_ids, k=5):
    """Hitung MRR@k, Hit@k, NDCG@k."""
    if not relevant_ids:
        return {'mrr': 0, 'hit@1': 0, f'hit@{k}': 0, 'hit@10': 0, f'ndcg@{k}': 0}

    relevant_set = set(relevant_ids)

    # MRR@k
    mrr = 0.0
    for rank, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_set:
            mrr = 1.0 / rank
            break

    # Hit@1
    hit1 = 1 if retrieved_ids[0] in relevant_set else 0

    # Hit@k
    hitk = 1 if any(rid in relevant_set for rid in retrieved_ids[:k]) else 0

    # Hit@10
    hit10 = 1 if any(rid in relevant_set for rid in retrieved_ids[:10]) else 0

    # NDCG@k — binary relevance
    dcg = sum(1.0 / np.log2(rank + 1)
              for rank, rid in enumerate(retrieved_ids[:k], 1)
              if rid in relevant_set)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {'mrr': mrr, 'hit@1': hit1, f'hit@{k}': hitk, 'hit@10': hit10, f'ndcg@{k}': ndcg}

# ── 7. Jalankan evaluasi ──────────────────────────────────────────────────────
print("=" * 70)
print("EVALUASI BEFORE vs AFTER ENRICHMENT — Dense Retrieval Only")
print("=" * 70)
print()

biblio_ids = df_before['biblio_id'].tolist()

results = []
skipped = 0

for i, q in enumerate(QUERIES, 1):
    # Cari ground truth
    relevant_ids = find_relevant_books(df_after, df_before, q['keywords'])

    if len(relevant_ids) == 0:
        print(f"[{i:2d}] SKIP — tidak ada ground truth untuk: {q['query'][:60]}")
        skipped += 1
        continue

    # Encode query
    q_emb = model.encode(q['query'], normalize_embeddings=True)

    # Retrieve with embeddings_before
    idx_before, _ = dense_retrieve(q_emb, emb_before)
    ids_before = [biblio_ids[idx] for idx in idx_before]
    m_before = compute_metrics(ids_before, relevant_ids)

    # Retrieve with embeddings_after
    idx_after, _ = dense_retrieve(q_emb, emb_after)
    ids_after = [biblio_ids[idx] for idx in idx_after]
    m_after = compute_metrics(ids_after, relevant_ids)

    delta_mrr  = m_after['mrr']    - m_before['mrr']
    delta_hit5 = m_after['hit@5']  - m_before['hit@5']
    delta_ndcg = m_after['ndcg@5'] - m_before['ndcg@5']

    trend = '⬆️ ' if delta_mrr > 0 else ('⬇️ ' if delta_mrr < 0 else '➡️ ')

    print(f"[{i:2d}] {trend} {q['category']:<25} GT={len(relevant_ids)} buku")
    print(f"      Query   : {q['query'][:65]}")
    print(f"      BEFORE  : MRR={m_before['mrr']:.3f}  Hit@5={m_before['hit@5']:.0f}  NDCG@5={m_before['ndcg@5']:.3f}")
    print(f"      AFTER   : MRR={m_after['mrr']:.3f}  Hit@5={m_after['hit@5']:.0f}  NDCG@5={m_after['ndcg@5']:.3f}")
    print(f"      DELTA   : MRR={delta_mrr:+.3f}  Hit@5={delta_hit5:+.0f}  NDCG@5={delta_ndcg:+.3f}")
    print()

    results.append({
        'query'       : q['query'],
        'category'    : q['category'],
        'n_relevant'  : len(relevant_ids),
        'mrr_before'  : m_before['mrr'],
        'mrr_after'   : m_after['mrr'],
        'hit5_before' : m_before['hit@5'],
        'hit5_after'  : m_after['hit@5'],
        'ndcg5_before': m_before['ndcg@5'],
        'ndcg5_after' : m_after['ndcg@5'],
        'hit10_before': m_before['hit@10'],
        'hit10_after' : m_after['hit@10'],
        'delta_mrr'   : delta_mrr,
        'delta_hit5'  : delta_hit5,
        'delta_ndcg5' : delta_ndcg,
    })

# ── 8. Ringkasan ──────────────────────────────────────────────────────────────
if not results:
    print("Tidak ada hasil. Periksa koneksi database dan keyword.")
else:
    df_res = pd.DataFrame(results)

    print("=" * 70)
    print("RINGKASAN HASIL")
    print("=" * 70)
    print(f"Query dievaluasi : {len(results)} (skip: {skipped})")
    print()
    print(f"{'Metrik':<15} {'BEFORE':>10} {'AFTER':>10} {'DELTA':>10} {'Naik/Turun'}")
    print("-" * 55)

    metrics = [
        ('MRR@5',   'mrr_before',   'mrr_after'),
        ('Hit@5',   'hit5_before',  'hit5_after'),
        ('NDCG@5',  'ndcg5_before', 'ndcg5_after'),
        ('Hit@10',  'hit10_before', 'hit10_after'),
    ]

    for label, col_b, col_a in metrics:
        b = df_res[col_b].mean()
        a = df_res[col_a].mean()
        d = a - b
        arrow = '⬆️ ' if d > 0.01 else ('⬇️ ' if d < -0.01 else '➡️ ')
        print(f"{label:<15} {b:>10.4f} {a:>10.4f} {d:>+10.4f}  {arrow}")

    print()
    naik  = (df_res['delta_mrr'] > 0.01).sum()
    turun = (df_res['delta_mrr'] < -0.01).sum()
    sama  = len(df_res) - naik - turun
    print(f"Per query (MRR): Naik={naik} | Turun={turun} | Sama={sama}")
    print()

    # Simpan hasil
    out_path = os.path.join(OUTPUT_DIR, 'before_after_enrichment.csv')
    df_res.to_csv(out_path, index=False)
    print(f"Hasil disimpan: {out_path}")

    # Interpretasi
    print()
    print("=" * 70)
    print("INTERPRETASI")
    print("=" * 70)
    avg_delta_mrr  = df_res['delta_mrr'].mean()
    avg_delta_ndcg = df_res['delta_ndcg5'].mean()

    if avg_delta_mrr > 0.05:
        print("✅ Enrichment MENINGKATKAN retrieval secara signifikan pada query semantik.")
        print("   → Bukti empiris: deskripsi LLM membantu dense embedding menemukan buku")
        print("     yang tidak bisa ditemukan dari judul/topik saja.")
    elif avg_delta_mrr > 0:
        print("✅ Enrichment memberikan peningkatan kecil pada query semantik.")
        print("   → Manfaat ada tapi terbatas; hybrid BM25 sudah menutup sebagian gap.")
    elif avg_delta_mrr > -0.05:
        print("➡️  Enrichment tidak memberikan perubahan signifikan pada metrik MRR.")
        print("   → Ini terjadi karena query semantik yang dipilih masih bisa dijawab")
        print("     dari judul/topik. Nilai enrichment ada pada coverage (90% buku")
        print("     kini punya representasi semantik), bukan pada rank improvement.")
    else:
        print("⬇️  MRR turun setelah enrichment.")
        print("   → Deskripsi LLM kemungkinan terlalu generik sehingga menambah noise.")
        print("     Perlu analisis lebih lanjut per kategori.")

    print()
    print("Catatan metodologi:")
    print("  - Evaluasi ini menggunakan DENSE-ONLY retrieval (bge-m3 cosine similarity)")
    print("  - Bukan Hybrid Config C yang dipakai production (BM25 0.3 + Dense 0.7)")
    print("  - Ground truth: buku yang SEBELUMNYA tidak punya deskripsi,")
    print("    ditemukan via keyword search di deskripsi opacv2")
    print("  - Bias inherent: keyword search bisa over-represent buku dengan")
    print("    deskripsi yang mengulang keyword — ini limitation yang harus disebutkan")
