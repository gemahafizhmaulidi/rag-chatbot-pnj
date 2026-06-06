"""api.py — Flask Backend Chatbot Perpustakaan PNJ

Endpoint:
    POST /chat                      — kirim pesan, terima jawaban
    GET  /health                    — status server
    GET  /enrich/status             — statistik enrichment metadata
    GET  /enrich/<id>/stream        — SSE stream progres enrichment (preview)
    POST /enrich/<id>               — enrichment non-streaming (preview)
"""

import os, re, time, logging, warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import mysql.connector
import requests
import chromadb
from sentence_transformers import SentenceTransformer

from core.config import (
    DB_CONFIG, LLM_MODEL, LLM_CHAT_ENDPOINT, LLM_TEMP,
    CHROMA_DIR, CHROMA_COLLECTION, KB_TOP_K, KB_DIST_THRESHOLD,
    RERANKER_THRESHOLD, EMBED_CACHE,
)
from core.prompts import (
    OOS_RESPONSE, NO_BOOK_RESPONSE, GREETING_RESPONSE, THANKS_RESPONSE,
    THANKS_WORDS, CLARIFICATION_MSG, CLARIFICATION_RECOM_MSG, PROCESSING_ERROR,
    SECURITY_PATTERNS, SYS_PROMPTS,
)
from core.db import load_catalog, execute_stats_query
from retrieval.retriever import HybridRetriever
from retrieval.query_expander import detect_beginner_intent
from routing.router import route_query
from enrichment.enricher import enrich_stream

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
_embed_model = None
_retriever   = None
_chroma_col  = None
_ready       = False


# ── ChromaDB ──────────────────────────────────────────────────────────────────
def load_chroma():
    if not os.path.exists(CHROMA_DIR):
        log.warning('ChromaDB belum ada. Jalankan: python knowledge_base/build/03_build_kb.py')
        return None
    try:
        client     = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(CHROMA_COLLECTION)
        log.info(f'ChromaDB KB: {collection.count()} chunks')
        return collection
    except Exception as e:
        log.error(f'Gagal load ChromaDB: {e}')
        return None


def search_kb(query: str, threshold_override: float | None = None) -> dict:
    if _chroma_col is None:
        return {'passages': [], 'best_distance': 1.0, 'relevant': False}
    thr   = threshold_override if threshold_override is not None else KB_DIST_THRESHOLD
    q_emb = _embed_model.encode([query], normalize_embeddings=True).tolist()
    res   = _chroma_col.query(
        query_embeddings=q_emb, n_results=KB_TOP_K,
        include=['documents', 'metadatas', 'distances'],
    )
    passages = []
    best_distance = 1.0
    if res and res['documents'] and res['documents'][0]:
        for doc, meta, dist in zip(res['documents'][0], res['metadatas'][0], res['distances'][0]):
            passages.append({
                'text'    : doc,
                'source'  : meta.get('source', ''),
                'doc_type': meta.get('doc_type', ''),
                'distance': round(dist, 4),
            })
        best_distance = res['distances'][0][0]
    relevant = best_distance <= thr
    log.info(f'KB search — dist={best_distance:.4f} (thr={thr:.4f}) → {"RELEVAN" if relevant else "OOS"}')
    return {'passages': passages, 'best_distance': best_distance, 'relevant': relevant}


# ── Context builders ──────────────────────────────────────────────────────────
def build_book_context(books: list) -> str:
    lines = []
    for i, b in enumerate(books, 1):
        entry = [f"[{i}] {b.get('judul', '-')}"]
        entry.append(f"    Penulis       : {b.get('penulis', '-') or '-'}")
        entry.append(f"    Nomor Panggil : {b.get('call_number', '-') or '-'}")
        penerbit = b.get('penerbit', '') or ''
        tahun    = b.get('tahun', '') or ''
        if penerbit:
            entry.append(f"    Penerbit  : {penerbit}{', ' + tahun if tahun else ''}")
        total    = int(b.get('total_eksemplar', 0) or 0)
        tersedia = int(b.get('tersedia', 0) or 0)
        loan_count = int(b.get('loan_count', 0) or 0)
        if total > 0:
            entry.append(f"    Stok      : {tersedia} dari {total} eksemplar tersedia")
        if loan_count > 0:
            entry.append(f"    Dipinjam  : {loan_count} kali")
        desc = b.get('deskripsi', '') or ''
        if len(desc.strip()) > 50:
            entry.append(f"    Deskripsi : {desc[:200]}{'...' if len(desc) > 200 else ''}")
        lines.append('\n'.join(entry))
    return '\n\n'.join(lines)


def build_kb_context(passages: list) -> str:
    parts = []
    for p in passages:
        src = os.path.basename(p.get('source', ''))
        parts.append(f"[Sumber: {src}]\n{p['text'][:600]}")
    return '\n\n---\n\n'.join(parts)


def build_hybrid_context(books: list, passages: list) -> str:
    parts = []
    if books:
        parts.append("=== KATALOG BUKU ===\n" + build_book_context(books))
    if passages:
        parts.append("=== INFORMASI LAYANAN PERPUSTAKAAN ===\n" + build_kb_context(passages))
    return '\n\n'.join(parts)


# ── LLM ───────────────────────────────────────────────────────────────────────
def generate_llm(query: str, context: str, query_type: str) -> str | None:
    system = SYS_PROMPTS.get(query_type, SYS_PROMPTS['general_info'])
    try:
        r = requests.post(LLM_CHAT_ENDPOINT, json={
            "model" : LLM_MODEL,
            "stream": False,
            "think" : False,
            "options": {"temperature": LLM_TEMP, "num_ctx": 8192, "num_predict": 800},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": f"{query}\n\nKonteks:\n{context}"},
            ],
        }, timeout=120)
        if r.status_code == 200:
            generated = (r.json().get('message', {}).get('content', '') or '').strip()
            log.info(f'LLM len={len(generated)} | preview={generated[:80]!r}')
            if not generated:
                return None
            generated = re.sub(r'<think>.*?</think>', '', generated, flags=re.DOTALL).strip()
            return generated or None
        log.error(f'LLM HTTP {r.status_code}: {r.text[:200]}')
    except Exception as e:
        log.error(f'LLM error: {e}')
    return None


# ── Stats ─────────────────────────────────────────────────────────────────────
_STATS_PATTERNS = [
    ('top_borrowed_week',
     ['minggu ini', 'seminggu terakhir', '7 hari terakhir', 'pekan ini'],
     ['dipinjam', 'populer', 'terlaris', 'sering', 'banyak', 'diminati', 'buku apa']),
    ('top_borrowed_month',
     ['bulan ini', 'sebulan terakhir', '30 hari terakhir'],
     ['dipinjam', 'populer', 'terlaris', 'sering', 'banyak', 'diminati', 'buku apa']),
    ('loans_this_month',
     ['peminjaman bulan ini', 'dipinjam bulan ini', 'transaksi bulan ini',
      'berapa peminjaman bulan', 'jumlah peminjaman bulan']),
    ('loans_this_year',
     ['peminjaman tahun ini', 'berapa peminjaman tahun', 'jumlah peminjaman tahun']),
    ('loans_total',
     ['total peminjaman', 'berapa total peminjaman', 'jumlah total peminjaman',
      'statistik peminjaman', 'riwayat peminjaman']),
    ('top_borrowed_alltime',
     ['paling sering dipinjam', 'paling banyak dipinjam', 'buku terpopuler',
      'buku paling populer', 'buku terlaris', 'buku favorit perpustakaan',
      'paling diminati', 'paling banyak dibaca']),
    ('available_books',
     ['berapa buku tersedia', 'berapa tersedia', 'berapa eksemplar tersedia',
      'eksemplar yang tersedia', 'stok tersedia', 'berapa yang bisa dipinjam',
      'koleksi tersedia', 'yang tersedia sekarang']),
    ('total_items',
     ['berapa eksemplar', 'jumlah eksemplar', 'total eksemplar', 'berapa fisik buku']),
    ('total_books',
     ['berapa total buku', 'berapa jumlah buku', 'berapa judul buku',
      'jumlah total buku', 'total koleksi buku', 'jumlah koleksi',
      'berapa koleksi', 'ada berapa judul', 'berapa buku di database',
      'berapa buku di perpustakaan', 'total buku perpustakaan',
      'berapa buku yang dimiliki', 'ada berapa buku']),
]


def classify_stats_query(ql: str) -> str | None:
    for entry in _STATS_PATTERNS:
        subtype, primary_kws = entry[0], entry[1]
        ctx_kws = entry[2] if len(entry) > 2 else []
        if not any(kw in ql for kw in primary_kws):
            continue
        if ctx_kws and not any(kw in ql for kw in ctx_kws):
            continue
        return subtype
    return None


def format_stats_answer(subtype: str) -> str:
    rows = execute_stats_query(subtype)
    if not rows:
        return "Maaf, data statistik tidak dapat diambil saat ini. Silakan hubungi petugas perpustakaan PNJ."

    if subtype == 'total_books':
        n = rows[0]['n']
        return (f"Perpustakaan PNJ saat ini memiliki **{n:,} judul buku teks** dalam database OPAC.\n\n"
                f"Untuk mencari buku tertentu: *\"Ada buku tentang machine learning?\"*")
    if subtype == 'total_items':
        return f"Total eksemplar fisik koleksi Perpustakaan PNJ: **{rows[0]['n']:,} eksemplar**."
    if subtype == 'available_books':
        return (f"Saat ini terdapat **{rows[0]['n']:,} eksemplar** yang tersedia untuk dipinjam.\n\n"
                f"Cek ketersediaan terkini di [OPAC PNJ](https://opac.pnj.ac.id).")
    if subtype == 'loans_this_month':
        return f"Jumlah transaksi peminjaman **bulan ini**: **{rows[0]['n']:,} peminjaman**."
    if subtype == 'loans_this_year':
        return f"Jumlah transaksi peminjaman **tahun ini**: **{rows[0]['n']:,} peminjaman**."
    if subtype == 'loans_total':
        return f"Total riwayat peminjaman Perpustakaan PNJ: **{rows[0]['n']:,} transaksi** (sejak 2015)."
    if subtype in ('top_borrowed_week', 'top_borrowed_month', 'top_borrowed_alltime'):
        label = {'top_borrowed_week': 'seminggu terakhir',
                 'top_borrowed_month': 'sebulan terakhir',
                 'top_borrowed_alltime': 'sepanjang waktu'}[subtype]
        if not rows or (rows[0].get('n') or 0) == 0:
            return f"Tidak ada data peminjaman untuk periode {label}."
        lines = [f"📚 Buku teks paling sering dipinjam **{label}**:\n"]
        for i, r in enumerate(rows, 1):
            lines.append(f"{i}. {(r.get('title') or '(judul tidak diketahui)').strip()} *({r.get('n', 0)}×)*")
        return '\n'.join(lines)
    return "Statistik tidak tersedia."


# ── Chat pipeline ─────────────────────────────────────────────────────────────
def chat(query: str) -> dict:
    # 1. Security pre-filter
    if any(p in query.lower() for p in SECURITY_PATTERNS):
        log.info('Security pre-filter → ditolak')
        return {'answer': OOS_RESPONSE, 'query_type': 'oos', 'sources': []}

    # 2. LLM-router
    d     = route_query(query)
    route = d['route']
    log.info('Router: %s | search_query=%r info_query=%r', route, d['search_query'], d['info_query'])

    # 3. Greeting
    if route == 'greeting':
        resp = THANKS_RESPONSE if any(w in query.lower() for w in THANKS_WORDS) else GREETING_RESPONSE
        return {'answer': resp, 'query_type': 'greeting', 'sources': []}

    # 4. OOS
    if route == 'oos':
        return {'answer': OOS_RESPONSE, 'query_type': 'oos', 'sources': []}

    # 5. Statistik
    if route == 'stats':
        subtype = d['stat_subtype'] or classify_stats_query(query.strip().lower()) or 'total_books'
        return {'answer': format_stats_answer(subtype), 'query_type': 'stats_query',
                'stat_subtype': subtype, 'sources': []}

    # 6. Rekomendasi
    if route == 'recommendation':
        if not d['search_query']:
            return {'answer': CLARIFICATION_RECOM_MSG, 'query_type': 'clarification', 'sources': []}
        books = _retriever.search_no_rerank(d['search_query'], top_k=10)
        if not books:
            return {'answer': NO_BOOK_RESPONSE, 'query_type': 'recommendation_search', 'sources': []}
        available   = sorted([b for b in books if b.get('tersedia', 0) > 0],
                             key=lambda b: b.get('loan_count', 0), reverse=True)
        unavailable = sorted([b for b in books if b.get('tersedia', 0) == 0],
                             key=lambda b: b.get('loan_count', 0), reverse=True)
        top5    = (available + unavailable)[:5]
        context = build_book_context(top5)
        if detect_beginner_intent(query):
            context += "\n\n[Catatan: user mencari buku untuk pemula atau tahap awal belajar]"
        answer = generate_llm(query, context, 'recommendation_search') or PROCESSING_ERROR
        return {'answer': answer, 'query_type': 'recommendation_search', 'sources': top5}

    # 7. Hybrid
    if route == 'hybrid':
        books = _retriever.search_no_rerank(d['search_query'] or query, top_k=3)
        kb    = search_kb(d['info_query'] or query)
        if not books and not kb['relevant']:
            return {'answer': OOS_RESPONSE, 'query_type': 'oos',
                    'sources': [], 'kb_distance': kb['best_distance']}
        if not books:
            context, qt, sources = build_kb_context(kb['passages']), 'general_info', kb['passages']
        elif not kb['relevant']:
            context, qt, sources = build_book_context(books), 'book_search', books
        else:
            context, qt, sources = build_hybrid_context(books, kb['passages']), 'hybrid', books + kb['passages']
        answer = generate_llm(query, context, qt) or PROCESSING_ERROR
        return {'answer': answer, 'query_type': qt, 'sources': sources}

    # 8. Pencarian buku
    if route == 'book_search':
        if not d['search_query']:
            return {'answer': CLARIFICATION_MSG, 'query_type': 'clarification', 'sources': []}
        books = _retriever.search_no_rerank(d['search_query'], top_k=5)
        if not books:
            return {'answer': NO_BOOK_RESPONSE, 'query_type': 'book_search', 'sources': []}
        answer = generate_llm(query, build_book_context(books), 'book_search') or PROCESSING_ERROR
        return {'answer': answer, 'query_type': 'book_search', 'sources': books}

    # 9. General info — hard-stop backstop
    kb = search_kb(d['info_query'] or query)
    if not kb['relevant']:
        log.info('Hard-stop backstop → OOS (dist=%.4f)', kb['best_distance'])
        return {'answer': OOS_RESPONSE, 'query_type': 'general_info',
                'sources': [], 'kb_distance': kb['best_distance']}
    answer = generate_llm(query, build_kb_context(kb['passages']), 'general_info') or PROCESSING_ERROR
    return {'answer': answer, 'query_type': 'general_info', 'sources': kb['passages']}


# ── Startup ───────────────────────────────────────────────────────────────────
def init_system():
    global _embed_model, _retriever, _chroma_col, _ready
    log.info('Memuat sistem RAG Chatbot PNJ...')
    _embed_model = SentenceTransformer('BAAI/bge-m3')
    conn         = mysql.connector.connect(**DB_CONFIG)
    df           = load_catalog(conn)
    conn.close()
    _retriever = HybridRetriever(
        df, _embed_model, reranker=None,
        bm25_weight=0.45, dense_weight=0.55,
        embed_cache=EMBED_CACHE,
        reranker_threshold=RERANKER_THRESHOLD,
    )
    _chroma_col = load_chroma()
    _ready = True
    log.info('Sistem siap ✓ (Config C: Hybrid BM25+Dense, tanpa reranker)')


# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route('/health')
def health():
    return jsonify({
        'status'              : 'ready' if _ready else 'loading',
        'version'             : 'v3',
        'llm_model'           : LLM_MODEL,
        'hardstop_threshold'  : KB_DIST_THRESHOLD,
        'reranker_threshold'  : RERANKER_THRESHOLD,
    })


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    if not _ready:
        return jsonify({'error': 'Sistem sedang loading...'}), 503
    body  = request.get_json(silent=True) or {}
    query = (body.get('message') or '').strip()
    if not query:
        return jsonify({'error': 'Field "message" tidak boleh kosong.'}), 400
    if len(query) > 500:
        return jsonify({'error': 'Pesan terlalu panjang (max 500 karakter).'}), 400

    t0     = time.time()
    result = chat(query)
    elapsed = round(time.time() - t0, 2)
    qt      = result['query_type']

    sources_clean = []
    for s in result['sources']:
        if 'biblio_id' in s:
            sources_clean.append({
                'source_type'    : 'catalog',
                'judul'          : s.get('judul', ''),
                'penulis'        : s.get('penulis', ''),
                'call_number'    : s.get('call_number', ''),
                'penerbit'       : s.get('penerbit', ''),
                'tahun'          : s.get('tahun', ''),
                'total_eksemplar': s.get('total_eksemplar', 0),
                'tersedia'       : s.get('tersedia', 0),
                'score'          : round(s.get('score', 0), 4),
            })
        elif 'text' in s:
            sources_clean.append({
                'source_type': 'kb',
                'source'     : s.get('source', ''),
                'doc_type'   : s.get('doc_type', ''),
                'distance'   : s.get('distance', 0),
            })

    answer_out = result['answer']
    if qt in ('book_search', 'recommendation_search', 'open_ended_recommendation', 'hybrid') \
            and any(s.get('source_type') == 'catalog' for s in sources_clean):
        answer_out += ("\n\n_ℹ️ Ringkasan/alasan buku dirangkum otomatis dari metadata "
                       "(bukan kutipan resmi); untuk detail isi, periksa bukunya langsung._")

    resp = {'answer': answer_out, 'query_type': qt, 'sources': sources_clean, 'elapsed_s': elapsed}
    if 'stat_subtype' in result:
        resp['stat_subtype'] = result['stat_subtype']
    if 'kb_distance' in result:
        resp['kb_distance'] = result['kb_distance']
    return jsonify(resp)


@app.route('/enrich/status')
def enrich_status():
    try:
        conn   = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            'SELECT COUNT(*) AS total, '
            'SUM(CASE WHEN notes IS NOT NULL AND notes != "" THEN 1 ELSE 0 END) AS dengan_desc, '
            'SUM(CASE WHEN notes IS NULL OR notes = "" THEN 1 ELSE 0 END) AS tanpa_desc '
            'FROM biblio b JOIN mst_gmd g ON b.gmd_id=g.gmd_id AND g.gmd_code="TE"'
        )
        row = cursor.fetchone()
        cursor.close(); conn.close()
        from core.config import ENRICH_MODEL
        return jsonify({
            'total'      : int(row['total']),
            'dengan_desc': int(row['dengan_desc']),
            'tanpa_desc' : int(row['tanpa_desc']),
            'persen_done': round(int(row['dengan_desc']) / int(row['total']) * 100, 1) if row['total'] else 0,
            'model'      : ENRICH_MODEL,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/enrich/<int:biblio_id>/stream')
def enrich_book_stream(biblio_id):
    return Response(
        enrich_stream(biblio_id),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no',
                 'Access-Control-Allow-Origin': '*'},
    )


@app.route('/enrich/<int:biblio_id>', methods=['POST'])
def enrich_book(biblio_id):
    import json as _json
    try:
        last = None
        for chunk in enrich_stream(biblio_id):
            raw = chunk.replace('data: ', '').strip()
            if raw:
                last = _json.loads(raw)
        if last and last.get('status') == 'success':
            return jsonify(last), 200
        return jsonify({'status': 'failed', 'message': (last or {}).get('msg', 'Gagal')}), 422
    except Exception as e:
        log.exception('Enrich error biblio_id=%d', biblio_id)
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):  return jsonify({'error': 'Endpoint tidak ditemukan.'}), 404

@app.errorhandler(500)
def server_error(e): return jsonify({'error': 'Internal server error.'}), 500


if __name__ == '__main__':
    init_system()
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
