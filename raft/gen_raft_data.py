"""
gen_raft_data.py — Fase 1 eksperimen RAG vs RAG+RAFT (skripsi Gema).

Membangun dataset Retrieval-Augmented Fine-Tuning (RAFT) untuk generator LLM:
  triple = (query, konteks_retrieval[+distraktor], gold_answer)

Prinsip validitas (jangan diubah tanpa alasan):
  1. KONTEKS ASLI  — diambil lewat retriever + ChromaDB yang SAMA dengan produksi
                     (api.py), bukan dikarang.
  2. FORMAT == INFERENCE — system prompt & format user "{query}\n\nKonteks:\n{ctx}"
                     direuse PERSIS dari api.generate_llm(). Kalau beda, fine-tune
                     belajar distribusi yang salah.
  3. HANYA 4 ROUTE yang lewat LLM generator: book_search, recommendation,
                     general_info, hybrid. greeting/stats/oos pakai template → tidak
                     perlu data fine-tune.
  4. RAFT DISTRAKTOR — tiap sample = dokumen golden + distraktor; sebagian sample
                     (P_DROP) golden-nya DIBUANG → mengajarkan model menolak jujur
                     saat konteks tidak memuat jawaban.
  5. HELD-OUT HARAM — query yang dipakai evaluasi (40 GT + 50 RAGAS + 82 routing +
                     50 blackbox) TIDAK BOLEH muncul di training. Divalidasi via
                     normalisasi string + cek substring.

Teacher gold answer : Qwen 35B via OpenRouter (api.ENRICH_MODEL). Teacher ≠ judge
                      (gpt-4o-mini) → bias rendah.

Pemakaian:
    # 1) validasi struktur TANPA biaya (tidak panggil OpenRouter):
    python experiments/raft/gen_raft_data.py --dry-run --n-book 8 --n-recom 6 \
        --n-info 6 --n-hybrid 4
    # 2) generate penuh:
    python experiments/raft/gen_raft_data.py --n-book 380 --n-recom 260 \
        --n-info 220 --n-hybrid 140

Output:
    experiments/raft/out/raft_train.jsonl   (chat-format: messages[])
    experiments/raft/out/raft_manifest.json (statistik + parameter, untuk lampiran skripsi)
"""
from __future__ import annotations
import os, sys, re, json, time, random, argparse, ast, hashlib
from collections import Counter

# ── path: experiments/raft/ → rag-system-v3/ ──────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(HERE, '..', '..'))   # rag-system-v3/
OUT_DIR  = os.path.join(HERE, 'out')
sys.path.insert(0, ROOT)

import requests
import api                     # reuse sistem produksi (import aman; server hanya di __main__)
from query_expander import expand_query, detect_beginner_intent

random.seed(42)

# ── Parameter RAFT ────────────────────────────────────────────────────────────
P_DROP_GOLDEN   = 0.20    # fraksi sample yang golden-nya dibuang (ajari "tidak tahu")
N_DISTRACTOR    = 3       # jumlah dokumen distraktor per sample
MIN_GOLDEN_SCORE = 0.45   # ambang fusi minimum agar buku dianggap golden relevan.
                          # Di bawah ini → query jadi sample "organic refusal" (gold = jujur
                          # tidak ditemukan). Memisahkan match kuat (~0.73+) dari sampah (~0.33),
                          # mencegah label noise (teacher menolak padahal ditandai golden).
TEACHER_TEMP    = 0.3
TEACHER_MAXTOK  = 700
TEACHER_DELAY   = 0.4     # jeda antar panggilan OpenRouter (rate-limit aman)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 1. HELD-OUT EXCLUSION                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', str(s).lower())).strip()


def load_heldout() -> set[str]:
    """Kumpulkan SEMUA query evaluasi yang haram masuk training (ternormalisasi)."""
    held: set[str] = set()
    data = os.path.join(ROOT, 'data')

    def add(s):
        n = _norm(s)
        if len(n) >= 3:
            held.add(n)

    # ground_truth.json (retrieval) — field 'query'
    for r in json.load(open(os.path.join(data, 'ground_truth.json'))):
        add(r.get('query', ''))
    # ground_truth_ragas.json — field 'question'
    for r in json.load(open(os.path.join(data, 'ground_truth_ragas.json'))):
        add(r.get('question', ''))
    # routing_testset.json — cases[].query
    rt = json.load(open(os.path.join(data, 'routing_testset.json')))
    for c in rt.get('cases', []):
        add(c.get('query', ''))
    # 08_blackbox_test.py — TEST_CASES[].input (parse AST, tanpa eksekusi)
    bb_path = os.path.join(ROOT, 'scripts', '08_blackbox_test.py')
    try:
        tree = ast.parse(open(bb_path).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                getattr(t, 'id', '') == 'TEST_CASES' for t in node.targets):
                for elt in ast.literal_eval(node.value):
                    if isinstance(elt, dict) and 'input' in elt:
                        add(elt['input'])
    except Exception as e:
        print(f'[WARN] gagal parse blackbox cases: {e}', file=sys.stderr)

    return held


def is_heldout(query: str, held: set[str]) -> bool:
    """Tolak jika exact-match ATAU query training jadi substring held-out (dan sebaliknya)."""
    n = _norm(query)
    if n in held:
        return True
    for h in held:                      # cegah varian dekat (substring berlebihan)
        if len(n) >= 12 and (n in h or h in n):
            return True
    return False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 2. QUERY POOL (sintetik, dari katalog & KB nyata)                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
BOOK_TEMPLATES = [
    'ada buku tentang {t}?', 'buku {t} ada gak?', 'cari buku {t}',
    'rekomendasi buku {t}', 'buku soal {t} dong', 'pengen baca tentang {t}',
    'ada referensi {t}?', 'koleksi buku {t}', 'buku {t} buat tugas',
]
RECOM_TEMPLATES = [
    'rekomendasi buku {t} untuk pemula', 'saran buku {t} yang bagus dong',
    'buku {t} buat belajar dari nol', 'rekomendasiin buku {t} kak',
    'buku {t} buat skripsi', 'mau belajar {t}, buku apa ya?',
]
HYBRID_TEMPLATES = [
    'ada buku {t} dan gimana cara minjamnya?',
    'cari buku {t}, terus perpus buka jam berapa?',
    'rekomendasi buku {t} dan berapa lama masa pinjamnya?',
    'buku {t} ada? sekalian denda telat berapa ya?',
]
# Seed pertanyaan layanan (general_info) — topik yang dijamin ada di KB PNJ.
INFO_SEEDS = [
    'jam buka perpustakaan PNJ hari jumat',
    'berapa denda keterlambatan pengembalian buku',
    'bagaimana prosedur bebas pustaka', 'syarat menjadi anggota perpustakaan',
    'berapa lama masa peminjaman buku', 'berapa buku yang boleh dipinjam sekaligus',
    'cara memperpanjang masa pinjam buku', 'fasilitas apa saja di perpustakaan PNJ',
    'apakah ada layanan bebas pustaka untuk wisuda', 'lokasi perpustakaan PNJ di mana',
    'jam buka perpustakaan hari sabtu', 'apakah ada ruang baca atau diskusi',
    'bagaimana cara mengembalikan buku', 'apakah boleh meminjam buku tanpa kartu',
    'apa itu BI Corner', 'apakah ada akses wifi di perpustakaan',
    'aturan tata tertib di perpustakaan', 'cara mencari buku di OPAC',
    'apakah ada koleksi tugas akhir atau skripsi', 'bagaimana jika buku hilang',
    'visi dan misi perpustakaan PNJ', 'sejarah perpustakaan PNJ',
    'apakah ada layanan untuk penyandang disabilitas', 'jam istirahat perpustakaan',
    'apakah dosen boleh meminjam buku', 'berapa lama masa pinjam untuk dosen',
    'apakah ada layanan fotokopi di perpustakaan', 'apakah boleh membawa tas ke dalam',
    'apakah ada loker penitipan barang', 'apakah ada ruang diskusi kelompok',
    'apakah koleksi referensi boleh dipinjam', 'cara mendapatkan kartu anggota',
    'apakah perpustakaan buka saat libur semester', 'sanksi jika merusak buku',
    'apakah bisa memperpanjang pinjaman secara online', 'kontak perpustakaan PNJ',
]


def topic_pool(df) -> list[str]:
    """Topik nyata dari katalog: gabungan kolom 'topik' + kata kunci judul."""
    topics = Counter()
    for col in ('topik', 'subjek', 'subject'):
        if col in df.columns:
            for v in df[col].dropna():
                for t in re.split(r'[;,/|]', str(v)):
                    t = t.strip().lower()
                    if 3 <= len(t) <= 40 and not t.isdigit():
                        topics[t] += 1
    # fallback / pelengkap: topik dari query_expander (pasti relevan ke koleksi)
    from query_expander import TOPIC_EXPANSION
    for k in TOPIC_EXPANSION:
        topics[k] += 2
    # ambil yang cukup sering muncul (hindari topik langka/aneh)
    return [t for t, c in topics.most_common(400) if c >= 1]


def build_query_pool(df, n_book, n_recom, n_info, n_hybrid, held) -> list[dict]:
    topics = topic_pool(df)
    random.shuffle(topics)
    pool, seen = [], set()

    def emit(route, query):
        n = _norm(query)
        if n in seen or is_heldout(query, held):
            return False
        seen.add(n)
        pool.append({'route': route, 'query': query})
        return True

    ti = 0
    def next_topic():
        nonlocal ti
        t = topics[ti % len(topics)]; ti += 1
        return t

    # book_search
    while sum(p['route'] == 'book_search' for p in pool) < n_book and ti < len(topics) * 12:
        emit('book_search', random.choice(BOOK_TEMPLATES).format(t=next_topic()))
    # recommendation
    while sum(p['route'] == 'recommendation' for p in pool) < n_recom and ti < len(topics) * 24:
        emit('recommendation', random.choice(RECOM_TEMPLATES).format(t=next_topic()))
    # hybrid
    while sum(p['route'] == 'hybrid' for p in pool) < n_hybrid and ti < len(topics) * 36:
        emit('hybrid', random.choice(HYBRID_TEMPLATES).format(t=next_topic()))
    # general_info — paraphrase seed via variasi sederhana (tanpa LLM di tahap pool)
    info_variants = []
    for s in INFO_SEEDS:
        info_variants += [s, s + '?', 'tolong info ' + s, s + ' di PNJ']
    random.shuffle(info_variants)
    for q in info_variants:
        if sum(p['route'] == 'general_info' for p in pool) >= n_info:
            break
        emit('general_info', q)

    random.shuffle(pool)
    return pool


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 3. KONTEKS ASLI + DISTRAKTOR (RAFT)                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _all_kb_passages() -> list[dict]:
    """Ambil seluruh chunk KB dari ChromaDB (untuk sumber distraktor info)."""
    if api._chroma_col is None:
        return []
    got = api._chroma_col.get(include=['documents', 'metadatas'])
    out = []
    for doc, meta in zip(got['documents'], got['metadatas']):
        out.append({'text': doc, 'source': meta.get('source', ''),
                    'doc_type': meta.get('doc_type', ''), 'distance': 0.0})
    return out


def _rand_books(df, k, exclude_ids):
    rows = df[~df['biblio_id'].isin(exclude_ids)].sample(min(k, len(df)))
    return [{'judul': str(r.get('judul', '') or ''), 'penulis': str(r.get('penulis', '') or ''),
             'call_number': str(r.get('call_number', '') or ''), 'penerbit': str(r.get('penerbit', '') or ''),
             'tahun': str(r.get('tahun', '') or ''), 'deskripsi': str(r.get('deskripsi', '') or ''),
             'total_eksemplar': int(r.get('total_eksemplar', 0) or 0),
             'tersedia': int(r.get('tersedia', 0) or 0), 'score': 0.0}
            for _, r in rows.iterrows()]


def make_context(item, df, kb_all, drop_golden: bool):
    """Bangun (system_prompt, context_str, meta) untuk satu query, gaya produksi + distraktor RAFT."""
    route = item['route']; query = item['query']

    if route in ('book_search', 'recommendation'):
        if route == 'recommendation':
            expanded, interp = expand_query(query)
            golden = api._retriever.search_no_rerank(expanded, top_k=10)
            avail = [b for b in golden if b.get('tersedia', 0) > 0]
            unav  = [b for b in golden if b.get('tersedia', 0) == 0]
            golden = (avail + unav)[:5]
        else:
            interp = ''
            golden = api._retriever.search_no_rerank(query, top_k=3)
        # Score gate: golden lemah (top fusi < ambang) → perlakukan sebagai organic refusal.
        top_score = golden[0]['score'] if golden else 0.0
        weak = top_score < MIN_GOLDEN_SCORE
        organic_refusal = weak and not drop_golden
        if drop_golden or weak:
            golden = []                       # konteks hanya distraktor → gold = penolakan jujur
        gold_ids = {b['biblio_id'] for b in golden} if golden else set()
        distract = _rand_books(df, N_DISTRACTOR, gold_ids)
        docs = golden + distract
        random.shuffle(docs)
        if not docs:
            return None
        ctx = api.build_book_context(docs)
        if route == 'recommendation' and interp and golden:
            ctx = f"[Interpretasi: {interp}]\n\n{ctx}"
            if detect_beginner_intent(query):
                ctx += "\n\n[Catatan: user mencari buku untuk pemula atau tahap awal belajar]"
        sysk = 'recommendation_search' if route == 'recommendation' else 'book_search'
        system = _system_for(sysk)
        return system, ctx, {'n_golden': len(golden), 'n_distractor': len(distract),
                             'drop_golden': drop_golden, 'organic_refusal': organic_refusal,
                             'top_score': round(top_score, 3)}

    if route == 'general_info':
        res = api.search_kb(query, threshold_override=0.6)   # permisif: ambil passage utk konteks
        golden = res['passages'][:4]
        gold_src = {p['source'] for p in golden}
        distract = [p for p in kb_all if p['source'] not in gold_src]
        random.shuffle(distract); distract = distract[:N_DISTRACTOR]
        docs = (distract if drop_golden else golden + distract)
        random.shuffle(docs)
        if not docs:
            return None
        ctx = api.build_kb_context(docs)
        return _system_for('general_info'), ctx, {
            'n_golden': 0 if drop_golden else len(golden),
            'n_distractor': len(distract), 'drop_golden': drop_golden}

    if route == 'hybrid':
        books = api._retriever.search_no_rerank(query, top_k=3)
        res = api.search_kb(query, threshold_override=0.6)
        kb = res['passages'][:3]
        if drop_golden:                 # buang salah satu sisi secara acak
            if random.random() < 0.5: books = _rand_books(df, 2, set())
            else: kb = []
        if not books and not kb:
            return None
        ctx = api.build_hybrid_context(books, kb)
        return _system_for('hybrid'), ctx, {
            'n_golden': len(books) + len(kb), 'n_distractor': 0, 'drop_golden': drop_golden}

    return None


def _system_for(qtype: str) -> str:
    """Ambil SYSTEM PROMPT persis dari api.generate_llm (reflektif, agar selalu sinkron)."""
    return _SYSTEM_PROMPTS[qtype]


# Salin verbatim dari api.generate_llm (snapshot — dicek sinkron via assert di main()).
_SYSTEM_PROMPTS = {
    'recommendation_search': (
        "Kamu adalah asisten perpustakaan Politeknik Negeri Jakarta (PNJ). "
        "Berikan rekomendasi buku dari daftar berikut. "
        "Untuk setiap buku yang direkomendasikan, sebutkan:\n"
        "- Judul dan penulis\n- Nomor panggil\n- Status ketersediaan (jika ada)\n"
        "- Alasan rekomendasi berdasarkan judul, topik, atau deskripsi yang tersedia\n"
        "Prioritaskan buku yang tersedia terlebih dahulu. "
        "Jika ada catatan level pemula/lanjut di konteks, sesuaikan dengan kebutuhan user. "
        "JANGAN mengarang isi atau sinopsis buku yang tidak ada di konteks. "
        "JANGAN menyebutkan buku yang tidak relevan hanya untuk mengisi jawaban. "
        "Jika tidak ada buku yang relevan, katakan jujur dan sarankan cek OPAC. "
        "Tutup dengan satu kalimat rekomendasi utama jika memungkinkan. "
        "Langsung tulis jawaban dalam Bahasa Indonesia tanpa basa-basi."
    ),
    'book_search': (
        "Kamu adalah asisten perpustakaan Politeknik Negeri Jakarta (PNJ). "
        "Dari daftar buku berikut, tampilkan buku yang relevan dengan TOPIK UTAMA "
        "pertanyaan pengguna. Sebutkan judul, penulis, nomor panggil, dan stok. "
        "Jika ada deskripsi buku, gunakan sebagai alasan rekomendasi. "
        "Jika tidak ada deskripsi, beri alasan berdasarkan judul dan topik. "
        "JANGAN mengarang buku yang tidak ada di daftar. "
        "JANGAN menduga atau mengarang isi, sinopsis, atau konten buku. "
        "Toleransi: buku yang relevan dengan topik utama tetap ditampilkan "
        "meski tidak semua kata kunci query cocok — fokus pada kecocokan topik. "
        "Jika jumlah buku relevan yang ditemukan lebih sedikit dari yang diminta user, "
        "informasikan bahwa hanya itu yang tersedia di koleksi PNJ dan sarankan cek OPAC untuk koleksi lengkap. "
        "Jika benar-benar tidak ada buku yang topiknya relevan, "
        "katakan tidak ditemukan dan sarankan cek OPAC. "
        "Langsung tulis jawaban dalam Bahasa Indonesia tanpa basa-basi."
    ),
    'hybrid': (
        "Kamu adalah asisten perpustakaan Politeknik Negeri Jakarta (PNJ). "
        "Pertanyaan pengguna mencakup dua hal: rekomendasi buku DAN informasi layanan. "
        "Jawab keduanya secara lengkap menggunakan HANYA informasi dari konteks yang diberikan. "
        "Untuk buku: sebutkan judul, penulis, nomor panggil, dan stok jika tersedia. "
        "JANGAN menduga atau mengarang isi buku yang tidak ada di konteks. "
        "Untuk layanan: jelaskan prosedur/informasi dengan jelas sesuai konteks. "
        "Jika salah satu bagian tidak ditemukan, katakan jujur. "
        "Langsung tulis jawaban dalam Bahasa Indonesia tanpa basa-basi."
    ),
    'general_info': (
        "Kamu adalah asisten perpustakaan PNJ. "
        "Jawab dalam Bahasa Indonesia menggunakan HANYA informasi yang ada di konteks. "
        "JANGAN mengarang atau menebak: nomor telepon, email, alamat, nama petugas, "
        "jam buka, angka denda, atau data apapun yang tidak tercantum di konteks. "
        "Jika informasi tidak ada di konteks, katakan: "
        "'Maaf, informasi tersebut tidak tersedia dalam basis pengetahuan saya. "
        "Silakan hubungi petugas perpustakaan secara langsung atau kunjungi https://perpustakaan.pnj.ac.id/' "
        "Langsung tulis jawaban tanpa basa-basi."
    ),
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 4. TEACHER (OpenRouter Qwen 35B) → gold answer                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def teacher_answer(system: str, query: str, context: str) -> str | None:
    if not api.ENRICH_API_KEY:
        raise RuntimeError('OPENROUTER_API_KEY tidak di-set di .env — teacher tidak bisa jalan.')
    user_content = f"{query}\n\nKonteks:\n{context}"
    try:
        r = requests.post(
            api.ENRICH_ENDPOINT,
            headers={'Authorization': f'Bearer {api.ENRICH_API_KEY}',
                     'Content-Type': 'application/json',
                     'HTTP-Referer': 'https://pnj.ac.id', 'X-Title': 'PNJ RAFT Data Gen'},
            json={'model': api.ENRICH_MODEL,
                  'messages': [{'role': 'system', 'content': system},
                               {'role': 'user', 'content': user_content}],
                  'temperature': TEACHER_TEMP, 'max_tokens': TEACHER_MAXTOK,
                  # Qwen 35B = model thinking. Produksi (student) think=False → gold answer
                  # WAJIB non-thinking, kalau tidak student belajar distribusi salah.
                  'reasoning': {'enabled': False}},
            timeout=120)
        if r.status_code != 200:
            print(f'[WARN] teacher HTTP {r.status_code}: {r.text[:160]}', file=sys.stderr)
            return None
        msg = (r.json()['choices'][0]['message'].get('content') or '').strip()
        if not msg:
            print('[WARN] teacher content kosong (reasoning bocor / finish=length?)', file=sys.stderr)
            return None
        msg = re.sub(r'<think>.*?</think>', '', msg, flags=re.DOTALL).strip()
        return msg or None
    except Exception as e:
        print(f'[WARN] teacher error: {e}', file=sys.stderr)
        return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ MAIN                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-book',   type=int, default=380)
    ap.add_argument('--n-recom',  type=int, default=260)
    ap.add_argument('--n-info',   type=int, default=220)
    ap.add_argument('--n-hybrid', type=int, default=140)
    ap.add_argument('--dry-run',  action='store_true',
                    help='bangun query+konteks tanpa panggil OpenRouter (gratis, validasi struktur)')
    ap.add_argument('--limit',    type=int, default=0, help='batasi jumlah sample (debug)')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Sinkronisasi system prompt snapshot vs api.py — gagal keras jika drift.
    _assert_prompts_in_sync()

    print('▶ Memuat sistem RAG (retriever + ChromaDB + katalog)...')
    api.init_system()
    df = api._retriever.df
    kb_all = _all_kb_passages()
    print(f'  katalog={len(df)} buku | KB chunks={len(kb_all)}')

    held = load_heldout()
    print(f'▶ Held-out terkumpul: {len(held)} query (haram masuk training)')

    pool = build_query_pool(df, args.n_book, args.n_recom, args.n_info, args.n_hybrid, held)
    if args.limit:
        pool = pool[:args.limit]
    by_route = Counter(p['route'] for p in pool)
    print(f'▶ Query pool: {len(pool)} | {dict(by_route)}')

    # Tetapkan drop_golden secara deterministik per sample (P_DROP_GOLDEN).
    rng = random.Random(7)
    out_path = os.path.join(OUT_DIR, 'raft_train.jsonl')
    n_written = n_skip = n_drop = n_organic = 0
    t0 = time.time()
    with open(out_path, 'w', encoding='utf-8') as fout:
        for i, item in enumerate(pool, 1):
            drop = rng.random() < P_DROP_GOLDEN
            built = make_context(item, df, kb_all, drop_golden=drop)
            if built is None:
                n_skip += 1; continue
            system, ctx, meta = built
            if meta['drop_golden']:
                n_drop += 1
            if meta.get('organic_refusal'):
                n_organic += 1

            if args.dry_run:
                gold = '«DRY-RUN: gold answer di-skip»'
            else:
                gold = teacher_answer(system, item['query'], ctx)
                time.sleep(TEACHER_DELAY)
                if not gold:
                    n_skip += 1; continue

            rec = {
                'messages': [
                    {'role': 'system',    'content': system},
                    {'role': 'user',      'content': f"{item['query']}\n\nKonteks:\n{ctx}"},
                    {'role': 'assistant', 'content': gold},
                ],
                'meta': {'route': item['route'], 'query': item['query'], **meta},
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_written += 1
            if i % 25 == 0:
                rate = i / max(time.time() - t0, 1e-6)
                print(f'  [{i}/{len(pool)}] tulis={n_written} skip={n_skip} '
                      f'drop_golden={n_drop} | {rate:.1f} q/s')

    manifest = {
        'created'        : time.strftime('%Y-%m-%d %H:%M:%S'),
        'mode'           : 'dry-run' if args.dry_run else 'full',
        'teacher_model'  : api.ENRICH_MODEL,
        'params'         : {'P_DROP_GOLDEN': P_DROP_GOLDEN, 'N_DISTRACTOR': N_DISTRACTOR,
                            'seed': 42, 'teacher_temp': TEACHER_TEMP},
        'requested'      : {'book': args.n_book, 'recom': args.n_recom,
                            'info': args.n_info, 'hybrid': args.n_hybrid},
        'pool_by_route'  : dict(by_route),
        'written'        : n_written, 'skipped': n_skip, 'drop_golden': n_drop,
        'organic_refusal': n_organic, 'min_golden_score': MIN_GOLDEN_SCORE,
        'heldout_count'  : len(held),
        'output'         : os.path.relpath(out_path, ROOT),
    }
    with open(os.path.join(OUT_DIR, 'raft_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'\n✓ Selesai. ditulis={n_written} skip={n_skip} drop_golden={n_drop} '
          f'organic_refusal={n_organic}')
    print(f'  → {out_path}')
    print(f'  → {os.path.join(OUT_DIR, "raft_manifest.json")}')
    if args.dry_run:
        print('\n[DRY-RUN] gold answer belum digenerate. Cek struktur, lalu jalankan tanpa --dry-run.')


def _assert_prompts_in_sync():
    """Pastikan snapshot system prompt identik dengan api.generate_llm (cegah drift diam-diam)."""
    src = open(os.path.join(ROOT, 'api.py')).read()
    for key, prompt in _SYSTEM_PROMPTS.items():
        # ambil kalimat penanda unik tiap prompt
        marker = prompt.split('. ')[1][:40]
        if marker not in src:
            raise SystemExit(
                f'[FATAL] system prompt "{key}" tidak cocok dengan api.py (marker: {marker!r}). '
                f'Update _SYSTEM_PROMPTS agar identik dengan api.generate_llm sebelum lanjut.')


if __name__ == '__main__':
    main()
