"""enrichment/enricher.py — Pipeline pengayaan metadata buku (preview only, tidak tulis DB)"""

import re, time, json, logging
import mysql.connector
import requests

from core.config import (
    DB_CONFIG, ENRICH_MODEL, ENRICH_ENDPOINT, ENRICH_API_KEY,
    ENRICH_TEMP, ENRICH_MAX_TOKENS, ENRICH_REQUEST_TIMEOUT, ENRICH_API_DELAY,
    GOOGLE_BOOKS_URL, OPEN_LIBRARY_URL,
)

log = logging.getLogger(__name__)


def _clean_isbn(raw):
    if not raw:
        return None, None
    cleaned = re.sub(r'[^0-9X]', '', str(raw).upper())
    isbn13 = isbn10 = None
    m13 = re.search(r'(97[89]\d{10})', cleaned)
    if m13:
        isbn13 = m13.group(1)
    m10 = re.search(r'(\d{9}[\dX])', cleaned)
    if m10:
        isbn10 = m10.group(1)
    if isbn10 and not isbn13:
        try:
            digits = '978' + isbn10[:9]
            s = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits))
            candidate = digits + str((10 - s % 10) % 10)
            chk = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(candidate[:12]))
            if (10 - chk % 10) % 10 == int(candidate[-1]):
                isbn13 = candidate
        except Exception:
            pass
    return isbn10, isbn13


def _clean_desc(text):
    if not text:
        return ''
    t = re.sub(r'^(Deskripsi\s*[:\-]\s*|PREFACE\s*[:\-]?\s*)', '', str(text).strip(), flags=re.IGNORECASE)
    t = t.replace('\xb6', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&nbsp;', ' ')
    t = re.sub(r'\s+', ' ', t).strip()
    return t if len(t) >= 50 else ''


def _title_match(api_title, our_title):
    a, b = api_title.lower().strip(), our_title.lower().strip()
    if b[:8] in a or a[:8] in b:
        return True
    stop = {'dan', 'the', 'and', 'for', 'dari', 'yang', 'dengan', 'dalam', 'atau', 'buku'}
    wa = {w for w in a.split() if len(w) > 4 and w not in stop}
    wb = {w for w in b.split() if len(w) > 4 and w not in stop}
    return len(wa & wb) >= 2


def _fetch_gb_isbn(isbn13, judul):
    try:
        r = requests.get(GOOGLE_BOOKS_URL, params={'q': f'isbn:{isbn13}'}, timeout=ENRICH_REQUEST_TIMEOUT)
        for item in r.json().get('items', []):
            desc = item.get('volumeInfo', {}).get('description', '')
            t    = item.get('volumeInfo', {}).get('title', '')
            if desc and len(desc) > 60 and _title_match(t, judul):
                return _clean_desc(desc)
    except Exception:
        pass
    return None


def _fetch_gb_title(judul, penulis):
    try:
        q = f'intitle:{judul}'
        if penulis:
            q += f'+inauthor:{penulis.split(",")[0].strip()}'
        r = requests.get(GOOGLE_BOOKS_URL, params={'q': q, 'maxResults': 5}, timeout=ENRICH_REQUEST_TIMEOUT)
        for item in r.json().get('items', []):
            vi   = item.get('volumeInfo', {})
            desc = vi.get('description', '')
            t    = vi.get('title', '')
            if desc and len(desc) > 60 and _title_match(t, judul):
                return _clean_desc(desc)
    except Exception:
        pass
    return None


def _fetch_ol_isbn(isbn13, judul):
    try:
        r = requests.get(OPEN_LIBRARY_URL,
                         params={'bibkeys': f'ISBN:{isbn13}', 'format': 'json', 'jscmd': 'data'},
                         timeout=ENRICH_REQUEST_TIMEOUT)
        data = r.json().get(f'ISBN:{isbn13}', {})
        desc = data.get('description', {})
        if isinstance(desc, dict):
            desc = desc.get('value', '')
        t = data.get('title', '')
        if desc and len(str(desc)) > 60 and _title_match(t, judul):
            return _clean_desc(str(desc))
    except Exception:
        pass
    return None


def _fetch_ol_title(judul, penulis):
    try:
        r = requests.get('https://openlibrary.org/search.json',
                         params={'title': judul, 'author': penulis, 'limit': 3},
                         timeout=ENRICH_REQUEST_TIMEOUT)
        for doc in r.json().get('docs', []):
            key = doc.get('key', '')
            t   = doc.get('title', '')
            if not _title_match(t, judul):
                continue
            r2   = requests.get(f'https://openlibrary.org{key}.json', timeout=ENRICH_REQUEST_TIMEOUT)
            desc = r2.json().get('description', '')
            if isinstance(desc, dict):
                desc = desc.get('value', '')
            if desc and len(str(desc)) > 60:
                return _clean_desc(str(desc))
    except Exception:
        pass
    return None


def _build_llm_prompt(judul, penulis, tahun, call_number, topik, penerbit):
    metadata = f'Judul: {judul}'
    if penulis:   metadata += f'\nPenulis: {penulis}'
    if tahun and str(tahun) not in ('0000', '', 'None', 'nan'):
        metadata += f'\nTahun: {tahun}'
    if penerbit:     metadata += f'\nPenerbit: {penerbit}'
    if call_number:  metadata += f'\nKlasifikasi DDC: {call_number}'
    if topik:        metadata += f'\nTopik/kata kunci: {topik}'
    system = (
        'Kamu adalah pustakawan senior dan pakar katalogisasi buku akademik Indonesia. '
        'Tugas: tulis deskripsi buku untuk katalog perpustakaan digital berbasis RAG.\n\n'
        'ATURAN:\n'
        '1. Sebutkan konsep, topik, dan istilah teknis SPESIFIK yang relevan dengan judul dan DDC\n'
        '2. Sebutkan metode, pendekatan, atau tools yang lazim digunakan\n'
        '3. Sertakan kata kunci teknis yang pengguna gunakan saat mencari buku ini\n'
        '4. HINDARI: "buku ini membahas", "memberikan pemahaman mendalam", "sangat bermanfaat"\n'
        '5. Tulis 3-5 kalimat padat, konkret, dalam Bahasa Indonesia'
    )
    return system, metadata


def _llm_stream(judul, penulis='', tahun='', call_number='', topik='', penerbit=''):
    if not ENRICH_API_KEY:
        log.warning('OPENROUTER_API_KEY tidak di-set — skip LLM enrichment')
        return
    system, metadata = _build_llm_prompt(judul, penulis, tahun, call_number, topik, penerbit)
    try:
        r = requests.post(
            ENRICH_ENDPOINT,
            headers={
                'Authorization': f'Bearer {ENRICH_API_KEY}',
                'Content-Type' : 'application/json',
                'HTTP-Referer' : 'https://pnj.ac.id',
                'X-Title'      : 'PNJ Library Enrichment',
            },
            json={
                'model'      : ENRICH_MODEL,
                'messages'   : [
                    {'role': 'system', 'content': system},
                    {'role': 'user',   'content': f'Tulis deskripsi katalog untuk buku berikut:\n\n{metadata}'},
                ],
                'temperature': ENRICH_TEMP,
                'max_tokens' : ENRICH_MAX_TOKENS,
                'stream'     : True,
            },
            stream=True, timeout=90,
        )
        for line in r.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            if not line.startswith('data: '):
                continue
            raw = line[6:].strip()
            if raw == '[DONE]':
                break
            try:
                chunk = json.loads(raw)
                token = chunk['choices'][0]['delta'].get('content') or ''
                if token:
                    yield token
            except Exception:
                continue
    except Exception as e:
        log.warning('LLM stream enrichment error: %s', e)


def enrich_stream(biblio_id: int):
    """
    Generator SSE: yield progress events per step, lalu hasil akhir.
    Tidak menyimpan ke DB — preview only. SLIMS klik Save untuk simpan.
    """
    def sse(payload):
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield sse({'step': 'init', 'msg': '📖 Mengambil data buku dari database...'})
    try:
        conn   = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            'SELECT b.biblio_id, b.title, b.publish_year, b.isbn_issn, b.notes, b.call_number, '
            'GROUP_CONCAT(DISTINCT a.author_name SEPARATOR ", ") AS penulis, '
            'GROUP_CONCAT(DISTINCT t.topic SEPARATOR ", ") AS topik, '
            'p.publisher_name AS penerbit '
            'FROM biblio b '
            'LEFT JOIN biblio_author ba ON b.biblio_id = ba.biblio_id '
            'LEFT JOIN mst_author a ON ba.author_id = a.author_id '
            'LEFT JOIN biblio_topic bt ON b.biblio_id = bt.biblio_id '
            'LEFT JOIN mst_topic t ON bt.topic_id = t.topic_id '
            'LEFT JOIN mst_publisher p ON b.publisher_id = p.publisher_id '
            'WHERE b.biblio_id = %s GROUP BY b.biblio_id',
            (biblio_id,)
        )
        row = cursor.fetchone()
        cursor.close(); conn.close()
    except Exception as e:
        yield sse({'step': 'error', 'msg': f'❌ Gagal koneksi DB: {e}'}); return

    if not row:
        yield sse({'step': 'error', 'msg': f'❌ biblio_id {biblio_id} tidak ditemukan'}); return

    judul    = str(row['title'] or '')
    penulis  = str(row['penulis'] or '')
    tahun    = str(row['publish_year'] or '')
    isbn_raw = str(row['isbn_issn'] or '')
    call_num = str(row['call_number'] or '')
    topik    = str(row['topik'] or '')
    penerbit = str(row['penerbit'] or '')

    isbn10, isbn13 = _clean_isbn(isbn_raw)
    desc = source = None

    steps = [
        ('Google Books (ISBN)',  lambda: _fetch_gb_isbn(isbn13, judul) if isbn13 else None),
        ('Google Books (Judul)', lambda: _fetch_gb_title(judul, penulis)),
        ('Open Library (ISBN)',  lambda: _fetch_ol_isbn(isbn13, judul) if isbn13 else None),
        ('Open Library (Judul)', lambda: _fetch_ol_title(judul, penulis)),
    ]

    for src_name, fetcher in steps:
        if desc:
            break
        if 'ISBN' in src_name and not isbn13:
            yield sse({'step': 'skip', 'source': src_name, 'msg': f'⏭️ ISBN tidak tersedia — lewati {src_name}'})
            continue
        yield sse({'step': 'try', 'source': src_name, 'msg': f'🔍 Mencari di {src_name}...'})
        time.sleep(ENRICH_API_DELAY)
        desc = fetcher()
        if desc:
            source = src_name
            yield sse({'step': 'found', 'source': src_name, 'msg': f'✅ Deskripsi ditemukan di {src_name}'})
        else:
            yield sse({'step': 'miss', 'source': src_name, 'msg': f'⚠️ Tidak ditemukan di {src_name}'})

    if not desc:
        model_short = ENRICH_MODEL.split('/')[-1]
        yield sse({'step': 'try', 'source': 'LLM', 'msg': f'🤖 Generating deskripsi dengan AI ({model_short})...'})
        tokens = []
        for token in _llm_stream(judul, penulis, tahun, call_num, topik, penerbit):
            tokens.append(token)
            yield sse({'step': 'token', 'text': token})
        desc = _clean_desc(''.join(tokens)) or None
        if desc:
            source = f'LLM ({ENRICH_MODEL})'
            yield sse({'step': 'found', 'source': source, 'msg': f'✅ Deskripsi digenerate oleh AI ({model_short})'})
        else:
            yield sse({'step': 'miss', 'source': 'LLM', 'msg': '❌ AI gagal generate deskripsi'})

    if not desc:
        yield sse({'step': 'done', 'status': 'failed', 'msg': '❌ Tidak dapat generate deskripsi dari semua sumber'})
        return

    log.info('Enrich preview biblio_id=%d via %s (%d karakter)', biblio_id, source, len(desc))
    yield sse({'step': 'done', 'status': 'success', 'source': source, 'deskripsi': desc, 'judul': judul})
