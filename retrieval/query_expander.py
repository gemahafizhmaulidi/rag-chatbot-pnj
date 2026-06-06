"""
query_expander.py — [SEBAGIAN DEPRECATED]

`expand_query()` (pemetaan mood/minat/topik berbasis kamus hardcoded) TIDAK LAGI
dipakai di produksi sejak ekspansi query dialihkan ke LLM-router (router.py meng-
ekstrak `search_query`, mis. "AI" -> "kecerdasan buatan machine learning"). Alasan:
pemetaan mood->genre (mis. 'bosan'->novel) tidak memiliki dasar empiris dan sulit
dipertanggungjawabkan; untuk query samar tanpa topik, sistem kini meminta klarifikasi.
Kamus & expand_query DIPERTAHANKAN sebagai dokumentasi pendekatan lama (baseline),
bukan jalur aktif.

YANG MASIH DIPAKAI: `detect_beginner_intent()` — deteksi level pemula (kata kunci
sederhana, bukan pemetaan topik) untuk hint konteks LLM.
"""

from __future__ import annotations
import re

# ── Topic Expansion ────────────────────────────────────────────────────────────
# keyword user → expanded search terms untuk BM25+Dense retrieval

TOPIC_EXPANSION: dict[str, list[str]] = {
    # Teknologi & IT
    'python':           ['python', 'pemrograman python', 'scripting'],
    'java':             ['java', 'pemrograman java', 'object oriented'],
    'web':              ['web', 'pemrograman web', 'html', 'javascript'],
    'database':         ['database', 'basis data', 'sql', 'sistem basis data'],
    'jaringan':         ['jaringan', 'jaringan komputer', 'networking', 'telekomunikasi'],
    'ai':               ['kecerdasan buatan', 'artificial intelligence', 'machine learning'],
    'machine learning': ['machine learning', 'deep learning', 'kecerdasan buatan', 'data mining'],
    'data mining':      ['data mining', 'machine learning', 'data science', 'analisis data'],
    'algoritma':        ['algoritma', 'struktur data', 'pemrograman', 'komputasi'],
    'android':          ['android', 'mobile', 'flutter', 'kotlin', 'pemrograman mobile'],
    'keamanan':         ['keamanan', 'kriptografi', 'cybersecurity', 'keamanan jaringan'],
    'embedded':         ['embedded', 'mikrokontroler', 'arduino', 'sistem tertanam'],
    'data science':     ['data science', 'analisis data', 'machine learning', 'statistika'],

    # Teknik
    'elektronika':      ['elektronika', 'rangkaian elektronika', 'teknik listrik'],
    'mekanika':         ['mekanika', 'mekanika teknik', 'dinamika', 'statika'],
    'termodinamika':    ['termodinamika', 'perpindahan panas', 'energi'],
    'konstruksi':       ['konstruksi', 'beton', 'pondasi', 'struktur bangunan'],
    'sipil':            ['sipil', 'konstruksi', 'beton', 'hidrolika', 'jalan'],

    # Bisnis & Manajemen
    'akuntansi':        ['akuntansi', 'auditing', 'perpajakan', 'keuangan'],
    'manajemen':        ['manajemen', 'organisasi', 'kepemimpinan', 'administrasi'],
    'pemasaran':        ['pemasaran', 'marketing', 'manajemen pemasaran'],
    'bisnis':           ['bisnis', 'kewirausahaan', 'entrepreneur', 'manajemen bisnis'],
    'kewirausahaan':    ['kewirausahaan', 'entrepreneurship', 'bisnis', 'startup'],
    'investasi':        ['investasi', 'keuangan', 'saham', 'ekonomi'],

    # Ilmu Dasar
    'matematika':       ['matematika', 'kalkulus', 'aljabar', 'statistika'],
    'statistik':        ['statistika', 'statistik', 'probabilitas', 'analisis data'],
    'fisika':           ['fisika', 'mekanika', 'termodinamika', 'gelombang'],
    'kimia':            ['kimia', 'kimia organik', 'kimia analitik'],

    # Pengembangan Diri
    'komunikasi':       ['komunikasi', 'public speaking', 'interpersonal', 'retorika'],
    'kepemimpinan':     ['kepemimpinan', 'leadership', 'manajemen', 'motivasi'],
    'motivasi':         ['motivasi', 'pengembangan diri', 'self improvement', 'sukses'],
    'psikologi':        ['psikologi', 'perilaku manusia', 'psikologi sosial', 'kepribadian'],

    # Kuliner & Gaya Hidup
    'masak':            ['masak', 'memasak', 'kuliner', 'resep', 'tata boga', 'gizi', 'makanan'],
    'kuliner':          ['kuliner', 'masak', 'memasak', 'resep', 'tata boga'],
    'gizi':             ['gizi', 'nutrisi', 'makanan', 'kesehatan', 'tata boga'],
    'kesehatan':        ['kesehatan', 'gizi', 'kedokteran', 'keperawatan'],

    # Bahasa & Sastra
    'bahasa inggris':   ['bahasa inggris', 'english', 'grammar', 'toefl'],
    'bahasa':           ['bahasa', 'linguistik', 'sastra', 'komunikasi'],
    'menulis':          ['menulis', 'penulisan', 'jurnalistik', 'karya tulis'],

    # Sejarah & Sosial
    'sejarah':          ['sejarah', 'history', 'peradaban', 'biografi', 'tokoh'],
    'hukum':            ['hukum', 'perundang-undangan', 'regulasi', 'tata negara'],
    'sosiologi':        ['sosiologi', 'masyarakat', 'sosial', 'ilmu sosial'],

    # Desain & Seni
    'desain':           ['desain grafis', 'desain komunikasi visual', 'tipografi', 'ilustrasi'],
    'fotografi':        ['fotografi', 'kamera', 'editing foto'],
}

# ── Mood/Kondisi → Topik ───────────────────────────────────────────────────────
# Untuk open_ended_recommendation: deteksi kondisi/mood user → kategori buku

MOOD_TO_TOPICS: dict[str, list[str]] = {
    'bosen':        ['novel', 'fiksi', 'cerita', 'biografi', 'motivasi', 'pengembangan diri'],
    'bosan':        ['novel', 'fiksi', 'cerita', 'biografi', 'motivasi', 'pengembangan diri'],
    'gabut':        ['novel', 'fiksi', 'cerita', 'motivasi', 'sains populer'],
    'iseng':        ['novel', 'fiksi', 'cerita', 'motivasi'],
    'stres':        ['motivasi', 'self improvement', 'psikologi', 'pengembangan diri'],
    'stress':       ['motivasi', 'self improvement', 'psikologi', 'pengembangan diri'],
    'cape':         ['motivasi', 'self improvement', 'psikologi', 'inspirasi'],
    'capek':        ['motivasi', 'self improvement', 'psikologi', 'inspirasi'],
    'sedih':        ['motivasi', 'self improvement', 'psikologi', 'novel'],
    'galau':        ['novel', 'motivasi', 'psikologi', 'pengembangan diri'],
    'semangat':     ['motivasi', 'bisnis', 'kewirausahaan', 'kepemimpinan', 'sukses'],
    'penasaran':    ['sains', 'sejarah', 'filsafat', 'teknologi'],

    # Kondisi personal — arahkan ke pengembangan diri, TIDAK diagnosis psikologis
    'pendiam':      ['komunikasi', 'public speaking', 'percaya diri', 'interpersonal', 'kepribadian'],
    'pemalu':       ['komunikasi', 'percaya diri', 'public speaking', 'interpersonal'],
    'kesepian':     ['pengembangan diri', 'komunikasi', 'psikologi', 'motivasi'],
    'overthinking': ['psikologi', 'self improvement', 'motivasi', 'mindfulness'],
}

# ── Minat Eksplisit → Topik ────────────────────────────────────────────────────

INTEREST_TO_TOPICS: dict[str, list[str]] = {
    'masak':    TOPIC_EXPANSION['masak'],
    'memasak':  TOPIC_EXPANSION['masak'],
    'kuliner':  TOPIC_EXPANSION['kuliner'],
    'desain':   TOPIC_EXPANSION['desain'],
    'fotografi': TOPIC_EXPANSION['fotografi'],
    'bisnis':   TOPIC_EXPANSION['bisnis'],
    'investasi': TOPIC_EXPANSION['investasi'],
    'sejarah':  TOPIC_EXPANSION['sejarah'],
    'psikologi': TOPIC_EXPANSION['psikologi'],
    'musik':    ['musik', 'seni musik', 'teori musik'],
    'olahraga': ['olahraga', 'kesehatan', 'kebugaran'],
}

# ── Beginner Keywords ─────────────────────────────────────────────────────────

BEGINNER_KW: list[str] = [
    'pemula', 'belajar', 'dasar', 'pengantar', 'introduction',
    'introductory', 'beginner', 'fundamental', 'basic',
    'untuk pemula', 'baru belajar', 'mulai belajar', 'awal',
]

# ── Noise pattern khusus untuk recommendation context ─────────────────────────

_RECOM_NOISE = re.compile(
    r'\b(rekomendasikan|rekomendasiin|rekomendasi|sarankan|saran|'
    r'buku|dong|nih|sih|deh|kak|bang|tolong|minta|carikan|ada|'
    r'lagi|gua|gue|gw|cape|capek|bosen|bosan|gabut|suka|pengen|'
    r'mau|cocok|bagus|terbaik|untuk|yang|aku|saya|gw)\b',
    re.IGNORECASE
)


# ── Core Functions ────────────────────────────────────────────────────────────

def detect_beginner_intent(query: str) -> bool:
    ql = query.strip().lower()
    return any(kw in ql for kw in BEGINNER_KW)


def expand_query(query: str) -> tuple[str, str]:
    """
    Expand query recommendation menjadi retrieval query yang lebih kaya.

    Returns:
        expanded_query  : string untuk diserahkan ke retriever
        interpretation  : teks penjelasan interpretasi (disertakan di context LLM)
    """
    ql = query.strip().lower()

    # 1. Deteksi mood/kondisi
    for mood, topics in MOOD_TO_TOPICS.items():
        if mood in ql:
            expanded = ' '.join(topics[:5])
            interp = (
                f"Berdasarkan pertanyaan kamu, saya arahkan ke buku "
                f"{', '.join(topics[:3])}."
            )
            return expanded, interp

    # 2. Deteksi minat eksplisit (INTEREST_TO_TOPICS lebih spesifik)
    for interest, topics in INTEREST_TO_TOPICS.items():
        if interest in ql:
            expanded = ' '.join(topics[:6])
            interp = f"Mencari buku tentang {interest}: {', '.join(topics[:3])}."
            return expanded, interp

    # 3. Deteksi topik dari TOPIC_EXPANSION
    for keyword, expansions in TOPIC_EXPANSION.items():
        if keyword in ql:
            expanded = ' '.join(expansions[:6])
            interp = f"Mencari buku tentang {keyword}: {', '.join(expansions[:3])}."
            return expanded, interp

    # 4. Fallback: bersihkan noise, pakai sisa teks
    cleaned = _RECOM_NOISE.sub(' ', ql)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(' .,?!')
    if len(cleaned) >= 3:
        return cleaned, f"Mencari buku terkait: {cleaned}."

    # 5. Tidak ada sinyal apapun
    return query, ""
