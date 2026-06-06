"""core/config.py — Konfigurasi terpusat sistem RAG Chatbot PNJ"""

import os, json, logging
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Database ──────────────────────────────────────────────────────────────────
DB_CONFIG = dict(
    host     = os.getenv('DB_HOST',     'localhost'),
    user     = os.getenv('DB_USER',     'root'),
    password = os.getenv('DB_PASSWORD', ''),
    database = os.getenv('DB_NAME2',    'opacv2'),
)

# ── LLM (Ollama) ──────────────────────────────────────────────────────────────
LLM_MODEL         = os.getenv('OLLAMA_MODEL',    'qwen3.5:4b')
LLM_ENDPOINT      = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434/api/generate')
LLM_CHAT_ENDPOINT = 'http://localhost:11434/api/chat'
LLM_TEMP          = 0.2
LLM_MAX_TOKENS    = 400

# ── Enrichment (OpenRouter — untuk pipeline pengayaan metadata, bukan produksi) ──
ENRICH_MODEL          = os.getenv('OPENROUTER_MODEL',   'qwen/qwen3.5-35b-a3b')
ENRICH_ENDPOINT       = 'https://openrouter.ai/api/v1/chat/completions'
ENRICH_API_KEY        = os.getenv('OPENROUTER_API_KEY', '')
ENRICH_TEMP           = 0.3
ENRICH_MAX_TOKENS     = 4000
GOOGLE_BOOKS_URL      = 'https://www.googleapis.com/books/v1/volumes'
OPEN_LIBRARY_URL      = 'https://openlibrary.org/api/books'
ENRICH_REQUEST_TIMEOUT = 5
ENRICH_API_DELAY       = 0.4

# ── Knowledge Base (ChromaDB) ─────────────────────────────────────────────────
CHROMA_DIR        = os.path.join(BASE_DIR, 'knowledge_base', 'chroma_db')
CHROMA_COLLECTION = 'kb_perpustakaan_pnj'
KB_TOP_K          = 12

# ── Thresholds ────────────────────────────────────────────────────────────────
CAL_PATH    = os.path.join(BASE_DIR, 'retrieval', 'output', 'calibration', 'thresholds.json')
EMBED_CACHE = os.path.join(BASE_DIR, 'retrieval', 'output', 'embeddings.npy')


def load_thresholds():
    if os.path.exists(CAL_PATH):
        with open(CAL_PATH) as f:
            cal = json.load(f)
        rt = cal.get('reranker_threshold', 0.0)
        ht = cal.get('hardstop_threshold', 0.55)
        log.info(f'Threshold dari kalibrasi: reranker={rt:.4f}, hardstop={ht:.4f}')
        return rt, ht
    log.warning('thresholds.json tidak ditemukan — pakai default (0.0, 0.55)')
    return 0.0, 0.55


RERANKER_THRESHOLD, _KB_DIST_RAW = load_thresholds()
# Hard-stop BACKSTOP permisif: deteksi OOS utama di LLM-router.
# Dilonggarkan dari 0.4108 → 0.45 berdasarkan audit_hardstop.py.
KB_DIST_THRESHOLD = 0.45
