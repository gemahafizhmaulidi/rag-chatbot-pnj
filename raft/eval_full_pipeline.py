"""
raft/eval_full_pipeline.py — RAFT Full Pipeline Evaluation

Membandingkan Base Qwen3.5-4B vs RAFT merged model pada pipeline RAG lengkap:
  routing → retrieval (MySQL catalog + ChromaDB KB) → generate

Jalankan dari root folder rag-chatbot-pnj:
    python raft/eval_full_pipeline.py --raft-path /path/to/raft_merged

Output: raft/out/eval_full_results.json → gunakan judge_pairwise.py untuk penilaian
"""

import os, sys, json, time, re, argparse, logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Tambah root ke sys.path supaya bisa import modul produksi
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import DB_CONFIG, CHROMA_DIR, CHROMA_COLLECTION, KB_TOP_K, KB_DIST_THRESHOLD
from core.prompts import SYS_PROMPTS
from core.db import load_catalog
from retrieval.retriever import HybridRetriever
from retrieval.query_expander import detect_beginner_intent
from routing.router import route_query
import mysql.connector
import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger(__name__)

BASE_MODEL  = 'Qwen/Qwen3.5-4B'
DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'ground_truth_ragas.json')
OUT_PATH    = os.path.join(os.path.dirname(__file__), 'out', 'eval_full_results.json')
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.2
TOP_K          = 20
TOP_P          = 0.95

# ── Init RAG system (shared, dipakai kedua arm) ───────────────────────────────

_embed_model = None
_retriever   = None
_chroma_col  = None


def init_rag():
    global _embed_model, _retriever, _chroma_col
    log.info('Loading bge-m3 embedding model...')
    _embed_model = SentenceTransformer('BAAI/bge-m3')

    log.info('Connecting to MySQL catalog...')
    conn = mysql.connector.connect(**DB_CONFIG)
    df   = load_catalog(conn)
    conn.close()

    _retriever = HybridRetriever(
        df, _embed_model, reranker=None,
        bm25_weight=0.45, dense_weight=0.55,
        reranker_threshold=0.0,
    )

    log.info('Loading ChromaDB KB...')
    client      = chromadb.PersistentClient(path=CHROMA_DIR)
    _chroma_col = client.get_collection(CHROMA_COLLECTION)
    log.info(f'KB: {_chroma_col.count()} chunks | Katalog: {len(df):,} buku')


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def search_kb(query: str) -> dict:
    q_emb = _embed_model.encode([query], normalize_embeddings=True).tolist()
    res   = _chroma_col.query(
        query_embeddings=q_emb, n_results=KB_TOP_K,
        include=['documents', 'metadatas', 'distances'],
    )
    passages, best_dist = [], 1.0
    if res and res['documents'] and res['documents'][0]:
        for doc, meta, dist in zip(res['documents'][0], res['metadatas'][0], res['distances'][0]):
            passages.append({'text': doc, 'source': meta.get('source', ''), 'distance': round(dist, 4)})
        best_dist = res['distances'][0][0]
    return {'passages': passages, 'best_distance': best_dist, 'relevant': best_dist <= KB_DIST_THRESHOLD}


def build_book_context(books: list) -> str:
    lines = []
    for i, b in enumerate(books, 1):
        entry = [f"[{i}] {b.get('judul', '-')}"]
        entry.append(f"    Penulis       : {b.get('penulis', '-') or '-'}")
        entry.append(f"    Nomor Panggil : {b.get('call_number', '-') or '-'}")
        total    = int(b.get('total_eksemplar', 0) or 0)
        tersedia = int(b.get('tersedia', 0) or 0)
        if total > 0:
            entry.append(f"    Stok      : {tersedia} dari {total} eksemplar tersedia")
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


def get_context(query: str) -> tuple[str, str]:
    """Jalankan routing + retrieval, kembalikan (context_string, query_type)."""
    d     = route_query(query)
    route = d['route']

    if route in ('greeting', 'oos'):
        return '', route

    if route == 'stats':
        return '', 'stats'

    if route == 'recommendation':
        if not d['search_query']:
            return '', 'clarification'
        books = _retriever.search_no_rerank(d['search_query'], top_k=10)
        available   = sorted([b for b in books if b.get('tersedia', 0) > 0],
                             key=lambda b: b.get('loan_count', 0), reverse=True)
        unavailable = sorted([b for b in books if b.get('tersedia', 0) == 0],
                             key=lambda b: b.get('loan_count', 0), reverse=True)
        top5    = (available + unavailable)[:5]
        context = build_book_context(top5)
        if detect_beginner_intent(query):
            context += "\n\n[Catatan: user mencari buku untuk pemula]"
        return context, 'recommendation_search'

    if route == 'book_search':
        if not d['search_query']:
            return '', 'clarification'
        books = _retriever.search_no_rerank(d['search_query'], top_k=5)
        return build_book_context(books), 'book_search'

    if route == 'hybrid':
        books = _retriever.search_no_rerank(d['search_query'] or query, top_k=3)
        kb    = search_kb(d['info_query'] or query)
        if books and kb['relevant']:
            ctx = "=== KATALOG BUKU ===\n" + build_book_context(books) + \
                  "\n\n=== INFORMASI LAYANAN ===\n" + build_kb_context(kb['passages'])
            return ctx, 'hybrid'
        if books:
            return build_book_context(books), 'book_search'
        if kb['relevant']:
            return build_kb_context(kb['passages']), 'general_info'
        return '', 'oos'

    # general_info
    kb = search_kb(d['info_query'] or query)
    if not kb['relevant']:
        return '', 'oos'
    return build_kb_context(kb['passages']), 'general_info'


# ── HuggingFace generate ──────────────────────────────────────────────────────

def load_hf_model(model_path: str):
    log.info(f'Loading model: {model_path}')
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
    )
    model.eval()
    log.info(f'  ✓ {model_path} | device={next(model.parameters()).device}')
    return model, tok


def build_prompt(tok, query: str, context: str, query_type: str) -> str:
    sys_prompt   = SYS_PROMPTS.get(query_type, SYS_PROMPTS['general_info'])
    user_content = f"{query}\n\nKonteks:\n{context}" if context.strip() else query
    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user',   'content': user_content},
    ]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_hf(model, tok, prompt: str) -> str:
    inputs    = tok(prompt, return_tensors='pt').to(model.device)
    input_len = inputs['input_ids'].shape[1]

    # Stop at im_end token (Qwen end-of-turn) untuk cegah multi-turn bleed
    im_end_id = tok.convert_tokens_to_ids('<|im_end|>')
    stop_ids  = [tok.eos_token_id]
    if im_end_id and im_end_id != tok.eos_token_id:
        stop_ids.append(im_end_id)

    out = model.generate(
        **inputs,
        max_new_tokens  = MAX_NEW_TOKENS,
        temperature     = TEMPERATURE,
        top_k           = TOP_K,
        top_p           = TOP_P,
        do_sample       = True,
        eos_token_id    = stop_ids,
        pad_token_id    = tok.eos_token_id,
    )
    new_tokens = out[0][input_len:]
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    # Potong di batas turn pertama jika masih ada sisa
    for marker in ['\nuser\n', '\n<|im_start|>user', '\nassistant\n<|im_start|>']:
        if marker in text:
            text = text[:text.index(marker)].strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_eval(raft_model_path: str):
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Load data
    data = json.load(open(DATA_PATH, encoding='utf-8'))
    log.info(f'Loaded {len(data)} queries dari {DATA_PATH}')

    # Init RAG system
    init_rag()

    # Pre-fetch context untuk semua query (dilakukan sekali, shared untuk kedua arm)
    log.info('Pre-fetching context via full pipeline...')
    contexts = []
    for i, item in enumerate(data):
        ctx, qt = get_context(item['question'])
        contexts.append({'context': ctx, 'query_type': qt})
        log.info(f'  [{i+1}/{len(data)}] {qt} | ctx_len={len(ctx)} | {item["question"][:50]}')

    results = []

    # ── Arm A: Base model ─────────────────────────────────────────────────────
    log.info(f'\n=== ARM A: {BASE_MODEL} ===')
    model_a, tok_a = load_hf_model(BASE_MODEL)
    answers_a = []
    for i, (item, ctx_info) in enumerate(zip(data, contexts)):
        prompt = build_prompt(tok_a, item['question'], ctx_info['context'], ctx_info['query_type'])
        t0     = time.time()
        ans    = generate_hf(model_a, tok_a, prompt)
        answers_a.append(ans)
        log.info(f'  [A][{i+1}/{len(data)}] {time.time()-t0:.1f}s | {ans[:60]!r}')
    del model_a, tok_a
    torch.cuda.empty_cache()
    log.info('✓ Arm A selesai')

    # ── Arm B: RAFT model ─────────────────────────────────────────────────────
    log.info(f'\n=== ARM B: {raft_model_path} ===')
    model_b, tok_b = load_hf_model(raft_model_path)
    answers_b = []
    for i, (item, ctx_info) in enumerate(zip(data, contexts)):
        prompt = build_prompt(tok_b, item['question'], ctx_info['context'], ctx_info['query_type'])
        t0     = time.time()
        ans    = generate_hf(model_b, tok_b, prompt)
        answers_b.append(ans)
        log.info(f'  [B][{i+1}/{len(data)}] {time.time()-t0:.1f}s | {ans[:60]!r}')
    del model_b, tok_b
    torch.cuda.empty_cache()
    log.info('✓ Arm B selesai')

    # ── Gabung & simpan ───────────────────────────────────────────────────────
    for item, ctx_info, ans_a, ans_b in zip(data, contexts, answers_a, answers_b):
        results.append({
            'question'    : item['question'],
            'ground_truth': item.get('ground_truth', ''),
            'query_type'  : ctx_info['query_type'],
            'context'     : ctx_info['context'],
            'answer_a'    : ans_a,
            'answer_b'    : ans_b,
            'model_a'     : BASE_MODEL,
            'model_b'     : raft_model_path,
        })

    json.dump(results, open(OUT_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    log.info(f'\n✓ Saved {len(results)} hasil ke {OUT_PATH}')

    # Summary
    empty_ctx  = sum(1 for r in results if not r['context'].strip())
    oos_count  = sum(1 for r in results if r['query_type'] == 'oos')
    book_count = sum(1 for r in results if r['query_type'] in ('book_search', 'recommendation_search'))
    info_count = sum(1 for r in results if r['query_type'] == 'general_info')
    log.info(f'Distribusi: book/rec={book_count}, general_info={info_count}, oos={oos_count}, empty_ctx={empty_ctx}')
    log.info(f'Download {OUT_PATH} → jalankan judge_pairwise.py di lokal')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raft-path', required=True,
                        help='Path ke folder raft_merged (hasil QLoRA training)')
    args = parser.parse_args()
    run_eval(args.raft_path)
