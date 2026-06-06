"""
08_ragas_eval.py — Evaluasi RAGAS untuk Chatbot Perpustakaan PNJ (v3)
=====================================================================
Metrik:
  - Faithfulness    : jawaban hanya berisi klaim yang ada di konteks?
  - Answer Relevancy: jawaban relevan dengan pertanyaan?
  - Hard Stop Accuracy: pertanyaan OOS ditolak dengan benar?

Prasyarat:
  - api.py harus running: python api.py
  - Ollama harus running dengan model qwen3:4b

Jalankan:
    python scripts/08_ragas_eval.py

Output:
  output/ragas_results.csv
  output/ragas_summary.txt
"""

import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

import requests
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR   = os.path.join(BASE_DIR, 'output')
CHROMA_DIR   = os.path.join(BASE_DIR, 'knowledge_base', 'chroma_db')
GOLDEN_PATH  = os.path.join(DATA_DIR, 'ground_truth_ragas.json')
RESULTS_CSV  = os.path.join(OUTPUT_DIR, 'ragas_results.csv')
SUMMARY_TXT  = os.path.join(OUTPUT_DIR, 'ragas_summary.txt')
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_URL              = 'http://127.0.0.1:5001/chat'
OPENROUTER_API_KEY   = os.getenv('OPENROUTER_API_KEY', '')
OPENROUTER_BASE_URL  = 'https://openrouter.ai/api/v1'
OPENROUTER_MODEL     = 'openai/gpt-4o-mini'
TOP_K                = 10
OOS_MARKER           = 'tidak menemukan informasi tersebut'

print('='*60)
print('RAGAS EVALUATION v3 — Chatbot Perpustakaan PNJ')
print('='*60)

# ── Setup ─────────────────────────────────────────────────────────────────────
print('\n[1/4] Loading embedding model...')
embed_model = SentenceTransformer('BAAI/bge-m3')
print('      OK')

print('[2/4] Connecting ChromaDB...')
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_col    = chroma_client.get_collection('kb_perpustakaan_pnj')
print(f'      {chroma_col.count()} chunks')

print('[3/4] Setup RAGAS...')
try:
    from datasets import Dataset
    from ragas import evaluate, RunConfig
    from ragas.metrics import Faithfulness, AnswerRelevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings

    if not OPENROUTER_API_KEY:
        raise ValueError('OPENROUTER_API_KEY tidak ditemukan di .env')

    # LLM judge: GPT-4o-mini via OpenRouter
    from langchain_openai import ChatOpenAI
    ragas_llm = LangchainLLMWrapper(ChatOpenAI(
        model=OPENROUTER_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0,
        max_retries=3,
    ))
    ragas_embs = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name='BAAI/bge-m3',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    ))
    RAGAS_RUN_CONFIG = RunConfig(timeout=120, max_workers=2, max_retries=3)
    RAGAS_AVAILABLE  = True
    print(f'      RAGAS siap — LLM judge: {OPENROUTER_MODEL} via OpenRouter')
except ImportError as e:
    print(f'      [WARN] RAGAS tidak tersedia: {e}')
    print(f'      Install: pip install "ragas>=0.2" langchain-openai langchain-huggingface datasets')
    RAGAS_AVAILABLE = False
except ValueError as e:
    print(f'      [ERROR] {e}')
    RAGAS_AVAILABLE = False

print('[4/4] Memuat golden dataset...')
with open(GOLDEN_PATH, encoding='utf-8') as f:
    golden = json.load(f)

in_scope  = [q for q in golden if q.get('query_type') != 'out_of_scope']
out_scope = [q for q in golden if q.get('query_type') == 'out_of_scope']
print(f'      {len(golden)} total | {len(in_scope)} in-scope | {len(out_scope)} OOS')

# ── Helper ────────────────────────────────────────────────────────────────────

def get_contexts(question: str) -> list:
    q_emb = embed_model.encode([question], normalize_embeddings=True).tolist()
    res   = chroma_col.query(query_embeddings=q_emb, n_results=TOP_K,
                              include=['documents'])
    return res['documents'][0] if res and res['documents'] and res['documents'][0] else []


def call_chatbot(question: str) -> dict:
    t0 = time.time()
    try:
        r = requests.post(API_URL, json={'message': question}, timeout=200)
        elapsed = round(time.time() - t0, 2)
        if r.status_code == 200:
            data = r.json()
            return {'answer': data.get('answer', ''),
                    'query_type': data.get('query_type', ''),
                    'elapsed_s': elapsed}
    except Exception as e:
        print(f'  [ERROR] API: {e}')
    return {'answer': '', 'query_type': '', 'elapsed_s': round(time.time() - t0, 2)}


# ── Collect answers ───────────────────────────────────────────────────────────
print('\n[5/x] Mengumpulkan jawaban dari chatbot...')
rows = []
for i, item in enumerate(in_scope, 1):
    q = item['question']
    print(f'  [{i:02d}/{len(in_scope)}] {q[:65]}')
    result   = call_chatbot(q)
    contexts = get_contexts(q)
    rows.append({
        'question'    : q,
        'answer'      : result['answer'],
        'contexts'    : contexts,
        'ground_truth': item['ground_truth'],
        'elapsed_s'   : result['elapsed_s'],
    })
    print(f'       ✓ {result["elapsed_s"]}s')

# ── Hard Stop ─────────────────────────────────────────────────────────────────
print('\n[OOS] Mengecek out-of-scope queries...')
oos_correct = 0
for item in out_scope:
    result  = call_chatbot(item['question'])
    correct = OOS_MARKER in result['answer'].lower()
    oos_correct += int(correct)
    mark = '✅' if correct else '❌'
    print(f'  {mark} [{result["elapsed_s"]}s] "{item["question"]}"')
oos_acc    = oos_correct / len(out_scope) if out_scope else 1.0
avg_elapsed = np.mean([r['elapsed_s'] for r in rows])
print(f'\n  Hard Stop Accuracy: {oos_correct}/{len(out_scope)} = {oos_acc:.1%}')

# ── RAGAS ─────────────────────────────────────────────────────────────────────
if RAGAS_AVAILABLE:
    print('\n[RAGAS] Memulai evaluasi (bisa 10-30 menit)...')
    ds = Dataset.from_dict({
        'question'    : [r['question']     for r in rows],
        'answer'      : [r['answer']       for r in rows],
        'contexts'    : [r['contexts']     for r in rows],
        'ground_truth': [r['ground_truth'] for r in rows],
    })
    try:
        result_ragas = evaluate(
            dataset=ds,
            metrics=[Faithfulness(llm=ragas_llm),
                     AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embs)],
            run_config=RAGAS_RUN_CONFIG,
        )
        df_result = result_ragas.to_pandas()
        df_result['ground_truth'] = [r['ground_truth'] for r in rows]
        df_result['elapsed_s']    = [r['elapsed_s']    for r in rows]
        df_result.to_csv(RESULTS_CSV, index=False)

        # result_ragas[] bisa return list atau float tergantung versi RAGAS
        # Gunakan df_result untuk aggregate agar konsisten
        faith_raw   = result_ragas['faithfulness']
        ans_rel_raw = result_ragas['answer_relevancy']

        def _to_float(v) -> str:
            """Konversi ke float; kembalikan string 'N/A (timeout)' kalau gagal."""
            try:
                if isinstance(v, (list, pd.Series)):
                    vals = pd.to_numeric(pd.Series(v), errors='coerce').dropna()
                    return f'{vals.mean():.4f} ({len(vals)}/{len(v)} valid)' if len(vals) else 'N/A (all timeout)'
                f = float(v)
                return f'{f:.4f}' if not (f != f) else 'N/A (NaN)'  # NaN check
            except Exception:
                return 'N/A'

        faith_str   = _to_float(faith_raw)
        ans_rel_str = _to_float(ans_rel_raw)
        SEP60       = '=' * 60

        summary = (
            f"\n{SEP60}\n"
            f"HASIL EVALUASI RAGAS v3 — Chatbot Perpustakaan PNJ\n"
            f"{SEP60}\n"
            f"Model LLM       : {OPENROUTER_MODEL} (via OpenRouter)\n"
            f"Embedding       : BAAI/bge-m3\n"
            f"Jumlah Test     : {len(in_scope)} in-scope + {len(out_scope)} OOS\n\n"
            f"METRIK RAG:\n"
            f"  Faithfulness       : {faith_str}\n"
            f"  Answer Relevancy   : {ans_rel_str}\n\n"
            f"HARD STOP:\n"
            f"  Out-of-Scope Accuracy : {oos_acc:.4f}  ({oos_correct}/{len(out_scope)})\n\n"
            f"PERFORMA:\n"
            f"  Rata-rata Response Time : {avg_elapsed:.2f}s\n\n"
            f"Hasil lengkap: {RESULTS_CSV}\n"
            f"{SEP60}"
        )
        print(summary)
        with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
            f.write(summary.strip())
        print(f'Ringkasan disimpan: {SUMMARY_TXT}')

    except Exception as e:
        print(f'\n[ERROR] RAGAS gagal: {e}')
        import traceback; traceback.print_exc()
else:
    print(f'\n{"="*60}')
    print('HASIL (tanpa RAGAS):')
    print(f'  Hard Stop Accuracy : {oos_acc:.4f}  ({oos_correct}/{len(out_scope)})')
    print(f'  Avg Response Time  : {avg_elapsed:.2f}s')
    print('Pasang RAGAS: pip install "ragas>=0.2" langchain-openai langchain-huggingface datasets')
    print('='*60)
