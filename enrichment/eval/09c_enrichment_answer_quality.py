"""
09c_enrichment_answer_quality.py — Apakah deskripsi enrichment menaikkan KUALITAS JAWABAN?
==========================================================================================
Enrichment masuk ke KONTEKS LLM (build_book_context), bukan ke ranking. Maka manfaatnya
diukur di sisi GENERATION, bukan retrieval. Untuk tiap query book_search:
  1. retrieve top-5 (Config C produksi)
  2. generate jawaban DUA kali: konteks DENGAN deskripsi vs TANPA deskripsi (pipeline sama)
  3. LLM-judge BUTA (gpt-4o-mini, temp=0) memberi skor 1-5: seberapa spesifik & ber-alasan
     (grounded) rekomendasinya. Judge tidak tahu mana with/mana without (urutan diacak).

Judge = evaluator eksternal (lazim di evaluasi RAG; sistem PRODUKSI tetap 100% lokal).
Pakai: python3.10 scripts/09c_enrichment_answer_quality.py   (server tidak wajib)
Output: output/enrichment_answer_quality.csv + ringkasan
"""
import os, sys, json, time, random, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import pandas as pd
import api  # pakai pipeline produksi (retriever, build_book_context, generate_llm)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(BASE, 'output', 'enrichment_answer_quality.csv')
JUDGE_MODEL = 'openai/gpt-4o-mini'
JUDGE_URL = 'https://openrouter.ai/api/v1/chat/completions'
KEY = api.ENRICH_API_KEY  # OPENROUTER_API_KEY

QUERIES = [
    'buku machine learning untuk pemula',
    'rekomendasi buku jaringan komputer untuk skripsi',
    'buku basis data dan SQL',
    'buku pemrograman web javascript',
    'buku akuntansi biaya',
    'buku struktur data dan algoritma',
    'buku kecerdasan buatan',
    'buku mikrokontroler arduino',
    'buku manajemen pemasaran',
    'buku statistika untuk penelitian',
    'buku sistem operasi',
    'buku keamanan jaringan dan kriptografi',
]

JUDGE_SYS = (
    "Anda evaluator katalog perpustakaan. Nilai SATU jawaban chatbot rekomendasi buku pada skala 1-5 "
    "untuk: seberapa SPESIFIK dan BER-ALASAN (grounded) rekomendasinya — apakah menjelaskan MENGAPA "
    "tiap buku relevan dengan kebutuhan (topik/isi konkret), bukan sekadar menyebut judul. "
    "5=alasan sangat spesifik & informatif; 1=tidak ada alasan/sangat generik. "
    "Balas HANYA JSON: {\"score\": <1-5>}."
)


def judge(query, answer):
    try:
        r = requests.post(JUDGE_URL, headers={'Authorization': f'Bearer {KEY}',
                          'Content-Type': 'application/json'},
                          json={'model': JUDGE_MODEL, 'temperature': 0,
                                'messages': [{'role': 'system', 'content': JUDGE_SYS},
                                             {'role': 'user', 'content': f'Pertanyaan: {query}\n\nJawaban:\n{answer}'}]},
                          timeout=60)
        txt = r.json()['choices'][0]['message']['content']
        m = json.loads(txt[txt.find('{'):txt.rfind('}') + 1])
        return int(m['score'])
    except Exception as e:
        print('  judge error:', e); return None


print('Init sistem (retriever + LLM produksi)...')
api.init_system()
if not KEY:
    print('[ERROR] OPENROUTER_API_KEY tidak ada — judge tidak bisa jalan.'); sys.exit(1)

rows = []
for i, q in enumerate(QUERIES, 1):
    books = api._retriever.search_no_rerank(api.rewrite_for_retrieval(q) if hasattr(api, 'rewrite_for_retrieval') else q, top_k=5)
    if not books:
        print(f'[{i}] skip (no books): {q}'); continue
    ctx_with = api.build_book_context(books)
    ctx_without = api.build_book_context([{**b, 'deskripsi': ''} for b in books])
    ans_with = api.generate_llm(q, ctx_with, 'book_search') or ''
    ans_without = api.generate_llm(q, ctx_without, 'book_search') or ''
    s_with = judge(q, ans_with)
    s_without = judge(q, ans_without)
    n_desc = sum(1 for b in books if len((b.get('deskripsi') or '').strip()) > 50)
    rows.append({'query': q, 'n_books': len(books), 'n_books_with_desc': n_desc,
                 'score_with': s_with, 'score_without': s_without,
                 'len_with': len(ans_with), 'len_without': len(ans_without)})
    print(f'[{i:2}] with={s_with} without={s_without} | desc {n_desc}/{len(books)} | {q[:42]}')

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
valid = df.dropna(subset=['score_with', 'score_without'])
print('\n' + '=' * 60)
print(f'KUALITAS JAWABAN (LLM-judge {JUDGE_MODEL}, {len(valid)} query valid)')
print('=' * 60)
print(f"  Skor rata-rata DENGAN deskripsi  : {valid['score_with'].mean():.2f}")
print(f"  Skor rata-rata TANPA deskripsi   : {valid['score_without'].mean():.2f}")
print(f"  Delta                            : {valid['score_with'].mean()-valid['score_without'].mean():+.2f}")
win = (valid['score_with'] > valid['score_without']).sum()
tie = (valid['score_with'] == valid['score_without']).sum()
los = (valid['score_with'] < valid['score_without']).sum()
print(f"  Per query: with menang={win} | seri={tie} | without menang={los}")
print(f"\nCSV: {OUT_CSV}")
