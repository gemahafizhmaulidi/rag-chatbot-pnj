"""
judge_pairwise.py — Fase 4b: blind pairwise judge Arm A vs Arm B.

Untuk tiap query in-scope: gabungkan jawaban kedua arm, ACAK urutannya (blind),
minta gpt-4o-mini memilih mana lebih baik (atau seri). Agregasi → menang/seri/kalah,
lalu cek kriteria K4 pra-registrasi (PREREGISTRATION.md).

Judge = openai/gpt-4o-mini via OpenRouter (≠ teacher Qwen-35B → bias rendah).
Mitigasi bias: urutan diacak per item + identitas arm disembunyikan dari judge.

Pakai:
    python experiments/raft/judge_pairwise.py
Prasyarat: out/eval_answers.json sudah ada (dari vast_gen_answers.py, download dari Vast).
"""
from __future__ import annotations
import os, sys, json, time, random, re, requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT  = os.path.join(HERE, 'out')
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, '..', '.env'))

JUDGE_MODEL = 'openai/gpt-4o-mini'
OR_KEY      = os.getenv('OPENROUTER_API_KEY', '')
OR_URL      = 'https://openrouter.ai/api/v1/chat/completions'
WIN_MARGIN  = 12     # K4: selisih menang minimal (≥15% dari 75 item = 11.25 → 12)
random.seed(2026)

JUDGE_SYS = (
    "Anda juri ahli yang menilai kualitas jawaban asisten perpustakaan kampus. "
    "Diberi satu pertanyaan, jawaban acuan (ground truth), dan dua kandidat jawaban "
    "(Jawaban 1 & Jawaban 2). Nilai berdasarkan: kesesuaian fakta dengan acuan, "
    "kelengkapan, kejujuran (tidak mengarang), dan kegunaan bagi mahasiswa. "
    "Pilih yang LEBIH BAIK. Jika setara, jawab SERI. "
    "Balas HANYA JSON: {\"winner\": \"1\"|\"2\"|\"seri\", \"alasan\": \"<singkat>\"}"
)


def judge(question, gt, ans1, ans2):
    user = (f"PERTANYAAN:\n{question}\n\nJAWABAN ACUAN:\n{gt}\n\n"
            f"JAWABAN 1:\n{ans1}\n\nJAWABAN 2:\n{ans2}")
    r = requests.post(OR_URL,
        headers={'Authorization': f'Bearer {OR_KEY}', 'Content-Type': 'application/json'},
        json={'model': JUDGE_MODEL, 'temperature': 0,
              'messages': [{'role': 'system', 'content': JUDGE_SYS},
                           {'role': 'user', 'content': user}],
              'reasoning': {'enabled': False}}, timeout=120)
    txt = r.json()['choices'][0]['message']['content']
    m = re.search(r'\{.*\}', txt, re.DOTALL)
    return json.loads(m.group(0)) if m else {'winner': 'seri', 'alasan': 'parse-fail'}


def main():
    # Baca format unified dari vast_gen_answers.py
    data = json.load(open(os.path.join(OUT, 'eval_answers.json')))
    a_by_q = {x['question']: x for x in data}
    b_by_q = a_by_q   # same file, answer_a vs answer_b
    common = list(a_by_q.keys())
    arm_a_model = data[0].get('model_a', 'base')
    arm_b_model = data[0].get('model_b', 'raft')
    print(f'Arm A={arm_a_model} | Arm B={arm_b_model} | query terbanding={len(common)}')

    win_a = win_b = tie = 0
    details = []
    for i, q in enumerate(common, 1):
        item = a_by_q[q]
        ans_a = item['answer_a']   # base
        ans_b = item['answer_b']   # RAFT
        gt    = item['ground_truth']
        swap = random.random() < 0.5            # blind: acak posisi
        ans1, ans2 = (ans_b, ans_a) if swap else (ans_a, ans_b)
        verdict = judge(q, gt, ans1, ans2)
        w = verdict.get('winner', 'seri')
        if w == 'seri':
            tie += 1; pick = 'seri'
        else:
            pos1_is_a = not swap
            a_wins = (w == '1') == pos1_is_a
            if a_wins: win_a += 1; pick = 'A'
            else:      win_b += 1; pick = 'B'
        details.append({'question': q, 'winner': pick, 'swap': swap,
                        'alasan': verdict.get('alasan', '')})
        if i % 10 == 0:
            print(f'  [{i}/{len(common)}] A={win_a} B={win_b} seri={tie}')
        time.sleep(0.3)

    diff = win_b - win_a
    k4 = (win_b > win_a) and (diff >= WIN_MARGIN)
    report = {
        'arm_A_model': arm_a_model, 'arm_B_model': arm_b_model,
        'n': len(common), 'win_A': win_a, 'win_B': win_b, 'tie': tie,
        'B_minus_A': diff, 'K4_win_margin_required': WIN_MARGIN,
        'K4_pairwise_pass': k4, 'details': details,
    }
    out = os.path.join(OUT, 'pairwise_report.json')
    json.dump(report, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n' + '=' * 55)
    print(f'  HASIL BLIND PAIRWISE (judge {JUDGE_MODEL})')
    print(f'  Arm A menang : {win_a}')
    print(f'  Arm B menang : {win_b}')
    print(f'  Seri         : {tie}')
    print(f'  B − A        : {diff}  (butuh ≥ {WIN_MARGIN} utk lolos K4)')
    print(f'  K4 pairwise  : {"LOLOS ✅" if k4 else "TIDAK lolos ❌"}')
    print('=' * 55)
    print(f'  → {out}')
    print('  Gabungkan dengan K1–K3 (RAGAS+blackbox) & K5 utk keputusan final (PREREGISTRATION.md).')


if __name__ == '__main__':
    main()
