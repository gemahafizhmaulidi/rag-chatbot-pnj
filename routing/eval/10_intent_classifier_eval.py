"""
10_intent_classifier_eval.py — Evaluasi formal Intent Classifier (routing) sebagai
tugas klasifikasi multi-kelas, menjawab permintaan dosen pembimbing:
"uji intent classifier (Qwen) dengan data — sajikan perhitungan nyata".

Membaca prediksi yang SUDAH ADA (`output/routing_eval/router_comparison.csv`,
82 query berlabel, 7 kelas) — TIDAK menjalankan ulang routing. Menghitung:
  - Confusion matrix per strategi (7×7)
  - Precision / Recall / F1 per kelas
  - Accuracy, macro-F1, weighted-F1
  - Tabel komparasi 3 strategi (Keyword vs XLM-R supervised vs Qwen LLM zero-shot)

Catatan taksonomi (untuk BAB III/IV):
  - Keyword  = rule-based (tanpa pembelajaran)
  - XLM-R    = supervised (fine-tuned pada label)
  - LLM-router (Qwen) = ZERO-SHOT classification (pre-trained + instruksi prompt,
    TIDAK dilatih pada label routing). 82 query berlabel dipakai sebagai TEST SET
    untuk evaluasi, bukan untuk melatih.

Output:
  output/routing_eval/INTENT_CLASSIFIER_EVAL.md
  output/routing_eval/confusion_matrix_intent.png  (3 panel: Keyword, XLM-R, LLM)
"""
from __future__ import annotations
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             accuracy_score)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(BASE, 'output', 'routing_eval')
CSV  = os.path.join(EVAL, 'router_comparison.csv')

# kolom CSV → nama strategi (dipetakan via akurasi: pure_rule=75.6, current=87.8, llm=97.6)
STRATEGIES = [
    ('pure_rule', 'Keyword (rule-based)'),
    ('current',   'XLM-RoBERTa (supervised)'),
    ('llm',       'Qwen LLM-router (zero-shot)'),
]
LABELS = ['book_search', 'recommendation', 'general_info', 'hybrid',
          'stats', 'greeting', 'oos']
SHORT  = {'book_search': 'book', 'recommendation': 'recom', 'general_info': 'info',
          'hybrid': 'hybrid', 'stats': 'stats', 'greeting': 'greet', 'oos': 'oos'}


def load():
    rows = list(csv.DictReader(open(CSV, encoding='utf-8-sig')))
    y_true = [r['expected'].strip() for r in rows]
    preds  = {col: [r[col].strip() for r in rows] for col, _ in STRATEGIES}
    return y_true, preds, len(rows)


def per_class_table(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    lines = ['| Kelas (n) | Precision | Recall | F1 |', '|---|---|---|---|']
    for i, lab in enumerate(LABELS):
        lines.append(f'| {lab} ({int(s[i])}) | {p[i]:.2f} | {r[i]:.2f} | {f[i]:.2f} |')
    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average='macro', zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average='weighted', zero_division=0)
    lines.append(f'| **Macro avg** | {mp:.2f} | {mr:.2f} | **{mf:.2f}** |')
    lines.append(f'| **Weighted avg** | {wp:.2f} | {wr:.2f} | **{wf:.2f}** |')
    return '\n'.join(lines), acc, mf, wf


def main():
    y_true, preds, n = load()
    md = ['# Evaluasi Intent Classifier (Routing) — Klasifikasi Multi-Kelas',
          '',
          f'Test set: `router_comparison.csv` — **{n} query berlabel**, **{len(LABELS)} kelas intent**. '
          'Prediksi sudah tersimpan (tidak di-run ulang). Sumber routing: `router.py`.',
          '',
          '> **Taksonomi metode.** Keyword = *rule-based*; XLM-RoBERTa = *supervised* '
          '(fine-tuned); **Qwen LLM-router = *zero-shot*** (pre-trained + instruksi prompt, '
          'tidak dilatih pada label routing). 82 query berlabel = **test set** untuk evaluasi, '
          'bukan data latih.',
          '']

    # ── tabel komparasi ringkas ──
    summary = [('Strategi', 'Accuracy', 'Macro-F1', 'Weighted-F1', 'Jenis')]
    types = ['rule-based', 'supervised', 'zero-shot']
    per_strategy_md = []
    for (col, name), typ in zip(STRATEGIES, types):
        tbl, acc, mf, wf = per_class_table(y_true, preds[col])
        summary.append((name, f'{acc*100:.1f}%', f'{mf:.3f}', f'{wf:.3f}', typ))
        per_strategy_md.append((name, tbl))

    md += ['## 1. Ringkasan Komparasi', '',
           '| ' + ' | '.join(summary[0]) + ' |',
           '|' + '---|' * len(summary[0])]
    for row in summary[1:]:
        bold = '**' if 'LLM' in row[0] else ''
        md.append('| ' + ' | '.join(f'{bold}{c}{bold}' for c in row) + ' |')
    md += ['',
           '> **Macro-F1** memberi bobot sama tiap kelas (penting karena kelas tidak seimbang, '
           'mis. oos=14 vs greeting=8) — lebih jujur dari accuracy untuk classifier multi-kelas.',
           '']

    # ── per-kelas tiap strategi ──
    md += ['## 2. Precision / Recall / F1 per Kelas', '']
    for name, tbl in per_strategy_md:
        md += [f'### {name}', '', tbl, '']

    # ── confusion matrix (3 panel) ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (col, name) in zip(axes, STRATEGIES):
        cm = confusion_matrix(y_true, preds[col], labels=LABELS)
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(name, fontsize=11)
        ax.set_xticks(range(len(LABELS))); ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels([SHORT[l] for l in LABELS], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([SHORT[l] for l in LABELS], fontsize=8)
        ax.set_xlabel('Prediksi'); ax.set_ylabel('Label sebenarnya')
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8,
                            color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.suptitle('Confusion Matrix Intent Classifier — 82 query, 7 kelas', fontsize=13)
    fig.tight_layout()
    png = os.path.join(EVAL, 'confusion_matrix_intent.png')
    fig.savefig(png, dpi=150, bbox_inches='tight')
    md += ['## 3. Confusion Matrix', '',
           f'![Confusion Matrix](confusion_matrix_intent.png)', '',
           '*Gambar: confusion matrix tiga strategi. Diagonal = prediksi benar.*', '']

    # ── kesalahan LLM-router (transparansi) ──
    rows = list(csv.DictReader(open(CSV, encoding='utf-8-sig')))
    errs = [r for r in rows if r['expected'].strip() != r['llm'].strip()]
    md += ['## 4. Analisis Kesalahan LLM-router (transparansi)', '',
           f'LLM-router salah pada **{len(errs)} dari {n}** query:', '']
    if errs:
        md += ['| Query | Seharusnya | Prediksi |', '|---|---|---|']
        for e in errs:
            md.append(f"| {e['query']} | {e['expected'].strip()} | {e['llm'].strip()} |")
    else:
        md.append('_Tidak ada kesalahan._')
    md += ['']

    out = os.path.join(EVAL, 'INTENT_CLASSIFIER_EVAL.md')
    open(out, 'w', encoding='utf-8').write('\n'.join(md))

    # ── ringkasan ke terminal ──
    print('=' * 60)
    print('  EVALUASI INTENT CLASSIFIER (82 query, 7 kelas)')
    print('=' * 60)
    for row in summary[1:]:
        print(f'  {row[0]:32s} acc={row[1]:>6s}  macroF1={row[2]}  ({row[4]})')
    print('=' * 60)
    print(f'  → {out}')
    print(f'  → {png}')


if __name__ == '__main__':
    main()
