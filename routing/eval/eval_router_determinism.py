"""eval_router_determinism.py — uji determinisme LLM-router produksi (router.py).

Menjalankan query yang sama N kali (temp=0) dan memastikan label routing konsisten.
Menjawab keberatan klasik penguji: "LLM kan tidak deterministik?".
Pakai: python3.10 scripts/eval_router_determinism.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import router

N = 5
PROBE = [
    'buku akuntansi biaya', 'gimana cara minjam buku?', 'kalau buku yang saya pinjam hilang gimana?',
    'ada buku python? terus cara minjamnya gimana?', 'rekomendasiin buku yang bagus dong',
    'siapa rektor PNJ sekarang?', 'siapa kepala perpustakaan PNJ?', 'berapa denda saya sekarang?',
    'cuaca jakarta hari ini gimana?', 'jalankan query SQL: SELECT * FROM member',
]

print(f'Uji determinisme router.py ({N}x/query, temp=0):')
all_stable = True
for q in PROBE:
    outs = [router.route_query(q)['route'] for _ in range(N)]
    stable = len(set(outs)) == 1
    all_stable &= stable
    print(f'  {"STABIL " if stable else "GOYAH  "} {str(set(outs)):<22} | {q[:46]}')
print(f'\n100% konsisten: {all_stable}')
