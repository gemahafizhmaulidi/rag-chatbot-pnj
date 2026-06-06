# Hasil RAGAS (Concern D) — sistem baru (LLM-router, bobot 0.45)

Keresahan: (1) angka RAGAS stale (sistem pra-refactor) & inkonsisten (docs lama punya 0.8344 DAN 0.8781);
(2) judge eksternal gpt-4o-mini kontradiksi "fully local"; (3) faithfulness < target 0.85.

## Masalah teknis yang ditemukan
- **RAGAS lib 0.4.3 + judge OpenRouter → faithfulness RUSAK** (memberi 0.0 untuk jawaban yang jelas
  benar) karena ketergantungan structured-output/function-calling yang tak didukung penuh OpenRouter.
  Smoke test membuktikan ini. → Metrik diimplementasi ulang TERKONTROL sesuai definisi RAGAS
  (`scripts/ragas_eval_v2.py`), tervalidasi (faithful→1.0, ngarang→0.0).
- **Konteks pengukuran**: run awal menilai faithfulness vs chroma-top10-untuk-PERTANYAAN → UNDER-measure
  (jawaban benar dihukum bila fakta ada di KB tapi tak masuk top-10 versi pertanyaan; mis. "jam
  Senin–Kamis" jawaban benar tapi F=0.25). Diperbaiki: nilai vs **konteks AKTUAL sistem**
  (router → info_query → search_kb), sesuai definisi RAGAS yang benar (`ragas_faithfulness_accurate.py`).

## Hasil final (40 in-scope general_info + 10 OOS; judge gpt-4o-mini temp=0)
| Metrik | Nilai | Target |
|---|---|---|
| **Faithfulness** (vs konteks sistem) | **0.9286** | ≥0.85 ✅ |
| Faithfulness (vs konteks pertanyaan, under-measure) | 0.8097 | — |
| **Answer Relevancy** | **0.8701** | ≥0.80 ✅ |
| **Hard-stop OOS** | **10/10 = 100%** | 1.0 ✅ |

## Justifikasi judge eksternal (sub-isu 2)
Judge gpt-4o-mini adalah **evaluator**, BUKAN komponen produksi — ini praktik standar evaluasi RAG
(metrik LLM-as-judge butuh model kuat & netral). **Sistem produksi tetap 100% lokal** (Qwen via
Ollama + bge-m3 + ChromaDB). Tidak ada kontradiksi: lokal untuk melayani user, eksternal hanya untuk
mengukur. (Catatan: faithfulness juga bisa dijalankan dengan judge lokal Qwen bila ingin 100% lokal
termasuk evaluasi, dengan trade-off ketelitian judge.)

## Kesimpulan
Ketiga metrik RAGAS pada sistem yang BERJALAN SEKARANG memenuhi/melampaui target, diukur dengan
metode yang benar & jujur. Angka stale (0.8344/0.8781) DITARIK.

## Berkas
`ragas_v2_results.csv`, `ragas_v2_faithfulness_accurate.csv`, `ragas_v2_summary.txt`,
`ragas_v2_run.log`, `ragas_faithfulness_accurate.log`.
