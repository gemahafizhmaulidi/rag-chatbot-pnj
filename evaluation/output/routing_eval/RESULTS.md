# Hasil Evaluasi & Komparasi Routing — Chatbot Perpustakaan PNJ

Konsolidasi bukti untuk penulisan skripsi (BAB IV). Semua angka reproducible.
Eksperimen dijalankan 2026-06-02. Test set: `routing_testset.json` (82 query, 7 kategori).

> Catatan metodologi: test set dianotasi oleh peneliti (Gema) dan label kebijakan
> divalidasi manual (mis. "kepala perpustakaan" = general_info, bukan OOS).
> Untuk rigor penuh disarankan menambah 1 anotator + hitung Cohen's Kappa.

---

## 1. Komparasi 3 Strategi Routing (kontribusi utama)

Akurasi routing pada test set 82 query — sumber: `router_comparison.csv`, log `3way_comparison_run.log`.

| Kategori (n) | Keyword (rule) | XLM-RoBERTa fine-tuned | **LLM-router** |
|---|---|---|---|
| **KESELURUHAN (82)** | **75,6%** | **87,8%** | **97,6%** |
| book_search (13) | 77% | 77% | 100% |
| recommendation (12) | 75% | 83% | 92% |
| general_info (15) | 80% | 73% | 100% |
| hybrid (10) | 100% | 100% | 100% |
| stats (10) | 100% | 100% | 100% |
| greeting (8) | 100% | 88% | 100% |
| oos (14) | 21% | 100% | 93% |

**LLM-router versi produksi** (`router.py`, prompt + ekstraksi sub-query): **98,8% (81/82)** —
sumber: `router_prod_eval.log`. Latensi ~1,0 dtk/query. Fully local (Qwen via Ollama).

### Temuan untuk argumen skripsi
1. **Keyword routing mentok di 75,6%** karena *substring matching buta*: `"akun"` ⊂ `"ak·untansi"`
   dan `"komputer"` ∈ keyword fasilitas → pencarian buku salah jadi `hybrid`; regex `saya+pinjam`
   → "buku saya hilang" (prosedur) salah jadi OOS. Menambah keyword = memindah bug, bukan menutup.
2. **XLM-R fine-tuned (87,8%)** memang membantu — terutama OOS (21%→100%) karena punya label `oos` —
   tapi turun di general_info (80→73%) & greeting (100→88%), butuh data latih, dan salah = harus retrain.
3. **Bug integrasi XLM-R (temuan jujur):** di kode lama, `classify_query()` memanggil pipeline
   `top_k=1` yang mengembalikan list bersarang lalu di-index `['label']` → **selalu error & diam-diam
   fallback ke keyword**. Artinya produksi lama **efektif keyword murni (75,6%)** — fine-tuning tidak
   pernah aktif. (Diperbaiki di snapshot baseline hanya untuk mengukur 87,8% yang jujur.)
4. **LLM-router (98,8%)** unggul telak, paham konteks (bukan kata), tanpa keyword/maintenance,
   tanpa data latih, deterministik, dan lokal → dipilih untuk produksi.

---

## 2. Determinisme LLM-router

Sumber: `router_determinism.log`. 10 query diulang 5× pada temp=0 → **100% konsisten**
(menjawab keberatan "LLM tidak deterministik").

---

## 3. Bukti Kualitatif (before/after refactor)

Transkrip jawaban penuh: `qualitative_BEFORE.md` vs `qualitative_AFTER.md`.
3 bug routing keyword yang terbukti hilang setelah pindah ke LLM-router:

| Query | BEFORE (keyword) | AFTER (LLM-router) |
|---|---|---|
| "ada buku python **buat** pemula?" | "Mohon maaf, tidak bisa mengerjakan tugas…" lalu cari buku (kontradiktif; kata "buat" ke-trigger task-verb) | langsung daftar buku, tanpa disclaimer |
| "rekomendasiin buku yang **bagus** dong" | daftar buku acak karya pengarang bernama "Bagus" | minta klarifikasi topik |
| "kalau buku yang **saya pinjam** hilang gimana?" | OOS instan (salah dikira akun personal) | jawab prosedur penggantian buku hilang |

Tetap benar (tidak regresi): pencarian by-author (Pressman), kejujuran "tidak ada" (quantum computing),
fakta (jam Jumat 16.30, denda Rp1.000, telepon dari KB), hybrid (buku + prosedur), penolakan OOS & injection.

---

## 4. Hard-stop KB (re-purpose jadi backstop)

Sumber: `hardstop_audit.log` (script `scripts/audit_hardstop.py`).

Diagnosis (pada threshold lama 0,4108): distance query in-scope dan OOS-personal **tumpang tindih**
(mis. "berapa lama masa peminjaman?" = 0,4015 in-scope vs "buku apa saja yang sudah saya pinjam?" = 0,4008 OOS)
→ **threshold tunggal mustahil memisahkan keduanya**; F1=1,0 lama overfit ke 10 query kalibrasi
(1 false-OOS "buka hari sabtu?", 3 miss-OOS akun personal).

Resolusi: OOS utama dipindah ke LLM-router (yang menangkap OOS-personal di hulu); hard-stop
dilonggarkan **0,4108 → 0,45** sebagai *backstop permisif* agar tidak menolak query sah di tepi ambang.
Defense-in-depth: (1) security pre-filter → (2) LLM-router → (3) hard-stop KB → (4) faithfulness LLM.

Pada threshold baru **0,45** (`hardstop_audit.log`): **FALSE-OOS = 0/15** (query sah seperti
"buka hari sabtu?" kini lolos — masalah lama teratasi); **MISS-OOS = 5/14**, namun kelimanya
(akun personal & "daftar ulang semester") **sudah ditangkap LLM-router di lapis hulu**, jadi tidak
sampai ke hard-stop saat runtime. Ini mengonfirmasi peran hard-stop sebagai backstop, bukan gerbang OOS utama.

---

## Berkas terkait
| Berkas | Isi |
|---|---|
| `routing_testset.json` | 82 query berlabel (DRAFT, perlu anotator ke-2) |
| `router_comparison.csv` | hasil per-query 3 router |
| `3way_comparison_run.log` | log run komparasi |
| `router_prod_eval.log` | akurasi router.py produksi (98,8%) |
| `router_determinism.log` | uji determinisme |
| `hardstop_audit.log` | audit hard-stop KB |
| `qualitative_BEFORE.md` / `qualitative_AFTER.md` | transkrip jawaban penuh |
| `../../../baseline_routing_backup/` | kode baseline (keyword+XLM-R) + model + README |
