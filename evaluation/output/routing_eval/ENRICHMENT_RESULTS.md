# Hasil Enrichment (Concern C) — METODOLOGI DIPERBAIKI

Keresahan: klaim "+25% Hit@5 / +12.9% MRR" dari `09_before_after_enrichment.py`.

## Cacat metode lama (klaim DITARIK)
1. **GT sirkular**: buku "relevan" = deskripsi opacv2-nya mengandung keyword, lalu diuji apakah
   dense (yang meng-embed deskripsi itu) menemukannya → hampir tautologi.
2. **Dense-only**, bukan Config C hybrid produksi.
→ "+25%" tidak mencerminkan sistem nyata. **Ditarik.**

## 1. Coverage (kontribusi data yang nyata)
Deskripsi tersedia: **5.8% (529/9084) → 93.4% (8481/9084)** buku teks. Ini kontribusi utama
enrichment: katalog OPAC PNJ yang nyaris tanpa deskripsi kini punya representasi semantik.

## 2. Dampak RETRIEVAL — ~NETRAL (jujur)  (`09b_enrichment_eval_sound.py`)
Config C Hybrid (0.45/0.55), GT independen `ground_truth.json` (40 query), before vs after:

| Metrik | Before (tanpa desc) | After (enrichment) | Δ |
|---|---|---|---|
| MRR@5 | 0.9258 | 0.9092 | −1.8% |
| Hit@5 | 0.9750 | 0.9750 | 0% |
| NDCG@5 | 0.7683 | 0.7740 | +0.7% |

Per query NDCG: naik 10 / turun 5 / sama 25. (39/40 query menyentuh ≥1 buku yang diperkaya.)
**Sebab:** query topik (mis. "pemrograman Python") sudah ditemukan oleh BM25 judul+topik;
deskripsi redundan untuk ranking. Sistem capai NDCG 0.768 **bahkan tanpa deskripsi**.

## 3. Dampak KUALITAS JAWABAN — SIGNIFIKAN  (`09c_enrichment_answer_quality.py`)
Deskripsi masuk ke KONTEKS LLM (`build_book_context`), bukan ke ranking → manfaatnya di GENERATION.
Jawaban di-generate DENGAN vs TANPA deskripsi (pipeline sama), dinilai LLM-judge BUTA
(gpt-4o-mini, temp=0, skala 1-5 "spesifik & ber-alasan/grounded"):

| | Skor rata-rata |
|---|---|
| DENGAN deskripsi | **4.92 / 5** |
| TANPA deskripsi | **3.50 / 5** |
| **Delta** | **+1.42** (≈ +40% relatif) |

Per query: **with menang 7, seri 5, kalah 0** (12 query). Kemenangan terbesar di judul terse
(mis. "manajemen pemasaran" 5 vs 1; "struktur data", "kecerdasan buatan", "akuntansi biaya" 5 vs 2).

## Kesimpulan jujur (untuk skripsi)
Enrichment metadata **bukan** alat peningkat ranking (klaim +25% lama keliru). Kontribusi nyatanya:
(a) **coverage deskripsi 5.8%→93.4%**, dan (b) **kualitas/justifikasi jawaban naik 3.50→4.92 / 5**
karena LLM memakai deskripsi untuk alasan rekomendasi konkret. Dampak ke ranking netral karena
metadata judul/topik sudah dominan di BM25.

## 4. Akurasi FAKTUAL deskripsi (audit `09c`/`audit_enrichment_accuracy.py`) — temuan KRITIS
Sampel 25 buku enriched (kosong→terisi), cross-check Google Books + LLM-judge:
| Metrik | Hasil |
|---|---|
| Terverifikasi sumber eksternal (Google Books) | **0%** → praktis SEMUA LLM-generated dari metadata |
| Konsistensi topik (judul/DDC) | 4.64/5 (mostly on-topic) |
| Mengandung **fabrikasi spesifik** (klaim isi tak terverifikasi) | **~48%** |
| Reasoning LLM bocor jadi deskripsi (scan 8448) | **~51 buku** (mis. "Queen Kilisuci": *"Wait, I need to... the user wants..."*) |
| Deskripsi menyebut meta "DDC ###" | ~2384 (28%, perlu tinjau) |

**Implikasi jujur:** peningkatan kualitas jawaban (3.50→4.92) sebagian ditopang oleh elaborasi
spesifik yang ~separuhnya fabrikatif → deskripsi membuat jawaban lebih meyakinkan SEKALIGUS lebih
mengarang. Bertentangan dgn misi "tidak mengarang". Deskripsi adalah **ringkasan topikal AI,
BUKAN abstrak otoritatif**.

### Mitigasi diterapkan (Opsi 1)
- **Sanitizer** di `clean_description()` (api.py): deskripsi dgn reasoning-leak otomatis di-blank
  (**47 ter-blank**); read-only ke DB (disaring di lapisan kode, sesuai aturan SQL read-only).
- **Disclaimer** pada jawaban berbasis katalog: "_Ringkasan/alasan buku dirangkum otomatis dari
  metadata (bukan kutipan resmi)..._".
- **Reframe skripsi**: enrichment = coverage + answer-assist; akurasi faktual = keterbatasan terukur.

### Opsi 2 (BELUM — kalau ada waktu)
Regenerate ~8448 deskripsi dgn prompt ANTI-FABRIKASI (hanya nyatakan topik/cakupan dari judul+DDC,
dilarang klaim isi spesifik) → rebuild KB → re-audit. Solusi penuh.

## Berkas
`enrichment_eval_sound.csv`, `enrichment_answer_quality.csv`, `enrichment_accuracy.csv`,
`enrichment_sound.log`, `enrichment_answer_quality.log`, `enrichment_accuracy.log`.
(Lama: `before_after_enrichment.csv` — metode sirkular, jangan dipakai.)
