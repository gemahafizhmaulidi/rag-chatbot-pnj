# Hasil Black Box (Concern B) — kriteria ketat vs longgar

## Masalah versi lama (100% menyesatkan)
`08_blackbox_test.py` lama memakai `_contains_any` (OR atas semua keyword) → lulus cukup dengan
SATU kata echo. Banyak case mencampur penanda sukses+gagal, dan kata boilerplate OOS ("petugas")
memuaskan must_contain in-scope → jawaban OOS yang SALAH dihitung lulus. Hasil "100% (50/50)" tidak bermakna.

## Perbaikan (v4, kriteria ketat)
- `expect_oos` eksplisit → query in-scope OTOMATIS GAGAL bila sistem menjawab OOS.
- `must_contain` = SEMUA elemen wajib (AND); tiap elemen tuple = sinonim (any-of) → cek FAKTA, bukan echo.
- `must_type` cek `query_type` untuk stats/greeting/clarification.
- Fakta usang diperbaiki (mis. maksimal pinjam 2 → **3** sesuai TATIB 2026).

## Hasil: 49/50 = 98.0% (run awal) → **50/50 = 100% setelah TC-29 diperbaiki** (run2)
Kedua run pakai kriteria ketat yang sama. Run awal menangkap 1 defect nyata (TC-29); setelah
defect diperbaiki (lihat bawah), re-run = 50/50 tanpa regresi. Ini 100% yang JUJUR (beda dari
100% longgar lama yang mustahil gagal).
Per kategori: Informasi Umum 13/13, Pencarian Katalog 8/8, Rekomendasi 5/5, Out-of-Scope 5/5,
Keamanan 5/5 (instan 0.0s via pre-filter), Edge Case 6/6, Greeting 2/2, Statistik 1/1,
Routing Ambigu **4/5**.

### Satu kegagalan NYATA (yang dulu tersembunyi)
- **TC-29 "cara bebas pustaka buat wisuda gimana ya?"** → sistem menjawab **prosedur peminjaman**,
  bukan prosedur **bebas pustaka**. Akar: KB punya `SOP LAYANAN BEBAS PUSTAKA ONLINE.xlsx`, tapi
  ekstraksi xlsx hanya menangkap **header SOP** (nomor POS, tanggal, pengesahan) — bukan langkah —
  sehingga chunk peminjaman menang ranking. → defect KB-chunking, status: **belum diperbaiki**
  (perlu re-chunk SOP xlsx + rebuild KB). Didokumentasikan jujur.

> Catatan: TC-04 sempat 989 dtk (anomali Ollama stall), tidak memengaruhi kebenaran hasil.

## TC-29 — DIPERBAIKI (2026-06-03)
Akar: langkah SOP bebas pustaka ada di sheet `flowchart` xlsx, tapi `extract_xlsx_sop` hanya baca
sheet pertama (metadata) → langkah hilang dari KB (bug GENERAL untuk semua SOP xlsx). PDF SOP 0 teks
(scanned). Fix: (1) `extract_xlsx_sop` baca SEMUA sheet + header `## <sheet>`; (2) tambah
`knowledge_base/prosedur_bebas_pustaka_pnj.md` (transkripsi setia SOP 701/PL3.A/OT.01.02/2021).
KB rebuild 196→212 chunks. Hard-stop tetap 0.45 (chunk additif, OOS via router). Verifikasi: bebas
pustaka kini dijawab dengan langkah benar (Google Form→repository→cek SLiMS→surat bebas pustaka).

Berkas: `blackbox_v4_run.log` (awal), `blackbox_v4_run2.log` (setelah fix), `kb_rebuild.log`,
`kb_rebuild2.log`, `../blackbox_results.csv`, `../blackbox_summary.txt`.
