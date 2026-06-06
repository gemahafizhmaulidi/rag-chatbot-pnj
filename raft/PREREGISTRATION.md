# Pra-Registrasi Eksperimen — RAG vs RAG+RAFT

> **Ditulis SEBELUM model RAFT dilatih & sebelum hasil dilihat.**
> Tujuan: mengunci kriteria keputusan agar komparasi tidak bisa dituduh *cherry-pick*.
> Tanggal kunci: 2026-06-04.

## 1. Pertanyaan penelitian
Apakah menambahkan **Retrieval-Augmented Fine-Tuning (RAFT)** pada generator LLM
(Qwen3.5-4B) meningkatkan kualitas jawaban chatbot perpustakaan PNJ secara
**bermakna** dibanding RAG dengan generator base (tanpa fine-tune)?

## 2. Dua arm (retrieval DIKUNCI identik)
| Arm | Generator | Retrieval | Serving |
|-----|-----------|-----------|---------|
| **A (baseline)** | Qwen3.5-4B base | Hybrid BM25(0.45)+Dense(0.55), Config C | Ollama GGUF Q4_K_M |
| **B (RAFT)** | Qwen3.5-4B + QLoRA RAFT | **sama persis** | Ollama GGUF Q4_K_M |

Satu-satunya variabel yang berubah = bobot generator. Semua lain (retriever, prompt
sistem, threshold, kuantisasi Q4_K_M, dekoding) identik.

## 3. Data
- **Latih RAFT**: 883 triple sintetik (`out/raft_train.jsonl`), teacher Qwen-35B,
  non-thinking. Held-out evaluasi **tidak** masuk training (divalidasi).
- **Uji (held-out, tidak pernah dilatih)**:
  - 50 query RAGAS (`data/ground_truth_ragas.json`): 40 in-scope + 10 OOS
  - 50 skenario black box (`scripts/08_blackbox_test.py`, kriteria ketat)

## 4. Metrik & alat
1. **RAGAS Faithfulness** & **Answer Relevancy** — judge `gpt-4o-mini` (≠ teacher → bias rendah).
2. **Black box pass rate** — kriteria ketat existing (expect_oos + fakta wajib + query_type).
3. **Blind pairwise judge** — `gpt-4o-mini`, urutan A/B diacak per item, menilai
   jawaban mana lebih baik (menang/seri/kalah) pada 40 query in-scope.
4. **Latensi** rata-rata per jawaban (sekunder).

## 5. KEPUTUSAN (dikunci sekarang)
Adopsi RAFT (Arm B) ke produksi **hanya jika SEMUA** terpenuhi:
- **(K1)** RAGAS Faithfulness B ≥ A − 0.01 (tidak menurunkan faithfulness), DAN
- **(K2)** RAGAS Answer Relevancy B ≥ A + 0.02 (naik bermakna), DAN
- **(K3)** Black box pass rate B ≥ A (tidak ada regresi keamanan/akurasi), DAN
- **(K4)** Blind pairwise: kemenangan B > kemenangan A dengan selisih ≥ 15% dari 40
  item (≥ 6 item), DAN
- **(K5)** Tidak ada regresi keras: B tidak pernah membocorkan system prompt,
  tidak menjawab OOS yang seharusnya ditolak, tidak menghaluskan fakta (denda,
  jam, nomor) yang di Arm A benar.

Jika **tidak semua** terpenuhi → **RAG-only (Arm A) dipertahankan**, dan hasil
dilaporkan sebagai bukti empiris bahwa RAFT tidak memberi manfaat bersih pada
kasus ini (temuan tetap valid & melampaui referensi yang tidak punya komparasi ini).

## 6. Konsekuensi yang diterima di muka
- Jika B menang telak (semua K1–K5 terpenuhi) → produksi **wajib** pindah ke Arm B;
  argumen *maintainability* turun jadi trade-off sadar, bukan klaim superioritas.
- Jika B kalah/seri → RAFT tidak dipakai; **bukan** karena enggan, tapi karena data
  menunjukkan demikian. Tidak ada re-run selektif untuk "mengejar" B menang.

## 7. Ancaman validitas yang diakui
- Base model strong → ruang perbaikan sempit (efek lantai/atap).
- Data latih sintetik (teacher 35B) → batas atas ≈ kualitas teacher.
- T4 + QLoRA 4-bit → kapasitas adaptasi terbatas; bukan full fine-tune.
- Judge LLM (gpt-4o-mini) punya bias gaya; dimitigasi blind + random-swap + teacher≠judge.
