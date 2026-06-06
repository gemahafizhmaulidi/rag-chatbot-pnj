"""
build_kb.py — Build ChromaDB Knowledge Base Perpustakaan PNJ
=============================================================
Membaca semua dokumen di folder knowledge_base (PDF + XLSX + MD),
melakukan chunking 512 token overlap 50, embedding dengan bge-m3,
lalu menyimpan ke ChromaDB.

Jalankan SEKALI sebelum api.py:
    pip install chromadb pdfplumber openpyxl
    python build_kb.py

Untuk rebuild dari nol:
    python build_kb.py --rebuild
"""

import os, re, sys, argparse, warnings, pickle
import warnings
warnings.filterwarnings('ignore')

import pdfplumber
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR     = os.path.join(BASE_DIR, 'knowledge_base')
CHROMA_DIR = os.path.join(BASE_DIR, 'knowledge_base', 'chroma_db')
COLLECTION = 'kb_perpustakaan_pnj'
EMBED_MODEL = 'BAAI/bge-m3'

CHUNK_TOKENS   = 512
OVERLAP_TOKENS = 50

# SOP scanned PDFs yang diganti versi XLSX
SKIP_PDFS = {
    'LAYANAN REFERENSI.pdf',
    'SOP LAYANAN BEBAS PUSTAKA ONLINE.pdf',
    'SOP LAYANAN LOKER.pdf',
    'SOP LAYANAN SIRKULASI.pdf',
    'SOP PENGADAAN BAHAN PUSTAKA.pdf',
    'SOP PROSEDUR PENGOLAHAN BAHAN PUSTAKA.pdf',
    # DRAFT belum resmi — konflik dengan TATIB 2024 (batas pinjam 3 vs 2 judul)
    '2026 DRAFT PERATURAN TATIB terbaru2026.docx.pdf',
    '2026 DRAFT SK PERATURAN PERPUSTAKAAN POLITEKNIK NEGERI JAKARTA.docx.pdf',
}

# ─── Text Extraction ──────────────────────────────────────────────────────────

def extract_pdf(path: str) -> str:
    """Ekstrak teks dari PDF berbasis teks (bukan scanned)."""
    text = ''
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + '\n'
    except Exception as e:
        print(f'  [WARN] PDF error {os.path.basename(path)}: {e}')
    return _clean_text(text)


def extract_xlsx_sop(path: str) -> str:
    """Ekstrak teks dari SEMUA sheet XLSX SOP — TERMASUK sheet 'flowchart' yang berisi
    LANGKAH-LANGKAH prosedur. Tiap sheet diberi header '## <nama sheet>' agar chunking
    section-aware memisahnya jadi chunk tersendiri. (Sebelumnya hanya sheet pertama =
    metadata yang terbaca, sehingga langkah SOP hilang dari KB — bug TC-29 bebas pustaka.)"""
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None)
        parts = []
        for sheet_name, df in sheets.items():
            rows = []
            for _, row in df.iterrows():
                cells = [str(v).strip() for v in row
                         if pd.notna(v) and str(v).strip() and str(v).strip() not in ('nan', 'NaN')]
                if cells:
                    rows.append(' | '.join(cells))
            if rows:
                parts.append(f"## {sheet_name}\n" + '\n'.join(rows))
        return _clean_text('\n\n'.join(parts))
    except Exception as e:
        print(f'  [WARN] XLSX error {os.path.basename(path)}: {e}')
        return ''


def extract_md(path: str) -> str:
    """Baca file Markdown."""
    try:
        with open(path, encoding='utf-8') as f:
            return _clean_text(f.read())
    except Exception as e:
        print(f'  [WARN] MD error {os.path.basename(path)}: {e}')
        return ''


def _clean_text(text: str) -> str:
    """Bersihkan teks: hapus whitespace berlebih, karakter sampah."""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\/\|\#\@\&\'\"\n]', ' ', text)
    return text.strip()


# ─── Chunking ─────────────────────────────────────────────────────────────────

# Pattern header section: "A. JUDUL", "1. JUDUL", "1.4. JUDUL", "## JUDUL", "BAB I", "Pasal N"
_SECTION_HEADER = re.compile(
    r'(?=\n(?:[A-Z]\.\s+[A-Z]|[IVX]+\.\s+[A-Z]|BAB\s+[IVX\d]|Pasal\s+\d|##\s+\S|\d+\.\d+\.\s+[A-ZÀ-ɏ]))',
    re.MULTILINE
)

def _split_sections(text: str) -> list[str]:
    """
    Pecah teks pada batas section/pasal/bab sehingga tiap section di-embed terpisah.
    Fallback ke teks utuh kalau tidak ada header ditemukan.
    """
    parts = _SECTION_HEADER.split(text)
    result = [p.strip() for p in parts if p.strip()]
    return result if len(result) > 1 else [text]


def chunk_text(text: str, tokenizer, chunk_tokens: int = CHUNK_TOKENS,
               overlap_tokens: int = OVERLAP_TOKENS) -> list[str]:
    """
    Section-aware sliding window chunking:
    1. Pecah teks pada header section (A., B., Pasal, BAB, ##)
    2. Tiap section di-chunk sendiri dengan sliding window
    Ini mencegah info dari section berbeda bercampur dalam satu embedding.
    """
    sections = _split_sections(text)
    all_chunks = []
    for section in sections:
        all_chunks.extend(_chunk_section(section, tokenizer, chunk_tokens, overlap_tokens))
    return all_chunks


def _chunk_section(text: str, tokenizer, chunk_tokens: int,
                   overlap_tokens: int) -> list[str]:
    """Sliding window chunking untuk satu section."""
    # Split jadi kalimat/paragraf dulu supaya batas chunk tidak potong di tengah kalimat
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]

    chunks: list[str] = []
    current_tokens: list = []
    current_text_parts: list[str] = []

    for para in paragraphs:
        para_tokens = tokenizer.encode(para, add_special_tokens=False)

        # Kalau satu paragraf lebih dari chunk_tokens, split per kalimat
        if len(para_tokens) > chunk_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
                if len(current_tokens) + len(sent_tokens) > chunk_tokens:
                    if current_tokens:
                        chunks.append(' '.join(current_text_parts))
                    # Mulai chunk baru dengan overlap
                    overlap_start = max(0, len(current_tokens) - overlap_tokens)
                    # Hitung token overlap (ambil dari akhir current)
                    overlap_text = tokenizer.decode(current_tokens[overlap_start:]) if overlap_start < len(current_tokens) else ''
                    current_text_parts = [overlap_text, sent] if overlap_text else [sent]
                    current_tokens = (tokenizer.encode(overlap_text, add_special_tokens=False) if overlap_text else []) + sent_tokens
                else:
                    current_text_parts.append(sent)
                    current_tokens.extend(sent_tokens)
        else:
            if len(current_tokens) + len(para_tokens) > chunk_tokens:
                if current_tokens:
                    chunks.append(' '.join(current_text_parts))
                overlap_start = max(0, len(current_tokens) - overlap_tokens)
                overlap_text = tokenizer.decode(current_tokens[overlap_start:]) if overlap_start < len(current_tokens) else ''
                current_text_parts = [overlap_text, para] if overlap_text else [para]
                current_tokens = (tokenizer.encode(overlap_text, add_special_tokens=False) if overlap_text else []) + para_tokens
            else:
                current_text_parts.append(para)
                current_tokens.extend(para_tokens)

    if current_tokens:
        chunks.append(' '.join(current_text_parts))

    # Filter chunk terlalu pendek (< 50 karakter)
    return [c.strip() for c in chunks if len(c.strip()) >= 50]


# ─── Collect Documents ────────────────────────────────────────────────────────

def collect_documents(kb_dir: str) -> list[dict]:
    """
    Kumpulkan semua dokumen dari folder KB.
    Return list of {'source': str, 'doc_type': str, 'text': str}
    """
    docs = []

    for root, dirs, files in os.walk(kb_dir):
        # Skip folder hidden
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for fname in sorted(files):
            if fname.startswith('.'):
                continue

            path = os.path.join(root, fname)
            rel  = os.path.relpath(path, kb_dir)

            if fname.endswith('.pdf'):
                if fname in SKIP_PDFS:
                    print(f'  [SKIP] {rel} (scanned, pakai versi XLSX)')
                    continue
                text = extract_pdf(path)
                if not text:
                    print(f'  [SKIP] {rel} (kosong setelah ekstraksi)')
                    continue
                doc_type = _infer_doc_type(fname)
                docs.append({'source': rel, 'doc_type': doc_type, 'text': text})
                print(f'  [PDF]  {rel} — {len(text):,} chars')

            elif fname.endswith('.xlsx'):
                text = extract_xlsx_sop(path)
                if not text:
                    print(f'  [SKIP] {rel} (kosong)')
                    continue
                doc_type = 'sop'
                docs.append({'source': rel, 'doc_type': doc_type, 'text': text})
                print(f'  [XLSX] {rel} — {len(text):,} chars')

            elif fname.endswith('.md'):
                text = extract_md(path)
                if not text:
                    continue
                docs.append({'source': rel, 'doc_type': 'knowledge_base', 'text': text})
                print(f'  [MD]   {rel} — {len(text):,} chars')

    return docs


def _infer_doc_type(fname: str) -> str:
    fname_lower = fname.lower()
    if 'sop' in fname_lower:      return 'sop'
    if 'tatib' in fname_lower or 'tata tertib' in fname_lower or 'peraturan' in fname_lower: return 'peraturan'
    if 'pedoman' in fname_lower:  return 'pedoman'
    if 'renstra' in fname_lower:  return 'renstra'
    if 'kubaca' in fname_lower:   return 'panduan_digital'
    if 'ebook' in fname_lower or 'e-book' in fname_lower: return 'panduan_digital'
    if 'akreditasi' in fname_lower: return 'akreditasi'
    return 'dokumen'


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Build ChromaDB KB for Perpustakaan PNJ')
    parser.add_argument('--rebuild', action='store_true', help='Hapus collection lama dan build ulang')
    args = parser.parse_args()

    print('=' * 60)
    print('BUILD KB — Perpustakaan PNJ')
    print('=' * 60)

    # ── Load embedding model ──────────────────────────────────────
    print(f'\n[1/4] Load embedding model {EMBED_MODEL}...')
    model = SentenceTransformer(EMBED_MODEL)
    tokenizer = model.tokenizer
    print('      Model loaded.')

    # ── ChromaDB setup ────────────────────────────────────────────
    print(f'\n[2/4] Setup ChromaDB di {CHROMA_DIR}...')
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if args.rebuild:
        try:
            client.delete_collection(COLLECTION)
            print(f'      Collection lama dihapus.')
        except Exception:
            pass

    try:
        collection = client.get_collection(COLLECTION)
        existing = collection.count()
        print(f'      Collection sudah ada: {existing} chunks.')
        if not args.rebuild:
            print('      Gunakan --rebuild untuk membangun ulang.')
            print('      Selesai (tidak ada perubahan).')
            return
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={'hnsw:space': 'cosine'}   # cosine distance
    )
    print('      Collection baru dibuat.')

    # ── Collect & chunk documents ──────────────────────────────────
    print(f'\n[3/4] Ekstrak dan chunk dokumen dari {KB_DIR}...')
    docs = collect_documents(KB_DIR)
    print(f'\n      Total dokumen: {len(docs)}')

    all_chunks    = []
    all_ids       = []
    all_metadatas = []

    for doc in docs:
        chunks = chunk_text(doc['text'], tokenizer, CHUNK_TOKENS, OVERLAP_TOKENS)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{re.sub(r'[^a-z0-9]', '_', doc['source'].lower())}_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                'source'   : doc['source'],
                'doc_type' : doc['doc_type'],
                'chunk_idx': i,
            })

    print(f'      Total chunks: {len(all_chunks)}')

    # ── Embed dan simpan ke ChromaDB ──────────────────────────────
    print(f'\n[4/4] Embed {len(all_chunks)} chunks dengan {EMBED_MODEL}...')
    BATCH = 64
    all_embeddings = []
    for start in range(0, len(all_chunks), BATCH):
        batch = all_chunks[start:start+BATCH]
        embs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(embs.tolist())
        print(f'      Embedded {min(start+BATCH, len(all_chunks))}/{len(all_chunks)}...', end='\r')
    print()

    # Simpan ke ChromaDB dalam batch
    CHROMA_BATCH = 500
    for start in range(0, len(all_chunks), CHROMA_BATCH):
        end = start + CHROMA_BATCH
        collection.add(
            documents  = all_chunks[start:end],
            embeddings = all_embeddings[start:end],
            ids        = all_ids[start:end],
            metadatas  = all_metadatas[start:end],
        )

    total = collection.count()
    print(f'\n✅ Selesai! {total} chunks tersimpan di ChromaDB.')
    print(f'   Path: {CHROMA_DIR}')
    print(f'\nSekarang jalankan: python api.py')


if __name__ == '__main__':
    main()
