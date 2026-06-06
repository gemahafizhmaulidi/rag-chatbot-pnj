"""
audit_enrichment_accuracy.py — Audit AKURASI FAKTUAL deskripsi hasil enrichment
================================================================================
Keresahan: pipeline enrichment punya fallback LLM yang MENGARANG deskripsi dari metadata.
Deskripsi itu ditampilkan chatbot → risiko menyesatkan (kontradiksi misi "tidak mengarang").

Audit: sampel buku yang KOSONG sebelum enrichment lalu TERISI (= hasil enrichment). Untuk tiap:
  (a) cross-check ke Google Books (sumber independen) — apakah deskripsi kita mirip sumber asli?
  (b) LLM-judge (gpt-4o-mini): konsistensi deskripsi vs metadata (judul/DDC) + flag FABRIKASI
      (klaim isi spesifik yang tak bisa diketahui hanya dari judul).

Pakai: python3.10 scripts/audit_enrichment_accuracy.py [N]
Output: output/enrichment_accuracy.csv + ringkasan
"""
import os, sys, json, time, random, warnings
warnings.filterwarnings('ignore')
import requests, pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import mysql.connector
DB = dict(host=os.getenv('DB_HOST', 'localhost'), user=os.getenv('DB_USER', 'root'),
          password=os.getenv('DB_PASSWORD', ''))
KEY = os.getenv('OPENROUTER_API_KEY', '')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
random.seed(42)

def q(dbname, sql):
    c = mysql.connector.connect(**DB, database=dbname); cur = c.cursor(dictionary=True)
    cur.execute(sql); rows = cur.fetchall(); c.close(); return rows

# buku enriched = kosong di opac_original, terisi di opacv2
print('Mengambil populasi buku enriched (kosong->terisi)...')
before = {r['biblio_id']: r['notes'] for r in q(os.getenv('DB_NAME_ORIGINAL','opac_original'),
    "SELECT b.biblio_id, b.notes FROM biblio b JOIN mst_gmd g ON b.gmd_id=g.gmd_id AND g.gmd_code='TE'")}
after = q(os.getenv('DB_NAME2','opacv2'), """
    SELECT b.biblio_id, b.title, b.publish_year, b.isbn_issn, b.notes, b.call_number,
           GROUP_CONCAT(DISTINCT a.author_name SEPARATOR ', ') AS penulis,
           GROUP_CONCAT(DISTINCT t.topic SEPARATOR ', ') AS topik
    FROM biblio b JOIN mst_gmd g ON b.gmd_id=g.gmd_id AND g.gmd_code='TE'
    LEFT JOIN biblio_author ba ON b.biblio_id=ba.biblio_id LEFT JOIN mst_author a ON ba.author_id=a.author_id
    LEFT JOIN biblio_topic bt ON b.biblio_id=bt.biblio_id LEFT JOIN mst_topic t ON bt.topic_id=t.topic_id
    GROUP BY b.biblio_id""")
enriched = [r for r in after
            if (r['notes'] and len(str(r['notes']).strip()) > 50)
            and len(str(before.get(r['biblio_id']) or '').strip()) <= 10]
print(f'Total buku enriched: {len(enriched)} | sampel: {N}')
sample = random.sample(enriched, min(N, len(enriched)))

def gbooks(title, author):
    try:
        qy = f'intitle:{title}'
        if author: qy += f'+inauthor:{author.split(",")[0]}'
        r = requests.get('https://www.googleapis.com/books/v1/volumes',
                         params={'q': qy, 'maxResults': 3}, timeout=8)
        for it in r.json().get('items', []):
            d = it.get('volumeInfo', {}).get('description', '')
            if d and len(d) > 60:
                return d[:1000]
    except Exception:
        pass
    return ''

def judge(meta, desc, external):
    sys_p = ("Anda auditor kualitas katalog perpustakaan. Diberi METADATA buku dan DESKRIPSI yang "
             "dihasilkan sistem. Nilai JSON {\"konsisten\": 1-5, \"fabrikasi\": true/false, \"catatan\": \"...\"}:\n"
             "- konsisten: apakah deskripsi konsisten dengan judul/DDC/topik (dan SUMBER EKSTERNAL bila ada)? "
             "5=sangat konsisten; 1=membahas buku berbeda/bertentangan.\n"
             "- fabrikasi: true bila deskripsi menyatakan FAKTA SPESIFIK (klaim isi/temuan/penulis/bab tertentu) "
             "yang TIDAK dapat diketahui hanya dari judul/metadata dan tidak didukung sumber eksternal.")
    u = f"METADATA:\n{meta}\n\nDESKRIPSI SISTEM:\n{desc}"
    if external:
        u += f"\n\nSUMBER EKSTERNAL (Google Books):\n{external}"
    else:
        u += "\n\n(Tidak ada sumber eksternal ditemukan.)"
    try:
        r = requests.post('https://openrouter.ai/api/v1/chat/completions',
            headers={'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'},
            json={'model': 'openai/gpt-4o-mini', 'temperature': 0,
                  'messages': [{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': u}]},
            timeout=60)
        t = r.json()['choices'][0]['message']['content']
        return json.loads(t[t.find('{'):t.rfind('}')+1])
    except Exception as e:
        return {'konsisten': None, 'fabrikasi': None, 'catatan': f'err {e}'}

rows = []
for i, b in enumerate(sample, 1):
    meta = (f"Judul: {b['title']}\nPenulis: {b.get('penulis') or '-'}\nTahun: {b.get('publish_year') or '-'}\n"
            f"DDC: {b.get('call_number') or '-'}\nTopik: {b.get('topik') or '-'}")
    ext = gbooks(str(b['title']), str(b.get('penulis') or ''))
    time.sleep(0.3)
    v = judge(meta, str(b['notes'])[:1200], ext)
    rows.append({'biblio_id': b['biblio_id'], 'judul': str(b['title'])[:50],
                 'ada_gbooks': bool(ext), 'konsisten': v.get('konsisten'),
                 'fabrikasi': v.get('fabrikasi'), 'catatan': str(v.get('catatan',''))[:80]})
    print(f"  [{i:02d}/{len(sample)}] konsisten={v.get('konsisten')} fabrikasi={v.get('fabrikasi')} gbooks={bool(ext)} | {str(b['title'])[:45]}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(BASE, 'output', 'enrichment_accuracy.csv'), index=False)
ok = df.dropna(subset=['konsisten'])
print('\n' + '=' * 60)
print(f"AKURASI DESKRIPSI ENRICHMENT (sampel {len(ok)})")
print('=' * 60)
print(f"  Konsistensi rata-rata (1-5)     : {ok['konsisten'].astype(float).mean():.2f}")
print(f"  Konsisten tinggi (>=4)          : {(ok['konsisten'].astype(float)>=4).mean():.0%}")
print(f"  Ada padanan di Google Books     : {df['ada_gbooks'].mean():.0%}  (sisanya: LLM-generated dari metadata)")
print(f"  Di-flag FABRIKASI               : {(ok['fabrikasi']==True).mean():.0%}")
print(f"\nCSV: output/enrichment_accuracy.csv")
