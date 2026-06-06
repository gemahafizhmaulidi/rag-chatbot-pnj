"""core/db.py — Catalog loader dan stats queries (read-only)"""

import re, logging
import pandas as pd
import mysql.connector
from core.config import DB_CONFIG

log = logging.getLogger(__name__)

_DESC_NOISE = re.compile(
    r'^(Deskripsi\s*katalog\s*:\s*|Deskripsi\s*:\s*|PREFACE\s*)', re.IGNORECASE)

_GARBAGE_DESC = re.compile(
    r"\b(wait,|i need to|the user wants|as an ai|i cannot|i should avoid|"
    r"let me (think|provide|generate|start|re-?read)|here(?:'s| is) a (description|summary)|"
    r"that's a lot of repetition|i'll (now |)provide|sebagai (model|ai))\b",
    re.IGNORECASE)


def clean_description(text: str) -> str:
    if not text:
        return ''
    t = _DESC_NOISE.sub('', str(text).strip())
    t = t.replace('¶', ' ').replace('&Quot;', '"').replace('&quot;', '"')
    t = t.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    t = re.sub(r'\s+', ' ', t).strip()
    if _GARBAGE_DESC.search(t):
        return ''
    return t


def load_catalog(conn) -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT b.biblio_id, b.title AS judul, b.call_number,
               b.notes AS deskripsi, b.publish_year AS tahun,
               GROUP_CONCAT(DISTINCT a.author_name  SEPARATOR ', ') AS penulis,
               GROUP_CONCAT(DISTINCT t.topic        SEPARATOR ', ') AS topik,
               p.publisher_name AS penerbit,
               COUNT(DISTINCT i.item_id) AS total_eksemplar,
               COUNT(DISTINCT CASE WHEN i.item_status_id IN ('0','R') THEN i.item_id END) AS tersedia,
               COALESCE(lh.loan_count, 0) AS loan_count
        FROM biblio b
        JOIN mst_gmd g ON b.gmd_id = g.gmd_id AND g.gmd_code = 'TE'
        LEFT JOIN biblio_author ba ON b.biblio_id = ba.biblio_id
        LEFT JOIN mst_author a    ON ba.author_id = a.author_id
        LEFT JOIN biblio_topic bt ON b.biblio_id = bt.biblio_id
        LEFT JOIN mst_topic t     ON bt.topic_id  = t.topic_id
        LEFT JOIN mst_publisher p ON b.publisher_id = p.publisher_id
        LEFT JOIN item i          ON b.biblio_id = i.biblio_id
        LEFT JOIN (
            SELECT biblio_id, COUNT(*) AS loan_count
            FROM loan_history
            GROUP BY biblio_id
        ) lh ON b.biblio_id = lh.biblio_id
        GROUP BY b.biblio_id
    """, conn)
    df['deskripsi'] = df['deskripsi'].apply(clean_description)
    for col in ['penulis', 'topik', 'penerbit']:
        df[col] = df[col].fillna('').astype(str)
    for col in ['total_eksemplar', 'tersedia', 'loan_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    n_desc  = df['deskripsi'].apply(lambda x: bool(x.strip())).sum()
    n_avail = int((df['tersedia'] > 0).sum())
    log.info(f'Katalog: {len(df):,} buku | {n_desc:,} ada deskripsi | {n_avail:,} tersedia')
    return df.reset_index(drop=True)


# ── Stats (read-only SQL) ─────────────────────────────────────────────────────

_STATS_SQL = {
    'total_books': (
        "SELECT COUNT(*) AS n "
        "FROM biblio b JOIN mst_gmd g ON b.gmd_id=g.gmd_id AND g.gmd_code='TE'"
    ),
    'total_items'    : "SELECT COUNT(*) AS n FROM item",
    'available_books': "SELECT COUNT(*) AS n FROM item WHERE item_status_id IN ('0','R')",
    'loans_this_month': (
        "SELECT COUNT(*) AS n FROM loan_history "
        "WHERE MONTH(loan_date)=MONTH(CURDATE()) AND YEAR(loan_date)=YEAR(CURDATE())"
    ),
    'loans_this_year' : "SELECT COUNT(*) AS n FROM loan_history WHERE YEAR(loan_date)=YEAR(CURDATE())",
    'loans_total'     : "SELECT COUNT(*) AS n FROM loan_history",
    'top_borrowed_week': (
        "SELECT title, COUNT(*) AS n FROM loan_history "
        "WHERE gmd_name='Text' AND loan_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) "
        "GROUP BY biblio_id, title ORDER BY n DESC LIMIT 10"
    ),
    'top_borrowed_month': (
        "SELECT title, COUNT(*) AS n FROM loan_history "
        "WHERE gmd_name='Text' AND loan_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) "
        "GROUP BY biblio_id, title ORDER BY n DESC LIMIT 10"
    ),
    'top_borrowed_alltime': (
        "SELECT title, COUNT(*) AS n FROM loan_history "
        "WHERE gmd_name='Text' "
        "GROUP BY biblio_id, title ORDER BY n DESC LIMIT 10"
    ),
}


def execute_stats_query(subtype: str) -> list:
    sql = _STATS_SQL.get(subtype)
    if not sql:
        return []
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor(dictionary=True)
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close(); conn.close()
        return rows
    except Exception as e:
        log.error(f'Stats query [{subtype}] error: {e}')
        return []
