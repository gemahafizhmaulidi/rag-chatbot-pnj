/**
 * chatbot-widget.js — Asisten Perpustakaan PNJ (Modern Redesign)
 * Embed: <script src="chatbot-widget.js" data-api="http://localhost:5001"></script>
 */
(function () {
  'use strict';

  const scriptEl = document.currentScript ||
    document.querySelector('script[src*="chatbot-widget"]');

  const API_BASE = (scriptEl && scriptEl.getAttribute('data-api')) || 'http://localhost:5001';
  const TITLE    = (scriptEl && scriptEl.getAttribute('data-title')) || 'Asisten Perpustakaan PNJ';

  // ── Maskot SVG (profesor) ──────────────────────────────────────────────────
  const PROF_SVG = `
    <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" fill="none">
      <!-- Body / jas -->
      <rect x="14" y="38" width="36" height="22" rx="6" fill="#283593"/>
      <!-- Kemeja putih -->
      <rect x="26" y="38" width="12" height="22" rx="2" fill="#E8EAF6"/>
      <!-- Dasi -->
      <polygon points="32,40 30,46 32,52 34,46" fill="#E53935"/>
      <!-- Leher -->
      <rect x="27" y="34" width="10" height="7" rx="3" fill="#FFCC80"/>
      <!-- Kepala -->
      <ellipse cx="32" cy="26" rx="13" ry="13" fill="#FFCC80"/>
      <!-- Telinga -->
      <ellipse cx="19" cy="26" rx="3" ry="4" fill="#FFCC80"/>
      <ellipse cx="45" cy="26" rx="3" ry="4" fill="#FFCC80"/>
      <!-- Rambut / kepala atas -->
      <ellipse cx="32" cy="15" rx="12" ry="6" fill="#5D4037"/>
      <!-- Topi wisuda (papan) -->
      <rect x="18" y="11" width="28" height="4" rx="2" fill="#1A237E"/>
      <!-- Topi wisuda (tudung) -->
      <rect x="25" y="7" width="14" height="6" rx="2" fill="#1A237E"/>
      <!-- Tali topi -->
      <line x1="46" y1="13" x2="50" y2="20" stroke="#FFD600" stroke-width="1.5" stroke-linecap="round"/>
      <circle cx="50" cy="21" r="2" fill="#FFD600"/>
      <!-- Mata kiri (kacamata) -->
      <rect x="22" y="24" width="9" height="7" rx="3.5" fill="none" stroke="#5D4037" stroke-width="1.5"/>
      <circle cx="26.5" cy="27.5" r="2.5" fill="#fff"/>
      <circle cx="27" cy="27" r="1.2" fill="#1A237E"/>
      <!-- Mata kanan (kacamata) -->
      <rect x="33" y="24" width="9" height="7" rx="3.5" fill="none" stroke="#5D4037" stroke-width="1.5"/>
      <circle cx="37.5" cy="27.5" r="2.5" fill="#fff"/>
      <circle cx="38" cy="27" r="1.2" fill="#1A237E"/>
      <!-- Sambungan kacamata -->
      <line x1="31" y1="27.5" x2="33" y2="27.5" stroke="#5D4037" stroke-width="1.5"/>
      <!-- Hidung -->
      <ellipse cx="32" cy="31" rx="2" ry="1.5" fill="#FFAB40" opacity="0.7"/>
      <!-- Kumis -->
      <path d="M27 33.5 Q29.5 32.5 32 33.5 Q34.5 32.5 37 33.5" stroke="#5D4037" stroke-width="1.2" stroke-linecap="round" fill="none"/>
      <!-- Senyum -->
      <path d="M28 35.5 Q32 38 36 35.5" stroke="#5D4037" stroke-width="1.5" stroke-linecap="round" fill="none"/>
      <!-- Tangan kiri pegang buku -->
      <rect x="6" y="44" width="12" height="9" rx="2" fill="#E53935"/>
      <line x1="12" y1="44" x2="12" y2="53" stroke="#fff" stroke-width="0.8" opacity="0.6"/>
      <rect x="6" y="43" width="12" height="2" rx="1" fill="#B71C1C"/>
      <!-- Lengan kiri -->
      <path d="M14 42 Q10 44 8 48" stroke="#283593" stroke-width="5" stroke-linecap="round"/>
    </svg>`;

  const PROF_SMALL = `
    <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" fill="none">
      <rect x="14" y="38" width="36" height="22" rx="6" fill="#283593"/>
      <rect x="26" y="38" width="12" height="22" rx="2" fill="#E8EAF6"/>
      <polygon points="32,40 30,46 32,52 34,46" fill="#E53935"/>
      <rect x="27" y="34" width="10" height="7" rx="3" fill="#FFCC80"/>
      <ellipse cx="32" cy="26" rx="13" ry="13" fill="#FFCC80"/>
      <ellipse cx="19" cy="26" rx="3" ry="4" fill="#FFCC80"/>
      <ellipse cx="45" cy="26" rx="3" ry="4" fill="#FFCC80"/>
      <ellipse cx="32" cy="15" rx="12" ry="6" fill="#5D4037"/>
      <rect x="18" y="11" width="28" height="4" rx="2" fill="#1A237E"/>
      <rect x="25" y="7" width="14" height="6" rx="2" fill="#1A237E"/>
      <line x1="46" y1="13" x2="50" y2="20" stroke="#FFD600" stroke-width="1.5" stroke-linecap="round"/>
      <circle cx="50" cy="21" r="2" fill="#FFD600"/>
      <rect x="22" y="24" width="9" height="7" rx="3.5" fill="none" stroke="#5D4037" stroke-width="1.5"/>
      <circle cx="26.5" cy="27.5" r="2.5" fill="#fff"/>
      <circle cx="27" cy="27" r="1.2" fill="#1A237E"/>
      <rect x="33" y="24" width="9" height="7" rx="3.5" fill="none" stroke="#5D4037" stroke-width="1.5"/>
      <circle cx="37.5" cy="27.5" r="2.5" fill="#fff"/>
      <circle cx="38" cy="27" r="1.2" fill="#1A237E"/>
      <line x1="31" y1="27.5" x2="33" y2="27.5" stroke="#5D4037" stroke-width="1.5"/>
      <ellipse cx="32" cy="31" rx="2" ry="1.5" fill="#FFAB40" opacity="0.7"/>
      <path d="M27 33.5 Q29.5 32.5 32 33.5 Q34.5 32.5 37 33.5" stroke="#5D4037" stroke-width="1.2" stroke-linecap="round" fill="none"/>
      <path d="M28 35.5 Q32 38 36 35.5" stroke="#5D4037" stroke-width="1.5" stroke-linecap="round" fill="none"/>
    </svg>`;

  // ── Styles ─────────────────────────────────────────────────────────────────
  const css = `
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    #pnj-widget-btn {
      position: fixed; bottom: 28px; right: 28px; z-index: 9999;
      width: 64px; height: 64px; border-radius: 50%;
      background: linear-gradient(135deg, #3949AB, #5C6BC0);
      border: none; cursor: pointer;
      box-shadow: 0 4px 20px rgba(57,73,171,0.45);
      display: flex; align-items: center; justify-content: center;
      transition: transform .2s cubic-bezier(.34,1.56,.64,1), box-shadow .2s;
      padding: 8px;
    }
    #pnj-widget-btn:hover {
      transform: scale(1.1);
      box-shadow: 0 8px 28px rgba(57,73,171,0.55);
    }
    #pnj-widget-btn svg { width: 48px; height: 48px; }

    #pnj-badge {
      position: absolute; top: -2px; right: -2px;
      width: 20px; height: 20px; border-radius: 50%;
      background: #F44336; color: #fff; font-size: 11px; font-weight: 700;
      display: none; align-items: center; justify-content: center;
      border: 2px solid #fff; font-family: Inter, sans-serif;
    }

    #pnj-widget-panel {
      position: fixed; bottom: 104px; right: 28px; z-index: 9998;
      width: 380px; max-width: calc(100vw - 32px);
      height: 580px; max-height: calc(100vh - 120px);
      background: #FAFAFA; border-radius: 24px;
      box-shadow: 0 12px 48px rgba(0,0,0,0.16), 0 2px 8px rgba(0,0,0,0.08);
      display: flex; flex-direction: column; overflow: hidden;
      transform: scale(0.88) translateY(20px); opacity: 0; pointer-events: none;
      transition: transform .25s cubic-bezier(.34,1.56,.64,1), opacity .2s;
      font-family: Inter, -apple-system, sans-serif;
    }
    #pnj-widget-panel.open {
      transform: scale(1) translateY(0); opacity: 1; pointer-events: all;
    }

    /* Header */
    #pnj-hdr {
      background: linear-gradient(135deg, #283593 0%, #3949AB 50%, #5C6BC0 100%);
      padding: 16px 16px 14px;
      display: flex; align-items: center; gap: 12px;
      flex-shrink: 0;
      position: relative; overflow: hidden;
    }
    #pnj-hdr::before {
      content: ''; position: absolute; top: -20px; right: -20px;
      width: 100px; height: 100px; border-radius: 50%;
      background: rgba(255,255,255,0.06);
    }
    #pnj-hdr::after {
      content: ''; position: absolute; bottom: -30px; left: 30%;
      width: 80px; height: 80px; border-radius: 50%;
      background: rgba(255,255,255,0.04);
    }
    #pnj-hdr-mascot {
      width: 48px; height: 48px; flex-shrink: 0;
      background: rgba(255,255,255,0.15);
      border-radius: 50%; padding: 4px;
      backdrop-filter: blur(4px);
    }
    #pnj-hdr-mascot svg { width: 40px; height: 40px; }
    #pnj-hdr-info { flex: 1; min-width: 0; }
    #pnj-hdr-name {
      font-size: 15px; font-weight: 700; color: #fff;
      letter-spacing: -0.2px;
    }
    #pnj-hdr-sub {
      font-size: 11.5px; color: rgba(255,255,255,0.75);
      margin-top: 2px; display: flex; align-items: center; gap: 5px;
    }
    #pnj-hdr-dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: #69F0AE; flex-shrink: 0;
      box-shadow: 0 0 6px #69F0AE;
      animation: pnjpulse 2s infinite;
    }
    @keyframes pnjpulse {
      0%,100% { opacity: 1; } 50% { opacity: 0.5; }
    }
    #pnj-hdr-close {
      background: rgba(255,255,255,0.12); border: none; color: #fff;
      cursor: pointer; width: 32px; height: 32px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 16px; transition: background .15s; flex-shrink: 0;
      backdrop-filter: blur(4px);
    }
    #pnj-hdr-close:hover { background: rgba(255,255,255,0.25); }

    /* Messages */
    #pnj-msgs {
      flex: 1; overflow-y: auto; padding: 16px 14px;
      display: flex; flex-direction: column; gap: 12px;
      background: #F0F2FF;
    }
    #pnj-msgs::-webkit-scrollbar { width: 4px; }
    #pnj-msgs::-webkit-scrollbar-thumb { background: #C5CAE9; border-radius: 4px; }

    .pnj-bubble-row {
      display: flex; gap: 8px; align-items: flex-end;
    }
    .pnj-bubble-row.bot { align-self: flex-start; max-width: 88%; }
    .pnj-bubble-row.user { align-self: flex-end; flex-direction: row-reverse; max-width: 80%; }

    .pnj-avatar {
      width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
      background: linear-gradient(135deg, #3949AB, #5C6BC0);
      display: flex; align-items: center; justify-content: center;
      padding: 4px; box-shadow: 0 2px 8px rgba(57,73,171,0.3);
    }
    .pnj-avatar svg { width: 24px; height: 24px; }

    .pnj-col { display: flex; flex-direction: column; }

    .pnj-bubble {
      padding: 10px 14px; border-radius: 18px;
      font-size: 13.5px; line-height: 1.55; word-break: break-word;
      font-family: Inter, -apple-system, sans-serif;
    }
    .pnj-bubble.bot {
      background: #fff; color: #1a1a2e;
      border-bottom-left-radius: 6px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .pnj-bubble.user {
      background: linear-gradient(135deg, #3949AB, #5C6BC0);
      color: #fff; border-bottom-right-radius: 6px;
      box-shadow: 0 2px 8px rgba(57,73,171,0.35);
    }
    .pnj-bubble strong { font-weight: 600; }
    .pnj-bubble a { color: #3949AB; text-decoration: underline; }
    .pnj-bubble.user a { color: #C5CAE9; }
    .pnj-bubble ul { margin: 6px 0 6px 16px; padding: 0; }
    .pnj-bubble li { margin-bottom: 3px; }

    .pnj-time {
      font-size: 10px; color: #9E9E9E; margin-top: 4px;
      font-family: Inter, sans-serif;
    }
    .pnj-bubble-row.user .pnj-time { text-align: right; }

    /* Typing */
    .pnj-typing { display: flex; gap: 5px; align-items: center; padding: 2px 4px; }
    .pnj-typing span {
      width: 8px; height: 8px; border-radius: 50%;
      background: #9FA8DA; animation: pnjbounce 1.2s infinite;
    }
    .pnj-typing span:nth-child(2) { animation-delay: .2s; }
    .pnj-typing span:nth-child(3) { animation-delay: .4s; }
    @keyframes pnjbounce {
      0%,80%,100% { transform: translateY(0); opacity: 0.5; }
      40% { transform: translateY(-7px); opacity: 1; }
    }

    /* Quick replies */
    #pnj-quick {
      padding: 8px 14px 4px; display: flex; gap: 6px; flex-wrap: wrap;
      background: #F0F2FF;
    }
    .pnj-qr {
      font-size: 12px; padding: 6px 12px; border-radius: 20px; cursor: pointer;
      border: 1.5px solid #9FA8DA; color: #3949AB; background: #fff;
      font-family: Inter, sans-serif; font-weight: 500;
      white-space: nowrap; transition: all .15s;
      box-shadow: 0 1px 4px rgba(57,73,171,0.1);
    }
    .pnj-qr:hover {
      background: linear-gradient(135deg, #3949AB, #5C6BC0);
      color: #fff; border-color: transparent;
      box-shadow: 0 2px 8px rgba(57,73,171,0.3);
    }

    /* Input */
    #pnj-input-row {
      padding: 12px 14px; display: flex; gap: 8px; align-items: flex-end;
      border-top: 1px solid #E8EAF6; background: #fff; flex-shrink: 0;
    }
    #pnj-input {
      flex: 1; resize: none;
      border: 1.5px solid #E8EAF6; border-radius: 14px;
      padding: 9px 12px; font-size: 13.5px; line-height: 1.4;
      font-family: Inter, -apple-system, sans-serif;
      outline: none; max-height: 90px; overflow-y: auto;
      transition: border-color .15s, box-shadow .15s;
      background: #FAFAFA; color: #1a1a2e;
    }
    #pnj-input::placeholder { color: #BDBDBD; }
    #pnj-input:focus {
      border-color: #5C6BC0;
      box-shadow: 0 0 0 3px rgba(92,107,192,0.12);
      background: #fff;
    }
    #pnj-send {
      width: 40px; height: 40px; border-radius: 12px; flex-shrink: 0;
      background: linear-gradient(135deg, #3949AB, #5C6BC0);
      border: none; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      transition: all .15s; padding: 0;
      box-shadow: 0 2px 8px rgba(57,73,171,0.35);
    }
    #pnj-send:hover {
      transform: scale(1.05);
      box-shadow: 0 4px 12px rgba(57,73,171,0.45);
    }
    #pnj-send:disabled { background: #E0E0E0; cursor: not-allowed; box-shadow: none; transform: none; }
    #pnj-send svg { width: 18px; height: 18px; }

    /* Source note */
    .pnj-src-note {
      font-size: 10.5px; color: #9E9E9E; margin-top: 4px;
      font-family: Inter, sans-serif; display: flex; align-items: center; gap: 4px;
    }

    /* Feedback */
    .pnj-feedback {
      display: flex; gap: 6px; margin-top: 6px; align-items: center;
    }
    .pnj-fb-btn {
      font-size: 14px; background: #F5F5F5; border: 1px solid #E0E0E0;
      cursor: pointer; padding: 3px 8px; border-radius: 20px;
      transition: all .15s; line-height: 1; font-family: Inter, sans-serif;
    }
    .pnj-fb-btn:hover { background: #E8EAF6; border-color: #9FA8DA; }
    .pnj-fb-label { font-size: 10.5px; color: #BDBDBD; font-family: Inter, sans-serif; }

    /* Divider waktu */
    .pnj-divider {
      text-align: center; font-size: 10.5px; color: #BDBDBD;
      font-family: Inter, sans-serif; margin: 2px 0;
      display: flex; align-items: center; gap: 8px;
    }
    .pnj-divider::before, .pnj-divider::after {
      content: ''; flex: 1; height: 1px; background: #E0E0E0;
    }
  `;

  // ── Inject styles ──────────────────────────────────────────────────────────
  const link = document.createElement('link');
  link.rel = 'preconnect'; link.href = 'https://fonts.googleapis.com';
  document.head.appendChild(link);

  const style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);

  // ── Toggle button ──────────────────────────────────────────────────────────
  const btn = document.createElement('button');
  btn.id = 'pnj-widget-btn';
  btn.setAttribute('aria-label', 'Buka asisten perpustakaan');
  btn.innerHTML = PROF_SVG + `<span id="pnj-badge"></span>`;
  document.body.appendChild(btn);

  // ── Panel ──────────────────────────────────────────────────────────────────
  const panel = document.createElement('div');
  panel.id = 'pnj-widget-panel';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-label', TITLE);
  panel.innerHTML = `
    <div id="pnj-hdr">
      <div id="pnj-hdr-mascot">${PROF_SMALL}</div>
      <div id="pnj-hdr-info">
        <div id="pnj-hdr-name">${TITLE}</div>
        <div id="pnj-hdr-sub">
          <span id="pnj-hdr-dot"></span>
          Online — siap membantu
        </div>
      </div>
      <button id="pnj-hdr-close" aria-label="Tutup">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>
    <div id="pnj-msgs" aria-live="polite"></div>
    <div id="pnj-quick"></div>
    <div id="pnj-input-row">
      <textarea id="pnj-input" placeholder="Ketik pertanyaan..." rows="1" aria-label="Pesan"></textarea>
      <button id="pnj-send" aria-label="Kirim">
        <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="22" y1="2" x2="11" y2="13"/>
          <polygon points="22 2 15 22 11 13 2 9 22 2"/>
        </svg>
      </button>
    </div>
  `;
  document.body.appendChild(panel);

  const msgsEl  = panel.querySelector('#pnj-msgs');
  const inputEl = panel.querySelector('#pnj-input');
  const sendBtn = panel.querySelector('#pnj-send');
  const quickEl = panel.querySelector('#pnj-quick');
  const badge   = btn.querySelector('#pnj-badge');

  // ── Quick replies ──────────────────────────────────────────────────────────
  const QUICK_REPLIES = [
    'Ada buku pemrograman Python?',
    'Jam buka perpustakaan?',
    'Berapa denda telat?',
    'Berapa total koleksi buku?',
  ];

  function renderQuickReplies() {
    quickEl.innerHTML = '';
    QUICK_REPLIES.forEach(q => {
      const b = document.createElement('button');
      b.className = 'pnj-qr';
      b.textContent = q;
      b.onclick = () => { sendMessage(q); quickEl.innerHTML = ''; };
      quickEl.appendChild(b);
    });
  }

  // ── Markdown renderer ──────────────────────────────────────────────────────
  function renderMarkdown(text) {
    return text
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/`(.+?)`/g, '<code style="background:#F3F4F6;padding:1px 5px;border-radius:4px;font-size:12px">$1</code>')
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
      .replace(/^[\*\-] (.+)$/gm, '<li>$1</li>')
      .replace(/(<li>[\s\S]*?<\/li>)/g, '<ul style="margin:6px 0 6px 16px">$1</ul>')
      .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
      .replace(/\n/g, '<br>');
  }

  function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function nowTime() {
    return new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
  }

  // ── Append message ─────────────────────────────────────────────────────────
  function appendMessage(role, text, opts = {}) {
    const row = document.createElement('div');
    row.className = `pnj-bubble-row ${role}`;

    if (role === 'bot') {
      const av = document.createElement('div');
      av.className = 'pnj-avatar';
      av.innerHTML = PROF_SMALL;
      row.appendChild(av);
    }

    const col = document.createElement('div');
    col.className = 'pnj-col';

    const bubble = document.createElement('div');
    bubble.className = `pnj-bubble ${role}`;
    bubble.innerHTML = role === 'bot' ? renderMarkdown(text) : escHtml(text);
    col.appendChild(bubble);

    const t = document.createElement('div');
    t.className = 'pnj-time';
    t.textContent = nowTime();
    col.appendChild(t);

    if (role === 'bot' && !opts.noFeedback) {
      const fb = document.createElement('div');
      fb.className = 'pnj-feedback';
      fb.innerHTML = `<span class="pnj-fb-label">Membantu?</span>
        <button class="pnj-fb-btn">👍</button>
        <button class="pnj-fb-btn">👎</button>`;
      fb.querySelectorAll('.pnj-fb-btn').forEach(b2 => {
        b2.onclick = () => {
          fb.innerHTML = `<span class="pnj-fb-label" style="color:#43A047">Terima kasih! 🙏</span>`;
        };
      });
      col.appendChild(fb);
    }

    row.appendChild(col);
    msgsEl.appendChild(row);
    msgsEl.scrollTop = msgsEl.scrollHeight;
    return bubble;
  }

  function showTyping() {
    const row = document.createElement('div');
    row.className = 'pnj-bubble-row bot'; row.id = 'pnj-typing-row';
    const av = document.createElement('div');
    av.className = 'pnj-avatar'; av.innerHTML = PROF_SMALL;
    const bubble = document.createElement('div');
    bubble.className = 'pnj-bubble bot';
    bubble.innerHTML = '<div class="pnj-typing"><span></span><span></span><span></span></div>';
    row.appendChild(av); row.appendChild(bubble);
    msgsEl.appendChild(row);
    msgsEl.scrollTop = msgsEl.scrollHeight;
  }

  function removeTyping() {
    const r = document.getElementById('pnj-typing-row');
    if (r) r.remove();
  }

  // ── API ────────────────────────────────────────────────────────────────────
  async function callAPI(message) {
    const resp = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
      signal: AbortSignal.timeout(150000),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
  }

  // ── Send ───────────────────────────────────────────────────────────────────
  let busy = false;

  async function sendMessage(text) {
    text = text.trim();
    if (!text || busy) return;

    busy = true;
    sendBtn.disabled = true;
    inputEl.value = '';
    inputEl.style.height = '';

    appendMessage('user', text);
    showTyping();

    try {
      const data = await callAPI(text);
      removeTyping();
      appendMessage('bot', data.answer || 'Maaf, tidak ada jawaban.');

      const srcs = data.sources || [];
      const lastRow = msgsEl.lastElementChild;
      if (srcs.length > 0) {
        const note = document.createElement('div');
        note.className = 'pnj-src-note';
        const icon = data.query_type === 'book_search' ? '📖' : '📄';
        const label = data.query_type === 'book_search'
          ? `${srcs.length} sumber katalog`
          : 'Basis pengetahuan perpustakaan';
        note.innerHTML = `${icon} <span>${label} · ${data.elapsed_s?.toFixed(1) ?? '?'}s</span>`;
        lastRow.querySelector('.pnj-col').appendChild(note);
      }
    } catch (err) {
      removeTyping();
      let msg = 'Maaf, terjadi kesalahan. Pastikan server sedang berjalan.';
      if (err.name === 'TimeoutError') msg = 'Permintaan timeout. Silakan coba lagi.';
      appendMessage('bot', msg, { noFeedback: true });
    }

    busy = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }

  // ── Toggle ─────────────────────────────────────────────────────────────────
  let opened = false;

  function openPanel() {
    panel.classList.add('open');
    btn.setAttribute('aria-expanded', 'true');
    opened = true;
    badge.style.display = 'none';
    setTimeout(() => inputEl.focus(), 250);
  }

  function closePanel() {
    panel.classList.remove('open');
    btn.setAttribute('aria-expanded', 'false');
    opened = false;
  }

  btn.addEventListener('click', () => opened ? closePanel() : openPanel());
  panel.querySelector('#pnj-hdr-close').addEventListener('click', closePanel);

  inputEl.addEventListener('input', () => {
    inputEl.style.height = '';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 90) + 'px';
  });

  inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(inputEl.value); }
  });

  sendBtn.addEventListener('click', () => sendMessage(inputEl.value));

  document.addEventListener('click', e => {
    if (opened && !panel.contains(e.target) && !btn.contains(e.target)) closePanel();
  });

  // ── Init ───────────────────────────────────────────────────────────────────
  function init() {
    // Divider
    const div = document.createElement('div');
    div.className = 'pnj-divider';
    div.textContent = 'Hari ini';
    msgsEl.appendChild(div);

    appendMessage('bot',
      'Halo! Saya **Prof. Pus** 🎓, asisten virtual Perpustakaan PNJ.\n\n' +
      'Saya siap membantu Anda:\n' +
      '* 📚 Mencari buku di koleksi perpustakaan\n' +
      '* 🔍 Memberikan rekomendasi buku\n' +
      '* ℹ️ Menjawab pertanyaan layanan & peraturan\n\n' +
      'Ada yang bisa saya bantu?',
      { noFeedback: true }
    );
    renderQuickReplies();
  }

  init();
})();
