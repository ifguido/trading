/* CryptoTrader Web UI - Main JS */

let ws = null;
let statusInterval = null;
const livePrices = {};

// ── Status polling ──────────────────────────────────────────
async function pollStatus() {
    try {
        const res = await fetch('/api/bot/status');
        const data = await res.json();
        updateStatusUI(data);
    } catch (e) { console.error('Status poll failed', e); }
}

function updateStatusUI(data) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    const uptime = document.getElementById('uptime-text');
    const btn = document.getElementById('btn-toggle');

    if (text) text.textContent = data.state;

    if (dot) {
        dot.className = 'w-2.5 h-2.5 rounded-full';
        if (data.state === 'RUNNING') dot.classList.add('bg-green-500');
        else if (data.state === 'ERROR') dot.classList.add('bg-red-500');
        else if (data.state === 'STARTING' || data.state === 'STOPPING') dot.classList.add('bg-yellow-500');
        else dot.classList.add('bg-gray-600');
    }

    if (uptime && data.uptime_seconds > 0) {
        const m = Math.floor(data.uptime_seconds / 60);
        const s = Math.floor(data.uptime_seconds % 60);
        uptime.textContent = `${m}m ${s}s`;
        uptime.classList.remove('hidden');
    } else if (uptime) {
        uptime.classList.add('hidden');
    }

    if (btn) {
        if (data.state === 'RUNNING') {
            btn.textContent = 'Stop Bot';
            btn.className = 'ml-3 px-3 py-1 rounded text-xs font-medium bg-red-600 hover:bg-red-500 text-white transition';
        } else if (data.state === 'STARTING' || data.state === 'STOPPING') {
            btn.textContent = data.state + '...';
            btn.disabled = true;
            btn.className = 'ml-3 px-3 py-1 rounded text-xs font-medium bg-gray-700 text-gray-400 cursor-not-allowed transition';
        } else {
            btn.textContent = 'Start Bot';
            btn.disabled = false;
            btn.className = 'ml-3 px-3 py-1 rounded text-xs font-medium bg-indigo-600 hover:bg-indigo-500 text-white transition';
        }
    }
}

// ── Bot toggle ──────────────────────────────────────────────
async function toggleBot() {
    const btn = document.getElementById('btn-toggle');
    const isRunning = btn && btn.textContent.trim() === 'Stop Bot';

    try {
        const url = isRunning ? '/api/bot/stop' : '/api/bot/start';
        const res = await fetch(url, { method: 'POST' });
        if (!res.ok) {
            const err = await res.json();
            alert(err.detail || 'Error');
        }
        // Connect/disconnect WS
        if (!isRunning) setTimeout(connectWS, 1000);
    } catch (e) { alert('Request failed: ' + e.message); }

    pollStatus();
}

// ── WebSocket ───────────────────────────────────────────────
function connectWS() {
    if (ws && ws.readyState <= 1) return;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws/live`);

    ws.onmessage = (evt) => {
        try {
            const msg = JSON.parse(evt.data);
            handleWSMessage(msg);
        } catch (e) { console.error('WS parse error', e); }
    };

    ws.onclose = () => { ws = null; };
}

// ── Throttled price updates ─────────────────────────────────
let _priceUpdateScheduled = false;
let _pricesInitialized = false;

function handleWSMessage(msg) {
    if (msg.type === 'tick') {
        livePrices[msg.symbol] = msg;
        // Batch price updates: max 1 per second
        if (!_priceUpdateScheduled) {
            _priceUpdateScheduled = true;
            setTimeout(() => {
                _priceUpdateScheduled = false;
                updatePricesGrid();
            }, 1000);
        }
    }
    if (msg.type === 'fill') {
        loadDashboardTrades();
    }
    if (msg.type === 'signal') {
        loadDashboardSignals();
    }
}

function updatePricesGrid() {
    const grid = document.getElementById('prices-grid');
    if (!grid) return;
    const symbols = Object.keys(livePrices).sort();
    if (!symbols.length) return;

    // Clear "Waiting for data..." on first update
    if (!_pricesInitialized) {
        grid.innerHTML = '';
        _pricesInitialized = true;
    }

    for (const s of symbols) {
        const t = livePrices[s];
        const price = parseFloat(t.last).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        const elId = 'price-' + s.replace('/', '-');
        let el = document.getElementById(elId);
        if (el) {
            el.querySelector('.price-value').textContent = '$' + price;
        } else {
            const div = document.createElement('div');
            div.id = elId;
            div.className = 'bg-gray-800/50 rounded px-3 py-2';
            div.innerHTML = `<p class="text-xs text-gray-500">${s}</p><p class="price-value text-lg font-mono font-bold">$${price}</p>`;
            grid.appendChild(div);
        }
    }
}

// ── Dashboard polling ───────────────────────────────────────
async function loadPortfolio() {
    try {
        const res = await fetch('/api/portfolio');
        const d = await res.json();
        const fmt = (v) => '$' + parseFloat(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        const pnlClass = (v) => v >= 0 ? 'pnl-positive' : 'pnl-negative';

        const eq = document.getElementById('metric-equity');
        const rp = document.getElementById('metric-realized');
        const up = document.getElementById('metric-unrealized');
        const ex = document.getElementById('metric-exposure');

        if (eq) eq.textContent = fmt(d.equity);
        if (rp) { rp.textContent = fmt(d.realized_pnl); rp.className = 'text-xl font-bold mt-1 ' + pnlClass(d.realized_pnl); }
        if (up) { up.textContent = fmt(d.unrealized_pnl); up.className = 'text-xl font-bold mt-1 ' + pnlClass(d.unrealized_pnl); }
        if (ex) ex.textContent = fmt(d.exposure);

        // Positions table
        const tbody = document.getElementById('positions-body');
        if (tbody) {
            if (d.positions && d.positions.length) {
                tbody.innerHTML = d.positions.map(p => `
                    <tr class="border-b border-gray-800/50">
                        <td class="py-2 px-2">${p.symbol}</td>
                        <td class="py-2 px-2 ${p.side==='buy'?'text-green-400':'text-red-400'}">${p.side.toUpperCase()}</td>
                        <td class="py-2 px-2 text-right">${parseFloat(p.qty).toFixed(6)}</td>
                        <td class="py-2 px-2 text-right">${parseFloat(p.entry_price).toFixed(2)}</td>
                        <td class="py-2 px-2 text-right">${parseFloat(p.current_price).toFixed(2)}</td>
                        <td class="py-2 px-2 text-right ${p.unrealized_pnl>=0?'pnl-positive':'pnl-negative'}">${parseFloat(p.unrealized_pnl).toFixed(2)}</td>
                    </tr>
                `).join('');
            } else {
                tbody.innerHTML = '<tr><td colspan="6" class="text-gray-500 py-4 text-center">No positions</td></tr>';
            }
        }

    } catch (e) { console.error(e); }
}

async function loadDashboardTrades() {
    try {
        const res = await fetch('/api/trades?limit=10');
        const trades = await res.json();
        const tbody = document.getElementById('trades-body');
        if (!tbody || !trades.length) return;
        tbody.innerHTML = trades.map(t => `
            <tr class="border-b border-gray-800/50">
                <td class="py-1 px-1">${t.symbol}</td>
                <td class="py-1 px-1 ${t.side==='buy'?'text-green-400':'text-red-400'}">${t.side.toUpperCase()}</td>
                <td class="py-1 px-1 text-right">${t.quantity.toFixed(4)}</td>
                <td class="py-1 px-1 text-right">${t.price.toFixed(2)}</td>
                <td class="py-1 px-1 text-right text-gray-500">${new Date(t.executed_at).toLocaleTimeString()}</td>
            </tr>
        `).join('');
    } catch (e) { console.error(e); }
}

async function loadDashboardSignals() {
    try {
        const res = await fetch('/api/signals?limit=10');
        const signals = await res.json();
        const tbody = document.getElementById('signals-body');
        if (!tbody || !signals.length) return;
        const dc = {long:'text-green-400',short:'text-red-400',close:'text-yellow-400',hold:'text-gray-500'};
        tbody.innerHTML = signals.map(s => `
            <tr class="border-b border-gray-800/50">
                <td class="py-1 px-1">${s.symbol}</td>
                <td class="py-1 px-1 ${dc[s.direction]||''}">${s.direction.toUpperCase()}</td>
                <td class="py-1 px-1 text-gray-400">${s.strategy}</td>
                <td class="py-1 px-1 text-right">${(s.confidence*100).toFixed(1)}%</td>
                <td class="py-1 px-1 text-right text-gray-500">${new Date(s.created_at).toLocaleTimeString()}</td>
            </tr>
        `).join('');
    } catch (e) { console.error(e); }
}

function startDashboardPolling() {
    loadPortfolio();
    loadDashboardTrades();
    loadDashboardSignals();
    setInterval(loadPortfolio, 30000);
    setInterval(loadDashboardTrades, 30000);
    setInterval(loadDashboardSignals, 30000);
    connectWS();
}

// ── Setup page ──────────────────────────────────────────────
async function loadSetupConfig() {
    try {
        const res = await fetch('/api/config');
        const cfg = await res.json();

        const ki = document.getElementById('key-indicator');
        const si = document.getElementById('secret-indicator');
        if (ki) ki.innerHTML = cfg.api_key_set ? '<span class="text-green-400">Set</span>' : '<span class="text-gray-500">Not set</span>';
        if (si) si.innerHTML = cfg.api_secret_set ? '<span class="text-green-400">Set</span>' : '<span class="text-gray-500">Not set</span>';

        const modeRadio = document.querySelector(`input[name="mode"][value="${cfg.mode}"]`);
        if (modeRadio) modeRadio.checked = true;

        const cap = document.getElementById('capital');
        if (cap) cap.value = cfg.initial_capital;

        // Check pairs
        document.querySelectorAll('.pair-cb').forEach(cb => {
            cb.checked = cfg.pairs.includes(cb.value);
        });
    } catch (e) { console.error(e); }
}

async function saveConfig(evt) {
    evt.preventDefault();
    const payload = {
        api_key: document.getElementById('api-key').value,
        api_secret: document.getElementById('api-secret').value,
        mode: document.querySelector('input[name="mode"]:checked').value,
        initial_capital: parseFloat(document.getElementById('capital').value) || 10000,
        pairs: Array.from(document.querySelectorAll('.pair-cb:checked')).map(cb => cb.value),
    };

    const status = document.getElementById('save-status');
    try {
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (res.ok) {
            if (status) { status.textContent = 'Saved!'; status.className = 'text-sm text-green-400'; }
            loadSetupConfig();
        } else {
            const err = await res.json();
            if (status) { status.textContent = err.detail || 'Error'; status.className = 'text-sm text-red-400'; }
        }
    } catch (e) {
        if (status) { status.textContent = e.message; status.className = 'text-sm text-red-400'; }
    }
    setTimeout(() => { if (status) status.textContent = ''; }, 3000);
}

// ── Init ────────────────────────────────────────────────────
pollStatus();
statusInterval = setInterval(pollStatus, 30000);
