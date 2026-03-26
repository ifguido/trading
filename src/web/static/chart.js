/* PnL Chart using Chart.js */
let pnlChart = null;
const pnlData = { labels: [], values: [] };

function initPnlChart() {
    const ctx = document.getElementById('pnl-chart');
    if (!ctx) return;
    pnlChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: pnlData.labels,
            datasets: [{
                label: 'Equity',
                data: pnlData.values,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99,102,241,0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { display: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b7280', maxTicksLimit: 10 } },
                y: { display: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b7280' } }
            },
            plugins: { legend: { display: false } },
            animation: { duration: 0 }
        }
    });
}

function updatePnlChart(equity) {
    if (!pnlChart) initPnlChart();
    if (!pnlChart) return;
    const now = new Date().toLocaleTimeString();
    pnlData.labels.push(now);
    pnlData.values.push(equity);
    // Keep last 100 points
    if (pnlData.labels.length > 100) {
        pnlData.labels.shift();
        pnlData.values.shift();
    }
    pnlChart.update();
}

// Init on load
document.addEventListener('DOMContentLoaded', initPnlChart);
