// Chart.js CDN for simple plotting
// This script is loaded at the bottom of index.html
(function(){
    if(!window.Chart) {
        var script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
        script.onload = function() { window.metricsChartReady = true; };
        document.head.appendChild(script);
    } else {
        window.metricsChartReady = true;
    }
})();

function refreshMetrics() {
    const profileId = document.getElementById('metrics-profile-select').value;
    fetch(`/metrics/${profileId}`)
        .then(r => r.json())
        .then(metrics => {
            let summary = '';
            if(metrics.mean_wer !== null) {
                summary = `<b>Mittlere Word Error Rate (WER):</b> ${(metrics.mean_wer*100).toFixed(2)}%`;
            } else {
                summary = '<span class="text-muted">Noch keine Auswertung verfügbar.</span>';
            }
            document.getElementById('metrics-summary').innerHTML = summary;
            // Tabelle
            let table = '';
            let badTestdata = [];
            let filteredTests = metrics.tests;
            if(document.getElementById('toggle-flagged-filter').classList.contains('active')) {
                filteredTests = metrics.tests.filter(t => t.flagged);
            }
            if(filteredTests && filteredTests.length > 0) {
                table = '<div><button class="btn btn-danger btn-sm mb-2 me-2" id="remove-bad-testdata-btn" style="display:none;" onclick="removeAllBadTestdata()">Alle schlechten Testdaten entfernen (WER > 30%)</button>' +
                        '<button class="btn btn-danger btn-sm mb-2 me-2" id="remove-flagged-testdata-btn" onclick="removeAllFlaggedTestdata()">Alle markierten Testdaten löschen</button>' +
                        '<button class="btn btn-outline-warning btn-sm mb-2" id="toggle-flagged-filter" onclick="toggleFlaggedFilter()">Nur markierte Testdaten anzeigen</button></div>' +
                        '<table class="table table-sm table-striped"><thead><tr><th>Datei</th><th>WER</th><th>Ref</th><th>Hyp</th><th></th></tr></thead><tbody>';
                filteredTests.forEach(t => {
                    let werVal = t.wer*100;
                    let flagged = t.flagged;
                    let flagIcon = flagged ? ' <span title="Zur Korrektur markiert" style="color:gold;">&#9888;</span>' : '';
                    let flagBtn = flagged
                        ? `<button class='btn btn-sm btn-warning me-1' onclick='flagTestdata("${t.file}",false)'>Markierung entfernen</button>`
                        : `<button class='btn btn-sm btn-outline-warning me-1' onclick='flagTestdata("${t.file}",true)'>Zur Korrektur markieren</button>`;
                    let werClass = werVal < 10 ? 'table-success' : (werVal > 30 ? 'table-danger' : '');
                    if(werVal > 30) badTestdata.push(t.file);
                    table += `<tr class="${werClass}"><td>${t.file}${flagIcon}</td><td title="Ref: ${t.ref.replace(/\"/g,'&quot;')}\nHyp: ${t.hyp.replace(/\"/g,'&quot;')}">${werVal.toFixed(2)}%</td><td>${t.ref}</td><td>${t.hyp}</td><td>${flagBtn}<button class='btn btn-sm btn-outline-danger' onclick='removeTestdata("${t.file}")'>Entfernen</button></td></tr>`;
                });
                table += '</tbody></table>';
            }
            // Button für alle schlechten Testdaten anzeigen
            let btn = document.getElementById('remove-bad-testdata-btn');
            if(btn) btn.style.display = (badTestdata.length > 0) ? '' : 'none';
            window.badTestdataList = badTestdata;
            document.getElementById('metrics-table').innerHTML = table;
            // Verlaufsgraf für aktuelle Testdaten
            if(window.metricsChartReady && metrics.tests && metrics.tests.length > 0) {
                let ctx = document.getElementById('metrics-wer-chart').getContext('2d');
                if(window.metricsChartObj) window.metricsChartObj.destroy();
                window.metricsChartObj = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: metrics.tests.map(t => t.file),
                        datasets: [{
                            label: 'WER (%)',
                            data: metrics.tests.map(t => t.wer*100),
                            borderColor: '#0d6efd',
                            backgroundColor: 'rgba(13,110,253,0.08)',
                            tension: 0.2,
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: { legend: { display: false } },
                        scales: { y: { beginAtZero: true, max: 100 } }
                    }
                });
            }
            // Verlaufsgraf für WER-Historie
            fetch(`/metrics_history/${profileId}`)
                .then(r=>r.json())
                .then(hist => {
                    if(hist && hist.length > 1 && window.metricsChartReady) {
                        let ctx2 = document.getElementById('metrics-wer-history-chart').getContext('2d');
                        if(window.metricsHistoryChartObj) window.metricsHistoryChartObj.destroy();
                        window.metricsHistoryChartObj = new Chart(ctx2, {
                            type: 'line',
                            data: {
                                labels: hist.map(h => h.timestamp),
                                datasets: [{
                                    label: 'WER-Verlauf (%)',
                                    data: hist.map(h => h.mean_wer*100),
                                    borderColor: '#dc3545',
                                    backgroundColor: 'rgba(220,53,69,0.07)',
                                    tension: 0.2,
                                }]
                            },
                            options: {
                                responsive: true,
                                plugins: { legend: { display: false } },
                                scales: { y: { beginAtZero: true, max: 100 } }
                            }
                        });
                        document.getElementById('metrics-history-section').style.display = '';
                    } else {
                        document.getElementById('metrics-history-section').style.display = 'none';
                    }
                });
        });
}

document.addEventListener('DOMContentLoaded', function() {
    setTimeout(refreshMetrics, 600);
    document.getElementById('metrics-profile-select').addEventListener('change', refreshMetrics);
});
