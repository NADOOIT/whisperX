function downloadMetricsHistoryCSV() {
    const profileId = document.getElementById('metrics-profile-select').value;
    fetch(`/metrics_history/${profileId}`)
        .then(r => r.json())
        .then(hist => {
            if(!hist || hist.length === 0) {
                alert('Keine Verlaufshistorie vorhanden.');
                return;
            }
            let csv = 'Zeitpunkt,Mean_WER\n';
            hist.forEach(h => {
                csv += `"${h.timestamp}",${h.mean_wer !== null ? (h.mean_wer*100).toFixed(2)+"%" : ''}\n`;
            });
            let blob = new Blob([csv], {type: 'text/csv'});
            let url = URL.createObjectURL(blob);
            let a = document.createElement('a');
            a.href = url;
            a.download = `metrics_history_${profileId}.csv`;
            document.body.appendChild(a);
            a.click();
            setTimeout(()=>{
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 100);
        });
}
