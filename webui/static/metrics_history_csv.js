function downloadMetricsHistoryCSV() {
    const profileId = document.getElementById('metrics-profile-select').value;
    fetch(`/metrics_history/${profileId}`)
        .then(r => r.json())
        .then(history => {
            if(!history || history.length === 0) {
                alert('Kein Verlauf vorhanden.');
                return;
            }
            let csv = 'Zeit,WER,Notiz\n';
            history.forEach(h => {
                const date = new Date(h.timestamp*1000).toLocaleString();
                const note = h.note ? h.note.replace(/"/g,'""') : '';
                csv += `"${date}",${(h.wer*100).toFixed(2)}%,"${note}"\n`;
            });
            let blob = new Blob([csv], {type: 'text/csv'});
            let url = URL.createObjectURL(blob);
            let a = document.createElement('a');
            a.href = url;
            a.download = `wer_history_${profileId}.csv`;
            document.body.appendChild(a);
            a.click();
            setTimeout(()=>{
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 100);
        });
}
