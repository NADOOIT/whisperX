function downloadMetricsCSV() {
    const profileId = document.getElementById('metrics-profile-select').value;
    fetch(`/metrics/${profileId}`)
        .then(r => r.json())
        .then(metrics => {
            if(!metrics.tests || metrics.tests.length === 0) {
                alert('Keine Testdaten vorhanden.');
                return;
            }
            let csv = 'Datei,WER,Ref,Hyp\n';
            metrics.tests.forEach(t => {
                csv += `"${t.file}",${(t.wer*100).toFixed(2)}%,"${t.ref.replace(/"/g,'""')}","${t.hyp.replace(/"/g,'""')}"\n`;
            });
            let blob = new Blob([csv], {type: 'text/csv'});
            let url = URL.createObjectURL(blob);
            let a = document.createElement('a');
            a.href = url;
            a.download = `metrics_${profileId}.csv`;
            document.body.appendChild(a);
            a.click();
            setTimeout(()=>{
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 100);
        });
}
