function removeAllFlaggedTestdata() {
    if(!confirm('Alle markierten Testdaten wirklich löschen?')) return;
    const profileId = document.getElementById('metrics-profile-select').value;
    fetch(`/metrics/${profileId}`)
        .then(r => r.json())
        .then(metrics => {
            if(!metrics.tests) return;
            let flagged = metrics.tests.filter(t => t.flagged).map(t => t.file);
            if(flagged.length === 0) {
                alert('Keine markierten Testdaten vorhanden.');
                return;
            }
            function removeNext() {
                if(flagged.length === 0) {
                    refreshMetrics();
                    setTimeout(()=>alert('Alle markierten Testdaten wurden gelöscht.'), 300);
                    return;
                }
                let file = flagged.shift();
                let form = new FormData();
                form.append('file', file);
                fetch('/delete_testdata', {method:'POST', body:form}).then(removeNext);
            }
            removeNext();
        });
}
