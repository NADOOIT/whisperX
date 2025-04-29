function toggleFlaggedFilter() {
    let btn = document.getElementById('toggle-flagged-filter');
    btn.classList.toggle('active');
    if(btn.classList.contains('active')) {
        btn.classList.remove('btn-outline-warning');
        btn.classList.add('btn-warning');
        btn.textContent = 'Alle Testdaten anzeigen';
    } else {
        btn.classList.remove('btn-warning');
        btn.classList.add('btn-outline-warning');
        btn.textContent = 'Nur markierte Testdaten anzeigen';
    }
    refreshMetrics();
}

function flagTestdata(file, flagged) {
    let form = new FormData();
    form.append('file', file);
    form.append('flagged', flagged ? '1' : '0');
    fetch('/flag_testdata', {method:'POST', body:form}).then(()=>{
        setTimeout(refreshMetrics, 300);
    });
}
