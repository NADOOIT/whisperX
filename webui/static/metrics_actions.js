function removeAllBadTestdata() {
    if(!window.badTestdataList || window.badTestdataList.length === 0) return;
    if(!confirm('Alle schlechten Testdaten (WER > 30%) wirklich entfernen?')) return;
    let files = [...window.badTestdataList];
    let removed = 0;
    function removeNext() {
        if(files.length === 0) {
            refreshMetrics();
            setTimeout(()=>alert('Alle schlechten Testdaten wurden entfernt.'), 300);
            return;
        }
        let file = files.shift();
        let form = new FormData();
        form.append('file', file);
        fetch('/delete_testdata', {method:'POST', body:form}).then(()=>{
            removed++;
            removeNext();
        });
    }
    removeNext();
}

function removeTestdata(file) {
    if(!confirm('Testdaten wirklich entfernen?')) return;
    let form = new FormData();
    form.append('file', file);
    fetch('/delete_testdata', {method:'POST', body:form}).then(()=>{
        setTimeout(refreshMetrics, 300);
    });
}
