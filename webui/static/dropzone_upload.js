// Drag & Drop Testdaten-Upload
const dropzone = document.getElementById('dropzone');
const dropzoneInput = document.getElementById('dropzone-input');
dropzone.addEventListener('click', () => dropzoneInput.click());
dropzone.addEventListener('dragover', e => {
    e.preventDefault();
    dropzone.classList.add('bg-primary','text-white');
});
dropzone.addEventListener('dragleave', e => {
    dropzone.classList.remove('bg-primary','text-white');
});
dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('bg-primary','text-white');
    handleFiles(e.dataTransfer.files);
});
dropzoneInput.addEventListener('change', e => {
    handleFiles(e.target.files);
});

function handleFiles(files) {
    if(!files || files.length === 0) return;
    // Gruppiere nach Basename (ohne Extension)
    let byBase = {};
    for(let file of files) {
        let m = file.name.match(/^(.*?)(\.[^.]+)?$/);
        let base = m ? m[1] : file.name;
        if(!byBase[base]) byBase[base] = {};
        if(file.type.startsWith('audio') || /\.(wav|mp3|flac|ogg|m4a)$/i.test(file.name)) {
            byBase[base].audio = file;
        } else if(/\.txt$/i.test(file.name)) {
            byBase[base].transcript = file;
        }
    }
    // Für jedes Paar Upload starten
    let pairs = Object.values(byBase).filter(p => p.audio && p.transcript);
    if(pairs.length === 0) {
        alert('Bitte jeweils Audio und zugehöriges Transkript (gleicher Name) auswählen!');
        return;
    }
    uploadPairs(pairs);
}

function uploadPairs(pairs) {
    let idx = 0;
    function next() {
        if(idx >= pairs.length) { location.reload(); return; }
        let form = new FormData();
        form.append('audio', pairs[idx].audio);
        form.append('transcript', pairs[idx].transcript);
        fetch('/upload_testdata', {method:'POST', body:form})
            .then(()=>{ idx++; next(); });
    }
    next();
}
