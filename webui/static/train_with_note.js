function startTrainingWithNote(speakerId) {
    const note = document.getElementById('train-note').value;
    fetch(`/train/${speakerId}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({note})
    }).then(()=>{
        document.getElementById('train-note').value = '';
        alert('Training gestartet!');
        refreshMetrics();
    });
}
