function downloadFlaggedTestdataZIP() {
    const profileId = document.getElementById('metrics-profile-select').value;
    window.open(`/download_flagged_zip/${profileId}`, '_blank');
}
