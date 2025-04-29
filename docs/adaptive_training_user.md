# 🗣️ Adaptive Sprecher-Modelle – Endnutzer-Dokumentation

## Was ist das?
Mit WhisperX kannst du für jede Person ein individuelles Sprachmodell anlegen, das sich kontinuierlich verbessert. Dank der neuen grafischen Oberfläche kannst du Trainingsdaten besonders einfach verwalten und dein Modell mit wenigen Klicks trainieren.

---

## Neue GUI-Funktion: Sprecher-Training per Mausklick

### Überblick
- **Trainingsdaten-Upload:** Füge für jeden Sprecher beliebig viele Audiodateien per Drag & Drop oder Dateiauswahl hinzu.
- **Transkripte optional:** Du kannst (musst aber nicht!) zu jeder Audiodatei ein Transkript (Textdatei) hinzufügen.
- **Automatisches Training:** Nach dem Hinzufügen neuer Dateien startet das Training automatisch – oder du startest es manuell per Button.
- **Fortschrittsanzeige:** Du siehst jederzeit, wie weit das Training ist und ob ein LoRA-Adapter für den Sprecher existiert.
- **Adapter-Management:** Du kannst Adapter löschen, neu trainieren oder den Status einsehen.

---

## Schritt-für-Schritt-Anleitung

### 1. WhisperX Launchpad starten
Starte die grafische Oberfläche über das Terminal:
```bash
python -m launchpad.app
```

### 2. Sprecher-Profil auswählen oder anlegen
- Wähle im Bereich „Speaker Management“ ein bestehendes Profil aus oder erstelle ein neues.

### 3. Audiodateien hinzufügen
- Klicke auf „Audiodateien hinzufügen“ oder ziehe Dateien direkt ins Fenster.
- Optional: Füge zu jeder Audiodatei ein Transkript hinzu (gleichnamige .txt-Datei).
- Die Dateien erscheinen in der Übersicht.

### 4. Training starten
- Nach dem Hinzufügen startet das Training automatisch **oder** du klickst auf „Training starten“.
- Der Fortschritt wird im Fenster angezeigt (z.B. Balken, Log-Ausgabe).
- Nach Abschluss siehst du, wann und mit welchen Daten zuletzt trainiert wurde.

### 5. Adapter-Management
- Du kannst den LoRA-Adapter für den Sprecher löschen oder neu trainieren.
- Der aktuelle Status (z.B. „Adapter vorhanden“, „Training läuft“) wird angezeigt.

### 6. Transkribieren mit deinem Modell
- Wähle beim Transkribieren dein Profil aus – WhisperX nutzt automatisch den trainierten Adapter für beste Ergebnisse.

---

## Tipps für beste Ergebnisse
- Nutze möglichst viele und unterschiedliche Aufnahmen deiner Stimme (verschiedene Geräte, Räume, Situationen).
- Auch ohne Transkripte kannst du trainieren – das System nutzt dann Self-Supervision.
- Füge regelmäßig neue Daten hinzu, um das Modell aktuell zu halten.

---

## Kontrolldaten & Erfolgskontrolle

Um sicherzustellen, dass das Training deines Sprecher-Modells tatsächlich zu besseren Ergebnissen führt, solltest du **Kontrolldaten** (Testdaten) verwenden:

### Was sind Kontrolldaten?
- Das sind Audiodateien, die **nicht** zum Training verwendet werden, sondern ausschließlich zur Überprüfung der Modellqualität dienen.
- Sie sollten möglichst realistische, aber neue Situationen enthalten (z.B. andere Aufnahmen, schwierige Passagen, verschiedene Mikrofone).

### So nutzt du Kontrolldaten im GUI
1. **Kontrolldateien hinzufügen:**
    - Im Trainingsbereich kannst du Audiodateien explizit als „Testdaten“ markieren oder in einen eigenen Bereich hochladen.
2. **Automatische Auswertung:**
    - Nach jedem Training wird das Modell automatisch auf den Kontrolldaten getestet.
    - Das GUI zeigt dir Metriken wie **Word Error Rate (WER)**, Vergleich Vorher/Nachher und ggf. eine Visualisierung der Verbesserungen.
3. **Erfolg prüfen:**
    - Du siehst sofort, ob das neue Modell wirklich besser ist als das alte – und kannst gezielt nachbessern.

### Tipps für gute Kontrolldaten
- Wähle Aufnahmen, die dem echten Einsatz möglichst ähnlich sind (z.B. Meetings, Telefonate, verschiedene Umgebungen).
- Nutze Kontrolldaten, die das Modell **noch nicht kennt** (keine Überschneidung mit Trainingsdaten).
- Halte die Testmenge klein, aber repräsentativ (z.B. 1-5 Minuten).

---
- Überwache den Trainingsfortschritt und wiederhole das Training bei Bedarf.

---

## Häufige Fragen (FAQ)

**Kann ich auch ohne Transkripte trainieren?**  
Ja! Du kannst einfach nur Audiodateien hinzufügen. Transkripte sind optional, verbessern aber das Training.

**Wie viele Dateien sollte ich verwenden?**  
Je mehr, desto besser. Schon mit wenigen Minuten kannst du starten, aber mehr Vielfalt bringt robustere Ergebnisse.

**Wie sehe ich, ob mein Adapter aktiv ist?**  
Im GUI siehst du für jedes Profil den Adapter-Status und das letzte Trainingsdatum.

**Kann ich das Training abbrechen?**  
Ja, über den Button „Training abbrechen“ im Fortschrittsbereich.

**Kann ich mehrere Sprecher-Modelle parallel verwalten?**  
Ja, für jedes Profil wird ein eigener Adapter erstellt und verwaltet.

Weitere Fragen? Siehe [FAQ](./faq.md) oder stelle sie im GitHub Issue Tracker.
