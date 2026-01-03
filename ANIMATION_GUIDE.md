# Guida alle Animazioni 3D delle Superfici

## Visualizzazione Interattiva con Plotly

### Utilizzo

1. Esegui lo script di addestramento:

   **Con UV (Raccomandato):**
   ```bash
   uv run python nn_pitagora.py
   ```

   **Con ambiente virtuale tradizionale:**
   ```bash
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   python nn_pitagora.py
   ```

2. Apri `3d_comparison_epochs.html` nel tuo browser

### Funzionalità
- **Superficie blu**: Funzione effettiva f(a,b) = √(a² + b²)
- **Superfici colorate**: Predizioni della rete neurale a diverse epoche con gradiente dal rosso (iniziale) al verde (convergenza)
- **Controlli interattivi**:
  - Clicca e trascina per ruotare
  - Scorri per ingrandire
  - Clicca sugli elementi della leggenda per attivare/disattivare le superfici

## Animazione 3D delle Superfici con Manim

### Installazione

Manim dovrebbe già essere installato se hai seguito le istruzioni di installazione nel README.

**Se non l'hai ancora fatto:**

Con UV:
```bash
uv add manim
```

Con pip:
```bash
pip install manim
```

### Generare l'Animazione

Animazione fluida con morphing che mostra le epoche 1, 3, 10, 30, 50 con camera rotante:

**Con UV (Raccomandato):**
```bash
uv run manim -pqh animate_surface.py EvolvingNNSurfaceSimple
```

**Con ambiente virtuale tradizionale:**
```bash
manim -pqh animate_surface.py EvolvingNNSurfaceSimple
```

### Flag di Comando
- `-p`: Riproduce il video dopo il rendering
- `-q`: Qualità
  - `-ql`: Bassa qualità (480p, veloce)
  - `-qm`: Media qualità (720p)
  - `-qh`: Alta qualità (1080p)
  - `-qk`: Qualità 4K (2160p)

### Output
I video sono salvati in `media/videos/animate_surface/1080p60/`

## Caratteristiche dell'Animazione

### EvolvingNNSurfaceSimple
- **Posizionamento ottimizzato della camera** per massima visibilità della superficie
  - Vista angolata a 60° enfatizza la curvatura della superficie
  - Angolo theta di -110° per prospettiva ottimale
  - Distanza camera più ravvicinata riempie lo schermo
  - Rotazione lenta (velocità 0.06) per facile visualizzazione
- **Superficie blu semi-trasparente**: Funzione target (opacità 0.15)
- **Superficie colorata evolutiva**: Predizione della rete neurale che si trasforma attraverso le epoche (opacità 0.65)
- **Gradiente di colore**: Rosso (epoca 1) → Verde (epoca 50) che mostra il progresso dell'apprendimento
- **Transizioni fluide** tra le epoche 1 → 3 → 10 → 30 → 50 (3 secondi ciascuna)
- **Scala proporzionale** su tutti gli assi per rappresentazione accurata
- **Superfici ad alta risoluzione** (30×30) per aspetto liscio
- **Contatore epoche pulito** (giallo, dimensione 32) - il testo vecchio scompare prima che appaia quello nuovo
- **Superfici grandi** che riempiono la maggior parte dello schermo per chiara visibilità
- Ideale per presentazioni e dimostrazioni

## Come Funziona il Processo di Apprendimento della Rete Neurale

### Epoca 1 (Rosso Scuro)
- Pesi iniziali casuali
- Approssimazione scadente, superficie lontana dal target
- Errore elevato su tutto il dominio

### Epoca 3 (Rosso-Arancione)
- Pesi che iniziano ad aggiustarsi
- Superficie che inizia a prendere forma
- Errore ancora significativo

### Epoca 10 (Arancione-Giallo)
- I pesi hanno appreso il pattern di base
- Superficie che si conforma alla forma corretta
- Errore che diminuisce notevolmente

### Epoca 30 (Giallo-Verde)
- Apprendimento sostanziale completato
- Superficie che segue da vicino il target
- Rimangono deviazioni minori

### Epoca 50 (Verde)
- Rete ben addestrata
- Superficie quasi sovrapposta alla funzione effettiva
- Errore minimo, eccellente generalizzazione

## Personalizzazione

### Epoche Diverse
Modifica `nn_pitagora.py` riga 54:
```python
epochs_to_plot = [1, 3, 10, 30, 50]  # Cambia questi valori
```

L'animazione userà automaticamente le epoche 1, 3, 10, 30, 50 (o quelle che esistono nei dati).

### Scala Generale
Modifica `animate_surface.py` righe 27-28 per cambiare la dimensione della superficie:
```python
xy_scale = 0.6  # Cambia per rendere le superfici più grandi/piccole
z_scale = 0.6   # Dovrebbe corrispondere a xy_scale per rappresentazione proporzionale
```

Nota: Per una rappresentazione accurata, mantieni xy_scale e z_scale uguali. Puoi amplificare l'asse z per un effetto drammatico, ma questo distorcerà la vera forma della funzione.

### Posizione della Camera
Modifica le impostazioni della camera in `animate_surface.py` (righe 34-38):
```python
self.set_camera_orientation(
    phi=60 * DEGREES,     # Angolo verticale (più alto = più dall'alto)
    theta=-110 * DEGREES,  # Angolo orizzontale
    distance=8            # Più basso = più vicino (vista più grande)
)
```

### Velocità di Rotazione della Camera
Modifica la velocità di rotazione in `animate_surface.py` (riga 40):
```python
self.begin_ambient_camera_rotation(rate=0.06)  # Più basso = più lento
```

### Velocità dell'Animazione
Modifica il tempo di transizione in `animate_surface.py` (riga 153):
```python
self.play(Transform(...), run_time=3)  # Regola la durata
```

### Opacità della Superficie
Modifica i valori di opacità in `animate_surface.py`:
```python
# Superficie effettiva (riga 77)
fill_opacity=0.15  # Trasparenza della funzione target

# Superficie rete neurale (riga 121)
fill_opacity=0.65  # Trasparenza della predizione della rete neurale
```

### Risoluzione
Modifica il parametro `resolution` per qualità della superficie vs prestazioni:
```python
resolution=(30, 30)  # Più basso = più veloce, Più alto = più liscio (righe 76, 120)
```

## Suggerimenti per Migliori Risultati

1. **Esegui prima l'addestramento**: Esegui sempre `nn_pitagora.py` prima di creare le animazioni per generare `epoch_surfaces.pkl`

2. **Inizia con bassa qualità**: Testa prima con il flag `-ql` per un'anteprima veloce:
   ```bash
   manim -pql animate_surface.py EvolvingNNSurfaceSimple
   ```

3. **Il posizionamento della camera è ottimizzato**: L'angolo phi di 60° e theta di -110° con distanza=8 enfatizzano la curvatura della superficie

4. **Scala proporzionale**: Tutti gli assi usano scala 0.6 per una rappresentazione accurata della vera forma della funzione

5. **Le superfici riempiono lo schermo**: La scala 0.6 con camera più vicina fa sì che le superfici occupino la maggior parte del frame

6. **La rotazione più lenta aiuta**: La velocità di rotazione 0.06 permette agli spettatori di apprezzare il morphing senza sfocatura da movimento

7. **Le transizioni più lunghe mostrano i dettagli**: I morphing di 3 secondi tra le epoche permettono agli spettatori di vedere la convergenza graduale

8. **Il contrasto è ottimizzato**: Target blu (15% opacità) vs superficie rete neurale colorata (65% opacità) mostra chiaramente il divario

9. **Nessuna sovrapposizione di testo**: L'animazione dissolve correttamente le vecchie etichette delle epoche prima di mostrare quelle nuove

10. **Le etichette gialle delle epoche risaltano**: Il testo giallo ad alto contrasto è visibile contro tutti i colori delle superfici
