# Implementazione del Teorema di Pitagora con Rete Neurale

Questo progetto implementa una rete neurale feedforward per apprendere il teorema di Pitagora: f(a,b) = √(a² + b²)

Utilizza TensorFlow/Keras per l'addestramento, Plotly per la visualizzazione 3D interattiva e Manim per animazioni professionali che mostrano il processo di apprendimento.

![Animazione Pedagogica](TriangoliRettangoli%20-%20Copia_processed_by_imagy.gif)

*Visualizzazione animata del processo di apprendimento: la rete neurale impara a predire l'ipotenusa di triangoli rettangoli attraverso le epoche di addestramento (1, 10, 75).*

## Architettura
- **Strato di Input**: 2 neuroni (a, b)
- **Strato Nascosto**: Neuroni configurabili con attivazione ReLU (default: 30)
- **Strato di Output**: 1 neurone con attivazione lineare

La rete si adatta dinamicamente - modificando la dimensione dello strato nascosto in `nn_pitagora.py`, le animazioni si adatteranno automaticamente.

## Installazione

### 1. Creare l'Ambiente Virtuale
```bash
python3 -m venv .venv
source .venv/bin/activate  # Su Windows: .venv\Scripts\activate
```

### 2. Installare le Librerie Richieste
```bash
pip install numpy matplotlib tensorflow plotly manim
```

**Pacchetti richiesti:**
- `numpy` - Calcolo numerico e generazione dati
- `matplotlib` - Visualizzazione della loss durante l'addestramento
- `tensorflow` (include `keras`) - Framework per l'addestramento della rete neurale
- `plotly` - Grafici 3D interattivi
- `manim` - Motore di animazione matematica (per la generazione di video)

**Dipendenze di sistema per Manim:**
- Distribuzione LaTeX (TeX Live o MiKTeX)
- FFmpeg
- Cairo

Su Ubuntu/Debian:
```bash
sudo apt-get install texlive texlive-latex-extra ffmpeg libcairo2-dev
```

Su macOS:
```bash
brew install --cask mactex
brew install ffmpeg cairo
```

### 3. Verificare l'Installazione
```bash
python -c "import numpy, matplotlib, keras, plotly, manim; print('Tutti i pacchetti installati con successo!')"
```

## File

### nn_pitagora.py
Script principale di addestramento che:
- Genera campioni di addestramento (configurabile, default: 5.000)
- Addestra la rete neurale (epoche configurabili, default: 75)
- Salva i modelli completi a epoche specifiche nella directory `saved_models/`
- Traccia la loss di addestramento/validazione e MAE
- Testa le predizioni su input di esempio
- Salva i pesi in `model_weights.pkl` per le animazioni
- Genera dati delle superfici per epoca per visualizzazioni 3D

**Funzioni chiave:**

- `genera_dati_addestramento()`: Genera i dati di addestramento
- `costruisci_modello()`: Crea l'architettura della rete neurale
- `addestra_modello()`: Addestra e salva i modelli a epoche specifiche
- `load_saved_model()`: Carica un modello da un'epoca specifica
- `salva_pesi_modello()`: Esporta i pesi in formato compatibile con Manim

### animate_weights.py
Script di animazione Manim con tre scene:
1. **NeuralNetworkWeights**: Visualizza la struttura della rete con connessioni colorate e valori dei pesi su ogni freccia
2. **WeightHeatmap**: Mostra mappe di calore delle matrici dei pesi
3. **SampleCalculation**: Dimostrazione passo-passo del calcolo per l'input (1, 2)

### animate_surface.py
Animazione 3D Manim che mostra l'evoluzione dell'approssimazione della superficie della rete neurale attraverso le epoche di addestramento. Include transizioni fluide di morphing, camera rotante e gradiente di colore dal rosso (epoca 1) al verde (epoca 50).

### right_triangles.py
Scena Manim che dimostra il teorema di Pitagora con predizioni della rete neurale:

- **TriangoliRettangoli**: Scena principale che mostra:
  - Pannello sinistro: 4 triangoli rettangoli di input con triple pitagoriche note (3-4-5, 5-12-13, 8-15-17, 7-24-25)
  - Pannello centrale: Visualizzazione animata della rete neurale feed-forward con 2 input, N neuroni nascosti e 1 output
  - Pannello destro: Triangoli di output con predizioni della rete neurale sovrapposti ai triangoli di input
  - Mostra l'accuratezza delle predizioni attraverso le epoche (1, 10, 75) con analisi degli errori
  - La visualizzazione della rete estrae automaticamente l'architettura dai modelli salvati
  - Frecce con aspetto ottimizzato (linee sottili, punte piccole, neuroni che si toccano)

## Utilizzo

### 1. Addestrare il Modello
```bash
python nn_pitagora.py
```

Questo:
- Addestra la rete neurale
- Salva `training_loss.png` con le curve di loss
- Salva `model_weights.pkl` per le animazioni
- Visualizza le predizioni di test

### 2. Generare le Animazioni (richiede Manim)

Installare Manim se necessario:
```bash
pip install manim
```

Renderizzare la visualizzazione della rete:
```bash
manim -pql animate_weights.py NeuralNetworkWeights
```

Renderizzare la mappa di calore dei pesi:
```bash
manim -pql animate_weights.py WeightHeatmap
```

Renderizzare il calcolo di esempio per (1, 2):
```bash
manim -pql animate_weights.py SampleCalculation
```

Renderizzare tutte le scene contemporaneamente:
```bash
manim -pql animate_weights.py NeuralNetworkWeights WeightHeatmap SampleCalculation
```

Renderizzare la scena dei triangoli rettangoli:
```bash
manim -pql right_triangles.py TriangoliRettangoli
```

Opzioni di qualità:
- `-ql`: Bassa qualità (480p) - veloce
- `-qm`: Media qualità (720p)
- `-qh`: Alta qualità (1080p)
- `-qk`: Qualità 4K
- `-p`: Anteprima dopo il rendering

## Risultati Attesi

Il modello dovrebbe raggiungere un errore molto basso (< 1%) nella predizione del teorema di Pitagora dopo l'addestramento.

Esempi di predizioni:
- f(3, 4) ≈ 5.0
- f(5, 12) ≈ 13.0
- f(6, 8) ≈ 10.0

## Dettagli delle Visualizzazioni

### Animazione della Rete
- **Connessioni rosse**: Pesi positivi
- **Connessioni blu**: Pesi negativi
- **Spessore della linea**: Magnitudine del peso
- **Valori dei pesi**: Visualizzati su ogni freccia di connessione
- Mostra automaticamente tutti i neuroni se ≤15, altrimenti mostra un sottoinsieme
- Si adatta dinamicamente a qualsiasi dimensione della rete

### Mappa di Calore dei Pesi
- **Blu**: Valori minimi dei pesi
- **Rosso**: Valori massimi dei pesi
- W1: Strato Input→Nascosto (matrice 2×N)
- W2: Strato Nascosto→Output (matrice N×1 mostrata in layout a griglia)
- Scala automaticamente il layout della griglia in base alla dimensione della rete

### Calcolo di Esempio
- Dimostra il passaggio in avanti completo per l'input (1, 2)
- **Passo 1**: Mostra il vettore di input [1, 2]
- **Passo 2**: Calcola le attivazioni dello strato nascosto
  - Mostra il calcolo della somma pesata per i primi 3 neuroni
  - Applica la funzione di attivazione ReLU
  - Codificato a colori: VERDE per neuroni attivi, ROSSO per neuroni soppressi
- **Passo 3**: Calcola l'output finale
  - Mostra la somma pesata delle attivazioni nascoste
  - Visualizza la predizione finale
- **Confronto**: Mostra il valore predetto vs il valore reale con percentuale di errore
- Output atteso: √5 ≈ 2.236
