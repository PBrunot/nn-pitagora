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

### Script di Addestramento

#### nn_pitagora.py
Script principale completo con tutte le funzioni per Manim che:
- Genera campioni di addestramento (configurabile, default: 5.000, range: -5 a 5)
- Addestra la rete neurale con SGD optimizer (epoche configurabili, default: 75)
- Architettura configurabile (default: 10 neuroni nascosti, attivazione tanh)
- Salva i modelli completi a epoche specifiche nella directory `saved_models/`
- Traccia la loss di addestramento/validazione e MAE
- Testa le predizioni su input di esempio (inclusi valori negativi)
- Salva i pesi in `model_weights.pkl` per le animazioni
- Genera visualizzazioni 3D interattive con Plotly
- Genera dati delle superfici per epoca per visualizzazioni 3D
- Crea grafici degli errori con scala logaritmica

**Funzioni chiave:**

- `genera_dati_addestramento()`: Genera i dati di addestramento
- `costruisci_modello()`: Crea l'architettura della rete neurale con SGD
- `addestra_modello()`: Addestra e salva i modelli a epoche specifiche
- `load_saved_model()`: Carica un modello da un'epoca specifica (con gestione errori dettagliata)
- `crea_modello_epoca_zero()`: Crea modello con pesi casuali per l'epoca 0
- `salva_pesi_modello()`: Esporta i pesi in formato compatibile con Manim
- `crea_visualizzazione_3d()`: Crea visualizzazione 3D Plotly confrontando funzione reale con predizioni NN
- `crea_visualizzazione_errori_3d()`: Crea visualizzazione 3D degli errori con scala logaritmica
- `salva_dati_epoca_per_animazione()`: Salva dati superficie per animazioni Manim

#### pitagora.py
Versione semplificata completa per la presentazione (corrisponde al codice mostrato nelle slide):
- 15 neuroni nascosti, attivazione ReLU
- 2.000 campioni di addestramento (range: 0-15)
- 250 epoche con optimizer Adam
- Salva tutti i modelli epoch-by-epoch in `saved_models/`
- Callback personalizzato `SalvaAdOgniEpoca` per salvare modelli automaticamente
- Funzione `print_model_weights()` per visualizzare tutti i pesi e bias in formato tabella
- Genera diagramma del modello (`modello.png`)
- Genera grafico andamento allenamento con scala logaritmica
- Funzione `Test_Pitagora()` per testare predizioni su esempi (3-4, 6-8)

#### pitagora2.py
Versione minimale educativa (più semplice per dimostrazioni interattive):
- 15 neuroni nascosti, attivazione ReLU
- 2.000 campioni di addestramento (range: 0-15)
- 100 epoche con optimizer Adam (meno epoche per tempi più veloci)
- Callback personalizzato `StampaMAECallback` per visualizzare MAE ad ogni epoca
- Ciclo interattivo `while True` per calcolare ipotenuse su input utente
- Mostra errore di calcolo e percentuale di errore per ogni predizione
- Versione compatta ideale per demo dal vivo

### Script di Visualizzazione Manim

#### visualizza_parametri.py
Script Manim che mostra l'evoluzione dei parametri durante l'addestramento:
- **VisualizzaParametri**: Scena principale che visualizza:
  - Tabella dei pesi e bias per tutte le epoche (1, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250)
  - Colori dinamici dal blu (valori bassi) al rosso (valori alti)
  - Grafico MAE in tempo reale che mostra il miglioramento della rete
  - Esempio di predizione per cateti (3, 4) ad ogni epoca
  - Transizioni fluide tra le epoche con aggiornamento dei valori
- Carica modelli salvati da `saved_models/model_epoch_XXX.keras`
- Richiede che `pitagora.py` sia stato eseguito per generare i modelli

#### animate_weights.py
Script di animazione Manim con tre scene:
1. **PesiReteNeurale**: Visualizza la struttura della rete con connessioni colorate e valori dei pesi su ogni freccia
2. **HeatmapPesi**: Mostra mappe di calore delle matrici dei pesi
3. **CalcoloEsempio**: Dimostrazione passo-passo del calcolo per l'input (1, 2)

#### animate_surface.py
Animazione 3D Manim che mostra l'evoluzione dell'approssimazione della superficie della rete neurale attraverso le epoche di addestramento:
- **SuperficieNNEvoluzioneSemplice**: Scena principale che mostra:
  - Superficie blu target (funzione reale √(a² + b²))
  - Superficie NN che si evolve attraverso le epoche (1, 3, 10, 30, 75)
  - Gradiente di colore dal rosso (epoca iniziale) al verde (converge)
  - Transizioni fluide di morphing (3 secondi per transizione)
  - Camera rotante lenta (rate 0.06) per visualizzazione ottimale
  - Posizionamento camera ottimizzato (phi=60°, theta=-110°, distance=8)
  - Scala proporzionale su tutti gli assi per rappresentazione accurata
- Carica dati da `superfici_epoche.pkl` generato da `nn_pitagora.py`

#### right_triangles.py
Scena Manim che dimostra il teorema di Pitagora con predizioni della rete neurale:

- **TriangoliRettangoli**: Scena principale che mostra:
  - Pannello sinistro: 4 triangoli rettangoli di input con triple pitagoriche note (3-4-5, 5-12-13, 8-15-17, 7-24-25)
  - Pannello centrale: Visualizzazione animata della rete neurale feed-forward con 2 input, N neuroni nascosti e 1 output
  - Pannello destro: Triangoli di output con predizioni della rete neurale sovrapposti ai triangoli di input
  - Mostra l'accuratezza delle predizioni attraverso le epoche (0, 1, 10, 75) con analisi degli errori
  - La visualizzazione della rete estrae automaticamente l'architettura dai modelli salvati usando `load_saved_model()`
  - Animazione del flusso dati attraverso la rete con pesi visualizzati sulle frecce
  - Frecce con larghezza proporzionale ai pesi e colore basato sui bias
  - Fase finale mostra le matrici complete di pesi e bias del modello addestrato
  - Calcolo e visualizzazione degli errori assoluti e percentuali per ogni triangolo
  - Frecce con aspetto ottimizzato (linee sottili, punte piccole, neuroni che si toccano)
- Richiede modelli salvati in `saved_models/` dalle epoche 0, 1, 10, 75

## Utilizzo

### 1. Addestrare il Modello

Per la versione completa con tutte le funzionalità:
```bash
python nn_pitagora.py
```

Per la versione della presentazione (250 epoche, tabella pesi, diagramma):
```bash
python pitagora.py
```

Per demo interattive rapide (100 epoche, ciclo input utente):
```bash
python pitagora2.py
```

Lo script genera:
- `saved_models/model_epoch_XXX.keras` - Modelli completi per ogni epoca
- `training_loss.png` / `perdita_addestramento.png` - Curve di loss e MAE
- `model_weights.pkl` / `pesi_modello.pkl` - Pesi per animazioni
- `superfici_epoche.pkl` / `epoch_surfaces.pkl` - Dati superficie per animazioni 3D
- `confronto_3d_epoche.html` / `3d_comparison_epochs.html` - Visualizzazione interattiva (solo `nn_pitagora.py`)
- `errori_3d_epoche.html` - Visualizzazione errori 3D (solo `nn_pitagora.py`)
- `modello.png` - Diagramma architettura (solo `pitagora.py`)

### 2. Generare le Animazioni (richiede Manim)

Installare Manim se necessario:
```bash
pip install manim
```

**Visualizzazione evoluzione parametri** (richiede esecuzione di `pitagora.py`):
```bash
manim -pql visualizza_parametri.py VisualizzaParametri
```

**Visualizzazione struttura rete con pesi**:
```bash
manim -pql animate_weights.py PesiReteNeurale
```

**Mappa di calore dei pesi**:
```bash
manim -pql animate_weights.py HeatmapPesi
```

**Calcolo di esempio passo-passo per (1, 2)**:
```bash
manim -pql animate_weights.py CalcoloEsempio
```

**Animazione 3D evoluzione superficie**:
```bash
manim -pqh animate_surface.py SuperficieNNEvoluzioneSemplice
```

**Dimostrazione triangoli rettangoli con predizioni NN**:
```bash
manim -pql right_triangles.py TriangoliRettangoli
```

Opzioni di qualità:
- `-ql`: Bassa qualità (480p) - veloce
- `-qm`: Media qualità (720p)
- `-qh`: Alta qualità (1080p) - raccomandato per animazioni 3D
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
