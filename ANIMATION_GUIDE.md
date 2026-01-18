# 3D Surface Animation Guide

## Interactive Plotly Visualization

Run training to generate the interactive HTML visualization:
```bash
python nn_pitagora.py
open 3d_comparison_epochs.html
```

Features:
- **Blue surface**: Actual function f(a,b) = √(a² + b²)
- **Colored surfaces**: NN predictions (red → green = epoch progression)
- **Interactive**: Click-drag to rotate, scroll to zoom

## Manim 3D Surface Animation

### Genera animazione

```bash
manim -pqh animate_surface.py SuperficieNNEvoluzioneSemplice
```

### Output

Video salvato in `media/videos/animate_surface/1080p60/`

### Features

- Camera ottimizzata (60° phi, -110° theta, distance 8)
- Superficie blu semi-trasparente: funzione target
- Superficie colorata: predizione NN che evolve (rosso → verde)
- Transizioni fluide tra epoche (1 → 3 → 10 → 30 → 75)
- Rotazione lenta (0.06) per visualizzazione ottimale
- Scala proporzionale per rappresentazione accurata

### Quality flags

- `-ql`: 480p (veloce)
- `-qh`: 1080p (raccomandato)
- `-qk`: 4K
- `-p`: Anteprima dopo render

## Customization

### Epoche da visualizzare
Modifica `animate_surface.py` linea 22:
```python
epoche_target = [1, 3, 10, 30, 75]
```

### Scala superficie
Modifica linee 27-28:
```python
scala_xy = 0.6  # Dimensione superficie
scala_z = 0.6   # Mantieni uguale a xy per proporzionalità accurata
```

### Posizione camera
Linee 34-38:
```python
self.set_camera_orientation(
    phi=60 * DEGREES,      # Angolo verticale
    theta=-110 * DEGREES,  # Angolo orizzontale
    distance=8             # Distanza (più basso = più vicino)
)
```

### Velocità rotazione
Linea 39:
```python
self.begin_ambient_camera_rotation(rate=0.06)
```

### Durata transizioni
Linea 154:
```python
run_time=3  # Secondi per transizione tra epoche
```

### Opacità superfici
```python
# Linea 80 - Superficie target
fill_opacity=1.0

# Linea 125 - Superficie NN
fill_opacity=0.75
```

### Risoluzione
Linee 79, 124:
```python
resolution=(30, 30)  # Più alto = più liscio ma più lento
```
