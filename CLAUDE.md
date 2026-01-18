# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Neural Network Pythagorean Theorem Project

## Architecture Overview

This project implements a neural network to learn the Pythagorean theorem: f(a,b) = √(a² + b²). The project consists of:

- **Core neural network implementation** (`nn_pitagora.py`): Training, data generation, model management
- **Three Manim animation scripts**: Visualizing network weights, 3D surfaces, and practical examples
- **Dependency management**: Uses `uv` package manager (uv.lock file)
- **Saved model artifacts**: Complete epoch-by-epoch training snapshots in `saved_models/`
- **Visualization outputs**: Interactive HTML plots and static PNG charts

### Key Technical Decisions
- Network architecture: 2 → N → 1 (default N=50, configurable)
- Default activation: ReLU for hidden layer, linear for output
- Training range: a,b ∈ [-5, 5], target function: √(a² + b²)
- Model persistence: Full Keras models saved at specific epochs for reproducibility
- Animation data: Preprocessed surfaces and weights stored in pickle files

## Common Commands

### Training and Basic Usage
```bash
python nn_pitagora.py                          # Complete version (5000 samples, SGD, -5 to 5, Plotly 3D)
python pitagora.py                             # Presentation version (2000 samples, Adam, 0-15, 250 epochs, weight table)
python pitagora2.py                            # Interactive demo version (100 epochs, while loop)
manim -pql visualizza_parametri.py VisualizzaParametri  # Parameter evolution animation (requires pitagora.py output)
manim -pql right_triangles.py TriangoliRettangoli       # Triangle demonstration with epochs 0,1,10,75
manim -pqh animate_surface.py SuperficieNNEvoluzioneSemplice  # 3D surface morphing
manim -pql animate_weights.py PesiReteNeurale          # Network weights visualization
manim -pql animate_weights.py HeatmapPesi              # Weight heatmaps
manim -pql animate_weights.py CalcoloEsempio           # Step-by-step calculation
```

### Development Workflow
```bash
# Environment setup (uses uv)
uv sync                             # Install dependencies from uv.lock

# Training with different parameters
python -c "from nn_pitagora import *; model = costruisci_modello(unita_nascoste=30)"

# Generate specific epoch models
python -c "from nn_pitagora import crea_modello_epoca_zero; crea_modello_epoca_zero()"

# Render all animation scenes
manim -pql animate_weights.py       # All weight visualization scenes
manim -pql right_triangles.py TriangoliRettangoli  # Specific scene
```

## Key Functions for Manim Integration

The following functions from `nn_pitagora.py` are specifically designed for use in Manim Python animation files:

### Data Generation
- **`genera_dati_addestramento(n_campioni=5000, range_min=-5, range_max=5)`**
  - Generates training data for the Pythagorean theorem neural network
  - Returns `(X_addestramento, y_addestramento)` tuple
  - Used to create consistent datasets for animations

### Model Creation and Training
- **`costruisci_modello(unita_nascoste=50, attivazione='tanh', tasso_apprendimento=0.001)`**
  - Creates and compiles a neural network model with SGD optimizer
  - Returns compiled Keras Sequential model
  - Default uses 'tanh' activation (configurable)
  - Uses SGD optimizer with configurable learning rate

- **`addestra_modello(modello, X_addestramento, y_addestramento, epoche=75, epoche_da_salvare=None)`**
  - Trains the model and saves complete models at specified epochs
  - Returns `(cronologia, callback_salvataggio)` tuple
  - Automatically saves models to `saved_models/model_epoch_XXX.keras`
  - Uses validation split of 20% by default

### Weight Management
- **`salva_pesi_modello(modello, nome_file='pesi_modello.pkl')`**
  - Saves final model weights in a structured format for Manim
  - Exports weights as: `{'W1': weights[0], 'b1': weights[1], 'W2': weights[2], 'b2': weights[3]}`

### Visualization Functions
- **`ottieni_predizioni_per_epoca(pesi, X_input, forma_output, unita_nascoste=50, attivazione='tanh')`**
  - Creates temporary model from saved weights and generates predictions
  - Essential for animating neural network learning progress
  - Used to show how predictions change across training epochs

- **`crea_visualizzazione_3d(callback_salvataggio, epoche_da_tracciare, risoluzione=50, unita_nascoste=50, attivazione='tanh')`**
  - Creates 3D surface comparison between actual function and NN predictions
  - Returns `(fig, range_a, range_b, Z_reale, griglia_X)` for further processing
  - Generates interactive Plotly visualization saved as HTML
  - Uses consistent color gradient (red to green) across epochs

- **`salva_dati_epoca_per_animazione(callback_salvataggio, epoche_da_tracciare, range_a, range_b, Z_reale, griglia_X, nome_file='superfici_epoche.pkl')`**
  - Saves complete epoch surface data specifically for Manim animations
  - Creates pickle file with all necessary data for frame-by-frame animation
  - Data structure: `{'a_range': range_a, 'b_range': range_b, 'Z_actual': Z_reale, 'epochs': {epoch: Z_predicted}}`

### Utility Functions
- **`ottieni_colore_epoca(indice_epoca, epoche_totali)`**
  - Generates color gradient from red to green for epoch progression
  - Useful for consistent color schemes in Manim animations

- **`imposta_seme_casuale(seme=42)`**
  - Ensures reproducible results for consistent animations

- **`crea_modello_epoca_zero()`**
  - Creates epoch 0 model with random weights for animation initialization
  - Essential for Manim scenes that need baseline untrained model

## Model Loading

- **`carica_modello_salvato(numero_epoca, directory_modelli="saved_models")`**
  - Loads a complete Keras model from a specific training epoch
  - Returns a fully functional model with architecture and weights
  - Used in Manim scenes to extract architecture information and make predictions
  - Provides detailed error messages with available model list
  - Example: `modello = carica_modello_salvato(numero_epoca=1)`

## Saved Data Files for Manim

The training process generates these files for Manim integration:

1. **`saved_models/model_epoch_XXX.keras`** - Complete saved models at ALL epochs (0-250+)
2. **`pesi_modello.pkl`** / **`model_weights.pkl`** - Final trained model weights in dictionary format
3. **`superfici_epoche.pkl`** / **`epoch_surfaces.pkl`** - Complete surface data for all training epochs
4. **`confronto_3d_epoche.html`** / **`3d_comparison_epochs.html`** - Interactive 3D visualization (reference)
5. **`perdita_addestramento.png`** / **`training_loss.png`** - Training/validation loss and MAE plots

## Usage in Manim

```python
import pickle
from nn_pitagora import *

# Load epoch data (supports both Italian and English filenames)
with open('superfici_epoche.pkl', 'rb') as f:  # or 'epoch_surfaces.pkl'
    dati_epoca = pickle.load(f)

# Access data for animation
range_a = dati_epoca['a_range'] 
range_b = dati_epoca['b_range']
Z_reale = dati_epoca['Z_actual'] 
predizioni_epoca = dati_epoca['epochs']  # Dict: {epoch: Z_predicted}

# Use consistent coloring
colori = [ottieni_colore_epoca(i, len(predizioni_epoca)) for i in range(len(predizioni_epoca))]

# Load specific epoch model
modello = carica_modello_salvato(numero_epoca=10)
architettura = modello.layers[0].units  # Get hidden layer size
```

## Manim Scene: Right Triangles (right_triangles.py)

### TriangoliRettangoli Scene

The `right_triangles.py` file contains a Manim scene that demonstrates neural network predictions on Pythagorean theorem examples using visual right triangles.

**Key Methods:**

- **`crea_triangolo_rettangolo(a, b, c, e_predizione=False)`**
  - Creates a right triangle with labeled sides
  - Parameters: cateto a, cateto b, hypotenuse c
  - `e_predizione=True` formats labels for predictions (2 decimal places, yellow color)
  - Returns VGroup with triangle, right angle indicator, and labels

- **`crea_rete_neurale(modello=None)`**
  - Creates visual representation of neural network architecture
  - Automatically extracts hidden layer size from loaded model using `modello.layers[0].units`
  - Displays up to 20 neurons, shows "..." if more exist
  - Network layout: Input (blue, 2 neurons) → Hidden (yellow, N neurons) → Output (orange, 1 neuron)
  - Arrows optimized with: `buff=0`, `stroke_width=0.5`, `max_tip_length_to_length_ratio=0.01`
  - Returns VGroup with all network components

- **`ottieni_predizioni_nn(dati_triangoli, epoch=1)`**
  - Loads model from specific epoch and generates predictions
  - Uses `carica_modello_salvato(numero_epoca=epoch)` to get model
  - Returns list of predicted hypotenuse values

- **`mostra_analisi_errore(triangoli_output, dati_triangoli, ipotenuse_predette)`**
  - Transforms prediction labels into error displays
  - Shows absolute error (Δ) and percentage error for each triangle
  - Displays average error at bottom of screen
  - Returns the error text object for cleanup

**Animation Flow:**

1. Display section labels (Input, Neural Network, Output)
2. Load epoch 1 model to extract architecture
3. Create and animate neural network visualization in center
4. Create and animate 4 input triangles (Pythagorean triples: 3-4-5, 5-12-13, 8-15-17, 7-24-25)
5. For each epoch (1, 10, 75):
   - Load model predictions
   - Create output triangles with predicted hypotenuse
   - Overlay output triangles on input triangles
   - Show error analysis (absolute and percentage)
   - Clean up output triangles and error text
   - Restore input triangle labels

**Data Structure:**

- Input triangles: `[(3,4,5), (5,12,13), (8,15,17), (7,24,25)]`
- Epochs displayed: `[1, 10, 75]`
- Triangle components: `[triangolo, angolo_retto, etichetta_a, etichetta_b, etichetta_c]`

## File Organization

### Core Implementation Files
- **`nn_pitagora.py`** - Complete neural network implementation with Manim functions (Italian function names, SGD optimizer, 10 neurons default, range -5 to 5)
- **`pitagora.py`** - Presentation version (15 neurons, 250 epochs, Adam optimizer, range 0-15, print_model_weights function, saves all models)
- **`pitagora2.py`** - Minimal educational version (15 neurons, 100 epochs, Adam optimizer, interactive while loop for demo)
- **`visualizza_parametri.py`** - Manim animation showing parameter evolution across epochs (1-250)

### Manim Animation Files
- **`right_triangles.py`** - Triangle demonstration scene (TriangoliRettangoli) showing NN predictions on Pythagorean triples across epochs 0,1,10,75 with error analysis and weight matrices visualization
- **`animate_weights.py`** - Three scenes: PesiReteNeurale (network structure), HeatmapPesi (weight heatmaps), CalcoloEsempio (step-by-step calculation for input 1,2)
- **`animate_surface.py`** - 3D surface evolution animation (SuperficieNNEvoluzioneSemplice) showing morphing from epoch 1 to 75 with blue target surface and color gradient red→green

### Generated Outputs
- **`saved_models/`** - Complete Keras models for each epoch (model_epoch_XXX.keras)
- **`media/`** - Manim-generated animation files and LaTeX artifacts
- **`*.pkl`** files - Serialized data for animations (weights, surfaces)
- **`*.png`** files - Training loss plots and performance charts
- **`*.html`** files - Interactive 3D Plotly visualizations

### Configuration
- **`uv.lock`** - Dependency lock file for `uv` package manager
- **`nn_pitagora.egg-info/`** - Package metadata

## Important Implementation Notes

### Function Name Localization
The codebase uses Italian function names. When working with this code:
- Use Italian function names: `costruisci_modello()` not `build_model()`
- File outputs may have Italian names: `perdita_addestramento.png` not `training_loss.png`
- Both Italian and English pickle files may exist for backward compatibility

### Model Architecture Details

**nn_pitagora.py (complete version)**:
- Default: 2 → 10 → 1 with 'tanh' activation (configurable)
- Optimizer: SGD with learning rate 0.001
- Training range: a,b ∈ [-5, 5]
- Default epochs: 75
- Saves epochs: [1, 3, 5, 10, 20, 30, 40, 50, 75]

**pitagora.py (presentation version)**:
- Fixed: 2 → 15 → 1 with 'relu' activation
- Optimizer: Adam
- Training range: a,b ∈ [0, 15]
- Epochs: 250 (saves all epochs)
- Includes print_model_weights() function for tabular output

**pitagora2.py (demo version)**:
- Fixed: 2 → 15 → 1 with 'relu' activation
- Optimizer: Adam
- Training range: a,b ∈ [0, 15]
- Epochs: 100 (faster for demos)
- Interactive while loop for user input

Target function for all: f(a,b) = √(a² + b²)

### Model Architecture Extraction
Manim scenes automatically extract network architecture from saved models:
```python
modello = carica_modello_salvato(numero_epoca=1)
n_neuroni_nascosti = modello.layers[0].units  # Get hidden layer size
```

### Epoch Model Management
- All epochs (0-250+) are saved as complete Keras models
- Epoch 0 represents untrained model with random weights
- Use `crea_modello_epoca_zero()` to initialize if missing
- Models are loaded with detailed error reporting for missing epochs

### Animation Data Pipeline
1. **Training** (`nn_pitagora.py`) → generates models + surface data
2. **Surface data** (`superfici_epoche.pkl`) → used by `animate_surface.py`
3. **Model weights** (`model_weights.pkl`) → used by `animate_weights.py`
4. **Epoch models** (`saved_models/`) → used by `right_triangles.py`

### Development Workflow
1. Run `python nn_pitagora.py` to train and generate all artifacts
2. Use `manim -pql <script>.py <Scene>` to render specific animations
3. Models and data persist between runs for consistent animations
4. Modify network architecture in `costruisci_modello()` - animations adapt automatically
