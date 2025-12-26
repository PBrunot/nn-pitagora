# Neural Network Pythagorean Theorem Project

## Key Functions for Manim Integration

The following functions from `nn_pitagora.py` are specifically designed for use in Manim Python animation files:

### Data Generation
- **`generate_training_data(n_samples=5000, range_min=-5, range_max=5)`**
  - Generates training data for the Pythagorean theorem neural network
  - Returns `(X_train, y_train)` tuple
  - Used to create consistent datasets for animations

### Model Creation and Training
- **`build_model(hidden_units=50, activation='relu', learning_rate=0.001)`**
  - Creates and compiles a neural network model
  - Returns compiled Keras Sequential model
  - Default activation changed to 'relu' for better performance

- **`train_model(model, X_train, y_train, epochs=75, epochs_to_save=None)`**
  - Trains the model and saves weights at specified epochs
  - Returns `(history, save_callback)` tuple
  - The `save_callback.saved_models` contains weights for each saved epoch

### Weight Management
- **`save_model_weights(model, filename='model_weights.pkl')`**
  - Saves final model weights in a structured format for Manim
  - Exports weights as: `{'W1': weights[0], 'b1': weights[1], 'W2': weights[2], 'b2': weights[3]}`

### Visualization Functions
- **`get_predictions_for_epoch(weights, X_input, output_shape, hidden_units=50, activation='relu')`**
  - Creates temporary model from saved weights and generates predictions
  - Essential for animating neural network learning progress
  - Used to show how predictions change across training epochs

- **`create_3d_visualization(save_callback, epochs_to_plot, resolution=50, hidden_units=50, activation='relu')`**
  - Creates 3D surface comparison between actual function and NN predictions
  - Returns `(fig, a_range, b_range, Z_actual, X_grid)` for further processing
  - Generates data needed for Manim 3D surface animations

- **`save_epoch_data_for_animation(save_callback, epochs_to_plot, a_range, b_range, Z_actual, X_grid, filename='epoch_surfaces.pkl')`**
  - Saves complete epoch surface data specifically for Manim animations
  - Creates pickle file with all necessary data for frame-by-frame animation
  - Data structure: `{'a_range': a_range, 'b_range': b_range, 'Z_actual': Z_actual, 'epochs': {epoch: Z_predicted}}`

### Utility Functions
- **`get_epoch_color(epoch_idx, total_epochs)`**
  - Generates color gradient from red to green for epoch progression
  - Useful for consistent color schemes in Manim animations

- **`set_random_seed(seed=42)`**
  - Ensures reproducible results for consistent animations

## Model Loading

- **`load_saved_model(epoch_number, models_directory="saved_models")`**
  - Loads a complete Keras model from a specific training epoch
  - Returns a fully functional model with architecture and weights
  - Used in Manim scenes to extract architecture information and make predictions
  - Example: `model = load_saved_model(epoch_number=1)`

## Saved Data Files for Manim

The training process generates these files for Manim integration:

1. **`saved_models/model_epoch_XXX.keras`** - Complete saved models at specific epochs (1, 3, 5, 10, 20, 30, 40, 50, 75)
2. **`model_weights.pkl`** - Final trained model weights in dictionary format
3. **`epoch_surfaces.pkl`** - Complete surface data for all training epochs
4. **`3d_comparison_epochs.html`** - Interactive 3D visualization (reference)

## Usage in Manim

```python
import pickle
from nn_pitagora import *

# Load epoch data
with open('epoch_surfaces.pkl', 'rb') as f:
    epoch_data = pickle.load(f)

# Access data for animation
a_range = epoch_data['a_range']
b_range = epoch_data['b_range'] 
Z_actual = epoch_data['Z_actual']
epoch_predictions = epoch_data['epochs']  # Dict: {epoch: Z_predicted}

# Use get_epoch_color() for consistent coloring
colors = [get_epoch_color(i, len(epoch_predictions)) for i in range(len(epoch_predictions))]
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
  - Uses `load_saved_model(epoch_number=epoch)` to get model
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

## Configuration Notes

- Default activation function changed to 'relu' for better convergence
- Model architecture: 2 → 50 → 1 (input → hidden → output)
- Training range: a,b ∈ [-5, 5]
- Target function: f(a,b) = √(a² + b²)
- Default epochs to save: [1, 3, 5, 10, 20, 30, 40, 50, 75]
- Models saved to: `saved_models/model_epoch_XXX.keras`
