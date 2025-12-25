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

## Saved Data Files for Manim

The training process generates these files for Manim integration:

1. **`model_weights.pkl`** - Final trained model weights
2. **`epoch_surfaces.pkl`** - Complete surface data for all training epochs
3. **`3d_comparison_epochs.html`** - Interactive 3D visualization (reference)

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

## Configuration Notes

- Default activation function changed to 'relu' for better convergence
- Model architecture: 2 → 50 → 1 (input → hidden → output)
- Training range: a,b ∈ [-5, 5]
- Target function: f(a,b) = √(a² + b²)
- Default epochs to save: [1, 3, 5, 10, 20, 30, 40, 50, 75]