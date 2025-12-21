# Neural Network Pythagorean Theorem Implementation

This project implements a feedforward neural network to learn the Pythagorean theorem: f(a,b) = √(a² + b²)

Uses TensorFlow/Keras for training, Plotly for interactive 3D visualization, and Manim for professional animations showing the learning process.

## Architecture
- **Input Layer**: 2 neurons (a, b)
- **Hidden Layer**: Configurable neurons with ReLU activation (default: 30)
- **Output Layer**: 1 neuron with linear activation

The network dynamically adapts - change the hidden layer size in `nn_pitagora.py` and the animations will automatically adjust.

## Installation

### 1. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Required Libraries
```bash
pip install numpy matplotlib tensorflow plotly manim
```

**Required packages:**
- `numpy` - Numerical computing and data generation
- `matplotlib` - Training loss visualization
- `tensorflow` (includes `keras`) - Neural network training framework
- `plotly` - Interactive 3D surface plots
- `manim` - Mathematical animation engine (for video generation)

**System dependencies for Manim:**
- LaTeX distribution (TeX Live or MiKTeX)
- FFmpeg
- Cairo

On Ubuntu/Debian:
```bash
sudo apt-get install texlive texlive-latex-extra ffmpeg libcairo2-dev
```

On macOS:
```bash
brew install --cask mactex
brew install ffmpeg cairo
```

### 3. Verify Installation
```bash
python -c "import numpy, matplotlib, keras, plotly, manim; print('All packages installed successfully!')"
```

## Files

### nn_pitagora.py
Main training script that:
- Generates training samples (configurable, default: 2,000)
- Trains the neural network (configurable epochs, default: 100)
- Plots training/validation loss and MAE
- Tests predictions on sample inputs
- Saves weights to `model_weights.pkl` for animation

**Configurable parameters:**
- `n_samples`: Number of training samples
- `Dense(N, ...)`: Number of hidden neurons
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training

### animate_weights.py
Manim animation script with three scenes:
1. **NeuralNetworkWeights**: Visualizes the network structure with colored connections and weight values on each arrow
2. **WeightHeatmap**: Shows heatmaps of the weight matrices
3. **SampleCalculation**: Step-by-step calculation demonstration for input (1, 2)

### animate_surface.py
Manim 3D animation showing the evolution of the NN's surface approximation through training epochs. Features smooth morphing transitions, rotating camera, and color gradient from red (epoch 1) to green (epoch 50). See ANIMATION_GUIDE.md for detailed usage and customization options.

## Usage

### 1. Train the Model
```bash
python nn_pitagora.py
```

This will:
- Train the neural network
- Save `training_loss.png` with loss curves
- Save `model_weights.pkl` for animations
- Display test predictions

### 2. Generate Animations (requires Manim)

Install Manim if needed:
```bash
pip install manim
```

Render the network visualization:
```bash
manim -pql animate_weights.py NeuralNetworkWeights
```

Render the weight heatmap:
```bash
manim -pql animate_weights.py WeightHeatmap
```

Render the sample calculation for (1, 2):
```bash
manim -pql animate_weights.py SampleCalculation
```

Render all scenes at once:
```bash
manim -pql animate_weights.py NeuralNetworkWeights WeightHeatmap SampleCalculation
```

Quality options:
- `-ql`: Low quality (480p) - fast
- `-qm`: Medium quality (720p)
- `-qh`: High quality (1080p)
- `-qk`: 4K quality
- `-p`: Preview after rendering

## Expected Results

The model should achieve very low error (< 1%) on the Pythagorean theorem prediction after training.

Example predictions:
- f(3, 4) ≈ 5.0
- f(5, 12) ≈ 13.0
- f(6, 8) ≈ 10.0

## Visualization Details

### Network Animation
- **Red connections**: Positive weights
- **Blue connections**: Negative weights
- **Line thickness**: Weight magnitude
- **Weight values**: Displayed on each connection arrow
- Automatically shows all neurons if ≤15, otherwise shows a subset
- Dynamically adapts to any network size

### Weight Heatmap
- **Blue**: Minimum weight values
- **Red**: Maximum weight values
- W1: Input→Hidden layer (2×N matrix)
- W2: Hidden→Output layer (N×1 matrix shown in grid layout)
- Automatically scales grid layout based on network size

### Sample Calculation
- Demonstrates complete forward pass for input (1, 2)
- **Step 1**: Shows input vector [1, 2]
- **Step 2**: Calculates hidden layer activations
  - Shows weighted sum calculation for first 3 neurons
  - Applies ReLU activation function
  - Color-coded: GREEN for active neurons, RED for suppressed
- **Step 3**: Calculates final output
  - Shows weighted sum of hidden activations
  - Displays final prediction
- **Comparison**: Shows predicted vs actual value with error percentage
- Expected output: √5 ≈ 2.236
