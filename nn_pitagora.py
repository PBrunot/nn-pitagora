import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import Callback
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)

# Generate training data
n_samples = 5000
a_train = np.random.uniform(-5, 5, n_samples)
b_train = np.random.uniform(-5, 5, n_samples)
y_train = np.sqrt(a_train**2 + b_train**2)

# Prepare input data (combine a and b into a single array)
X_train = np.column_stack((a_train, b_train))
NN_COUNT = 50
ACTIVATION = 'tanh'

# Build the neural network
model = Sequential([
    Dense(NN_COUNT, activation=ACTIVATION, input_shape=(2,)),  # Hidden layer with 10 neurons
    Dense(1, activation='linear')  # Output layer (linear for regression)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])

# Display model architecture
print("Model Architecture:")
model.summary()

# Custom callback to save model at specific epochs
class SaveModelAtEpochs(Callback):
    def __init__(self, epochs_to_save):
        super().__init__()
        self.epochs_to_save = epochs_to_save
        self.saved_models = {}

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1  # Keras uses 0-indexed epochs
        if epoch_num in self.epochs_to_save:
            print(f"\nSaving model at epoch {epoch_num}")
            # Clone current weights
            self.saved_models[epoch_num] = [w.copy() for w in self.model.get_weights()]

# Train the model
print("\nTraining the model...")
epochs_to_plot = [1, 3, 5, 10, 20, 30, 40, 50, 75]
save_callback = SaveModelAtEpochs(epochs_to_plot)

history = model.fit(X_train, y_train,
                    epochs=75,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[save_callback])

# Save model weights for Manim animation
weights = model.get_weights()
weights_data = {
    'W1': weights[0],  # Input to hidden layer weights (2, n_hidden)
    'b1': weights[1],  # Hidden layer biases (n_hidden,)
    'W2': weights[2],  # Hidden to output layer weights (n_hidden, 1)
    'b2': weights[3]   # Output layer bias (1,)
}
with open('model_weights.pkl', 'wb') as f:
    pickle.dump(weights_data, f)
print("\nModel weights saved to 'model_weights.pkl'")

# Plot the loss during training
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Loss During Training')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('MAE During Training')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_loss.png')
print("\nTraining plots saved as 'training_loss.png'")
plt.show()

# Verify on test cases
print("\n" + "="*60)
print("Testing the model on specific (a, b) couples:")
print("="*60)

test_cases = [
    (1, 2),      # Should be sqrt(5) ≈ 2.236
    (3, 4),      # Should be 5
    (-3, 4),     # Should be 5
    (3, -4),     # Should be 5
    (-3, -4),    # Should be 5
    (1, 1),      # Should be sqrt(2) ≈ 1.414
    (-1, -1),    # Should be sqrt(2) ≈ 1.414
    (0, 5),      # Should be 5
    (5, 0),      # Should be 5
    (-5, 0),     # Should be 5
    (0, -5),     # Should be 5
    (4.5, 4.5),  # Should be sqrt(40.5) ≈ 6.364
]

for a, b in test_cases:
    X_test = np.array([[a, b]])
    prediction = model.predict(X_test, verbose=0)[0][0]
    actual = np.sqrt(a**2 + b**2)
    error = abs(prediction - actual)
    error_pct = (error / actual) * 100 if actual != 0 else 0

    print(f"a={a:6.2f}, b={b:6.2f} | "
          f"Predicted: {prediction:7.4f} | "
          f"Actual: {actual:7.4f} | "
          f"Error: {error:6.4f} ({error_pct:5.2f}%)")

print("="*60)

# 3D Visualization: Compare original function with NN predictions at different epochs
print("\nGenerating 3D comparison plot for epochs 1, 10, 25, 50...")

# Create a meshgrid for the interval [-5, 5]
a_range = np.linspace(-5, 5, 50)  # Reduced resolution for better performance
b_range = np.linspace(-5, 5, 50)
A, B = np.meshgrid(a_range, b_range)

# Calculate the actual function values
Z_actual = np.sqrt(A**2 + B**2)

# Prepare input for predictions
X_grid = np.column_stack((A.ravel(), B.ravel()))

# Function to create a temporary model and get predictions
def get_predictions_for_epoch(weights, X_input):
    temp_model = Sequential([
        Dense(NN_COUNT, activation=ACTIVATION, input_shape=(2,)),
        Dense(1, activation='linear')
    ])
    temp_model.build((None, 2))
    temp_model.set_weights(weights)
    return temp_model.predict(X_input, verbose=0).reshape(A.shape)

# Create single plot with all epochs
fig = go.Figure()

# Dynamic color scheme - interpolate from red to green based on epoch progression
import plotly.colors as pc

def get_epoch_color(epoch_idx, total_epochs):
    """Generate color gradient from red to green"""
    colors = ['red', 'orange', 'yellow', 'yellowgreen', 'green']
    # Map epoch index to color scale
    position = epoch_idx / (total_epochs - 1) if total_epochs > 1 else 0
    color_idx = position * (len(colors) - 1)
    idx_low = int(color_idx)
    idx_high = min(idx_low + 1, len(colors) - 1)

    return colors[idx_low] if idx_low == idx_high else colors[idx_low]

print("\nApproximation Quality at different epochs:")
print("="*60)

# Add actual function surface (blue, semi-transparent)
fig.add_trace(go.Surface(
    x=A, y=B, z=Z_actual,
    colorscale=[[0, 'blue'], [1, 'blue']],
    opacity=0.3,
    showscale=False,
    name='Actual Function',
    hovertemplate='a: %{x}<br>b: %{y}<br>Actual: %{z:.2f}<extra></extra>'
))

# Add NN prediction surfaces for each epoch with different colors
for idx, epoch in enumerate(epochs_to_plot):
    Z_predicted = get_predictions_for_epoch(save_callback.saved_models[epoch], X_grid)

    # Calculate metrics
    difference = np.abs(Z_predicted - Z_actual)
    mean_error = np.mean(difference)
    max_error = np.max(difference)
    rmse = np.sqrt(np.mean(difference**2))

    print(f"Epoch {epoch:2d} | MAE: {mean_error:7.4f} | Max Error: {max_error:7.4f} | RMSE: {rmse:7.4f}")

    color = get_epoch_color(idx, len(epochs_to_plot))
    fig.add_trace(go.Surface(
        x=A, y=B, z=Z_predicted,
        colorscale=[[0, color], [1, color]],
        opacity=0.4,
        showscale=False,
        name=f'Epoch {epoch}',
        hovertemplate=f'Epoch {epoch}<br>a: %{{x}}<br>b: %{{y}}<br>Predicted: %{{z:.2f}}<extra></extra>'
    ))

print("="*60)

# Update layout
epoch_range = f"Epochs {epochs_to_plot[0]}→{epochs_to_plot[-1]}"
fig.update_layout(
    title=f'Neural Network Learning Progress: Approximating f(a,b) = √(a² + b²)<br><sub>Blue=Actual, Red→Green={epoch_range}</sub>',
    scene=dict(
        xaxis_title='a',
        yaxis_title='b',
        zaxis_title='f(a,b)',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3)
        )
    ),
    height=800,
    width=1200,
    showlegend=True,
    legend=dict(x=0.7, y=0.9)
)

# Save epoch data for Manim animation
print("\nSaving epoch data for Manim animation...")
epoch_data = {
    'a_range': a_range,
    'b_range': b_range,
    'Z_actual': Z_actual,
    'epochs': {}
}

for epoch in epochs_to_plot:
    Z_predicted = get_predictions_for_epoch(save_callback.saved_models[epoch], X_grid)
    epoch_data['epochs'][epoch] = Z_predicted

with open('epoch_surfaces.pkl', 'wb') as f:
    pickle.dump(epoch_data, f)
print("Epoch surface data saved to 'epoch_surfaces.pkl' for Manim animation")

# Save as HTML
fig.write_html('3d_comparison_epochs.html')
print("\nInteractive 3D comparison plot saved as '3d_comparison_epochs.html'")
