import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import Callback
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def generate_training_data(n_samples=5000, range_min=-5, range_max=5):
    """Generate training data for Pythagorean theorem neural network.
    
    Args:
        n_samples (int): Number of samples to generate
        range_min (float): Minimum value for a and b
        range_max (float): Maximum value for a and b
        
    Returns:
        tuple: (X_train, y_train) where X_train is input data and y_train is target
    """
    a_train = np.random.uniform(range_min, range_max, n_samples)
    b_train = np.random.uniform(range_min, range_max, n_samples)
    y_train = np.sqrt(a_train**2 + b_train**2)
    X_train = np.column_stack((a_train, b_train))
    return X_train, y_train
def build_model(hidden_units=50, activation='tanh', learning_rate=0.001):
    """Build and compile the neural network model.
    
    Args:
        hidden_units (int): Number of neurons in hidden layer
        activation (str): Activation function for hidden layer
        learning_rate (float): Learning rate for optimizer
        
    Returns:
        Sequential: Compiled Keras model
    """
    model = Sequential([
        Dense(hidden_units, activation=activation, input_shape=(2,)),
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def display_model_info(model):
    """Display model architecture information."""
    print("Model Architecture:")
    model.summary()

class SaveModelAtEpochs(Callback):
    """Custom callback to save model weights at specific epochs."""
    
    def __init__(self, epochs_to_save):
        super().__init__()
        self.epochs_to_save = epochs_to_save
        self.saved_models = {}

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        if epoch_num in self.epochs_to_save:
            print(f"\nSaving model at epoch {epoch_num}")
            self.saved_models[epoch_num] = [w.copy() for w in self.model.get_weights()]

def train_model(model, X_train, y_train, epochs=75, batch_size=32, validation_split=0.2, epochs_to_save=None):
    """Train the neural network model.
    
    Args:
        model: Compiled Keras model
        X_train: Training input data
        y_train: Training target data
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        validation_split (float): Fraction of data to use for validation
        epochs_to_save (list): Epochs at which to save model weights
        
    Returns:
        tuple: (history, save_callback) Training history and callback with saved models
    """
    print("\nTraining the model...")
    
    if epochs_to_save is None:
        epochs_to_save = [1, 3, 5, 10, 20, 30, 40, 50, 75]
    
    save_callback = SaveModelAtEpochs(epochs_to_save)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
        callbacks=[save_callback]
    )
    
    return history, save_callback

def save_model_weights(model, filename='model_weights.pkl'):
    """Save model weights for external use (e.g., Manim animation).
    
    Args:
        model: Trained Keras model
        filename (str): Output filename for weights
    """
    weights = model.get_weights()
    weights_data = {
        'W1': weights[0],  # Input to hidden layer weights
        'b1': weights[1],  # Hidden layer biases
        'W2': weights[2],  # Hidden to output layer weights
        'b2': weights[3]   # Output layer bias
    }
    with open(filename, 'wb') as f:
        pickle.dump(weights_data, f)
    print(f"\nModel weights saved to '{filename}'")

def plot_training_history(history, save_filename='training_loss.png', show_plot=True):
    """Plot training loss and MAE history.
    
    Args:
        history: Keras training history object
        save_filename (str): Filename to save the plot
        show_plot (bool): Whether to display the plot
    """
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
    plt.savefig(save_filename)
    print(f"\nTraining plots saved as '{save_filename}'")
    
    if show_plot:
        plt.show()

def evaluate_model_on_test_cases(model):
    """Evaluate the model on specific test cases.
    
    Args:
        model: Trained Keras model
    """
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

def get_predictions_for_epoch(weights, X_input, output_shape, hidden_units=50, activation='tanh'):
    """Create a temporary model and get predictions for a specific epoch.
    
    Args:
        weights: Model weights from specific epoch
        X_input: Input data for predictions
        output_shape: Shape to reshape predictions
        hidden_units (int): Number of hidden units
        activation (str): Activation function
        
    Returns:
        ndarray: Reshaped predictions
    """
    temp_model = Sequential([
        Dense(hidden_units, activation=activation, input_shape=(2,)),
        Dense(1, activation='linear')
    ])
    temp_model.build((None, 2))
    temp_model.set_weights(weights)
    return temp_model.predict(X_input, verbose=0).reshape(output_shape)


def get_epoch_color(epoch_idx, total_epochs):
    """Generate color gradient from red to green.
    
    Args:
        epoch_idx (int): Index of current epoch
        total_epochs (int): Total number of epochs
        
    Returns:
        str: Color name for the epoch
    """
    colors = ['red', 'orange', 'yellow', 'yellowgreen', 'green']
    position = epoch_idx / (total_epochs - 1) if total_epochs > 1 else 0
    color_idx = position * (len(colors) - 1)
    idx_low = int(color_idx)
    return colors[idx_low]


def create_3d_visualization(save_callback, epochs_to_plot, resolution=50, hidden_units=50, activation='tanh'):
    """Create 3D visualization comparing actual function with NN predictions.
    
    Args:
        save_callback: Callback object with saved model weights
        epochs_to_plot (list): List of epochs to visualize
        resolution (int): Grid resolution for visualization
        hidden_units (int): Number of hidden units in model
        activation (str): Activation function used in model
        
    Returns:
        tuple: (fig, a_range, b_range, Z_actual, X_grid)
    """
    print("\nGenerating 3D comparison plot...")
    
    # Create meshgrid
    a_range = np.linspace(-5, 5, resolution)
    b_range = np.linspace(-5, 5, resolution)
    A, B = np.meshgrid(a_range, b_range)
    
    # Calculate actual function values
    Z_actual = np.sqrt(A**2 + B**2)
    
    # Prepare input for predictions
    X_grid = np.column_stack((A.ravel(), B.ravel()))
    
    # Create figure
    fig = go.Figure()
    
    print("\nApproximation Quality at different epochs:")
    print("="*60)
    
    # Add actual function surface
    fig.add_trace(go.Surface(
        x=A, y=B, z=Z_actual,
        colorscale=[[0, 'blue'], [1, 'blue']],
        opacity=0.3,
        showscale=False,
        name='Actual Function',
        hovertemplate='a: %{x}<br>b: %{y}<br>Actual: %{z:.2f}<extra></extra>'
    ))
    
    # Add NN prediction surfaces for each epoch
    for idx, epoch in enumerate(epochs_to_plot):
        Z_predicted = get_predictions_for_epoch(
            save_callback.saved_models[epoch], 
            X_grid, 
            A.shape, 
            hidden_units, 
            activation
        )
        
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
    
    return fig, a_range, b_range, Z_actual, X_grid


def save_epoch_data_for_animation(save_callback, epochs_to_plot, a_range, b_range, Z_actual, X_grid, hidden_units=50, activation='tanh', filename='epoch_surfaces.pkl'):
    """Save epoch data for Manim animation.
    
    Args:
        save_callback: Callback object with saved model weights
        epochs_to_plot (list): List of epochs to save
        a_range, b_range: Coordinate ranges
        Z_actual: Actual function values
        X_grid: Input grid for predictions
        hidden_units (int): Number of hidden units
        activation (str): Activation function
        filename (str): Output filename
    """
    print("\nSaving epoch data for Manim animation...")
    epoch_data = {
        'a_range': a_range,
        'b_range': b_range,
        'Z_actual': Z_actual,
        'epochs': {}
    }
    
    for epoch in epochs_to_plot:
        Z_predicted = get_predictions_for_epoch(
            save_callback.saved_models[epoch], 
            X_grid, 
            Z_actual.shape, 
            hidden_units, 
            activation
        )
        epoch_data['epochs'][epoch] = Z_predicted
    
    with open(filename, 'wb') as f:
        pickle.dump(epoch_data, f)
    print(f"Epoch surface data saved to '{filename}' for Manim animation")


def save_interactive_plot(fig, filename='3d_comparison_epochs.html'):
    """Save interactive 3D plot as HTML file.
    
    Args:
        fig: Plotly figure object
        filename (str): Output filename
    """
    fig.write_html(filename)
    print(f"\nInteractive 3D comparison plot saved as '{filename}'")


def main():
    """Main function to orchestrate the entire neural network workflow."""
    # Configuration
    hidden_units = 50
    activation = 'relu'
    learning_rate = 0.001
    epochs = 75
    epochs_to_plot = [1, 3, 5, 10, 20, 30, 40, 50, 75]
    
    # Set random seed
    set_random_seed(42)
    
    # Generate training data
    X_train, y_train = generate_training_data(n_samples=5000)
    
    # Build and display model
    model = build_model(hidden_units, activation, learning_rate)
    display_model_info(model)
    
    # Train the model
    history, save_callback = train_model(
        model, X_train, y_train, 
        epochs=epochs, 
        epochs_to_save=epochs_to_plot
    )
    
    # Save model weights
    save_model_weights(model)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model on test cases
    evaluate_model_on_test_cases(model)
    
    # Create 3D visualization
    fig, a_range, b_range, Z_actual, X_grid = create_3d_visualization(
        save_callback, epochs_to_plot, 
        hidden_units=hidden_units, 
        activation=activation
    )
    
    # Save epoch data for animation
    save_epoch_data_for_animation(
        save_callback, epochs_to_plot, 
        a_range, b_range, Z_actual, X_grid,
        hidden_units=hidden_units, 
        activation=activation
    )
    
    # Save interactive plot
    save_interactive_plot(fig)


if __name__ == "__main__":
    main()
