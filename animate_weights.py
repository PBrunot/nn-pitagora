from manim import *
import numpy as np
from keras.models import load_model
import pickle

class NeuralNetworkWeights(Scene):
    def construct(self):
        # Load the trained model weights
        try:
            with open('model_weights.pkl', 'rb') as f:
                weights_data = pickle.load(f)
            W1 = weights_data['W1']  # Shape: (2, 100)
            b1 = weights_data['b1']  # Shape: (100,)
            W2 = weights_data['W2']  # Shape: (100, 1)
            b2 = weights_data['b2']  # Shape: (1,)
        except FileNotFoundError:
            self.add(Text("Error: model_weights.pkl not found!", color=RED))
            return

        # Title
        title = Text("Neural Network: f(a,b) = √(a² + b²)", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Get actual network size
        n_hidden_neurons = W1.shape[1]

        # Network architecture text
        arch_text = Text(f"Architecture: 2 → {n_hidden_neurons} (ReLU) → 1 (Linear)", font_size=24)
        arch_text.next_to(title, DOWN)
        self.play(FadeIn(arch_text))
        self.wait(0.5)

        # Create simplified network visualization (show all if <=15, else show 15)
        n_display_neurons = min(n_hidden_neurons, 15)

        # Input layer (2 neurons)
        input_neurons = VGroup()
        input_labels = ["a", "b"]
        for i in range(2):
            neuron = Circle(radius=0.3, color=BLUE, fill_opacity=0.5)
            neuron.move_to(LEFT * 5 + UP * (1 - i) * 1.5)
            label = Text(input_labels[i], font_size=24)
            label.move_to(neuron.get_center())
            input_neurons.add(VGroup(neuron, label))

        # Hidden layer (showing subset of 100)
        hidden_neurons = VGroup()
        y_positions = np.linspace(3, -3, n_display_neurons)
        for i in range(n_display_neurons):
            neuron = Circle(radius=0.25, color=GREEN, fill_opacity=0.5)
            neuron.move_to(RIGHT * 0 + UP * y_positions[i])
            hidden_neurons.add(neuron)

        # Output neuron
        output_neuron = Circle(radius=0.3, color=RED, fill_opacity=0.5)
        output_neuron.move_to(RIGHT * 5 + UP * 0)
        output_label = Text("√(a²+b²)", font_size=20)
        output_label.move_to(output_neuron.get_center())
        output_group = VGroup(output_neuron, output_label)

        # Draw neurons
        self.play(
            *[FadeIn(inp) for inp in input_neurons],
            run_time=0.5
        )
        self.play(
            *[FadeIn(h) for h in hidden_neurons],
            run_time=0.5
        )
        self.play(FadeIn(output_group), run_time=0.5)

        # Sample indices to display from the hidden neurons
        display_indices = np.linspace(0, n_hidden_neurons - 1, n_display_neurons, dtype=int)

        # Create connections from input to hidden layer with weight labels
        connections_input_hidden = VGroup()
        weight_labels_ih = VGroup()

        for i in range(2):
            for j, h_idx in enumerate(display_indices):
                weight = W1[i, h_idx]
                # Normalize weight for color (assuming weights are roughly in [-1, 1] range)
                normalized_weight = np.clip(weight / 2, -1, 1)

                if normalized_weight > 0:
                    color = interpolate_color(WHITE, RED, abs(normalized_weight))
                else:
                    color = interpolate_color(WHITE, BLUE, abs(normalized_weight))

                line = Line(
                    input_neurons[i][0].get_center(),
                    hidden_neurons[j].get_center(),
                    stroke_width=abs(weight) * 2,
                    color=color,
                    stroke_opacity=0.6
                )
                connections_input_hidden.add(line)

                # Add weight value label on the connection
                weight_text = Text(f"{weight:.2f}", font_size=10, color=color)
                weight_text.move_to(line.get_center())
                weight_text.rotate(line.get_angle())
                weight_labels_ih.add(weight_text)

        # Create connections from hidden to output layer with weight labels
        connections_hidden_output = VGroup()
        weight_labels_ho = VGroup()

        for j, h_idx in enumerate(display_indices):
            weight = W2[h_idx, 0]
            normalized_weight = np.clip(weight / 2, -1, 1)

            if normalized_weight > 0:
                color = interpolate_color(WHITE, RED, abs(normalized_weight))
            else:
                color = interpolate_color(WHITE, BLUE, abs(normalized_weight))

            line = Line(
                hidden_neurons[j].get_center(),
                output_neuron.get_center(),
                stroke_width=abs(weight) * 2,
                color=color,
                stroke_opacity=0.6
            )
            connections_hidden_output.add(line)

            # Add weight value label on the connection
            weight_text = Text(f"{weight:.2f}", font_size=10, color=color)
            weight_text.move_to(line.get_center())
            weight_text.rotate(line.get_angle())
            weight_labels_ho.add(weight_text)

        # Animate connections
        self.play(
            *[Create(conn) for conn in connections_input_hidden],
            run_time=2
        )
        self.play(
            *[FadeIn(label) for label in weight_labels_ih],
            run_time=1
        )
        self.play(
            *[Create(conn) for conn in connections_hidden_output],
            run_time=2
        )
        self.play(
            *[FadeIn(label) for label in weight_labels_ho],
            run_time=1
        )

        # Add legend
        legend = VGroup()
        legend_title = Text("Weight Colors:", font_size=20)
        legend_pos = Text("Red = Positive", font_size=18, color=RED)
        legend_neg = Text("Blue = Negative", font_size=18, color=BLUE)
        legend_thick = Text("Thickness = Magnitude", font_size=18)

        legend.add(legend_title, legend_pos, legend_neg, legend_thick)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.to_corner(DL)

        self.play(FadeIn(legend))

        # Add note about simplified view
        if n_display_neurons < n_hidden_neurons:
            note = Text(
                f"Showing {n_display_neurons} of {n_hidden_neurons} hidden neurons",
                font_size=18,
                color=YELLOW
            )
            note.to_corner(DR)
            self.play(FadeIn(note))
        else:
            note = Text(
                f"Showing all {n_hidden_neurons} hidden neurons",
                font_size=18,
                color=YELLOW
            )
            note.to_corner(DR)
            self.play(FadeIn(note))

        self.wait(2)

        # Show weight statistics
        stats = VGroup()
        w1_mean = Text(f"W1 mean: {np.mean(W1):.4f}", font_size=20)
        w1_std = Text(f"W1 std: {np.std(W1):.4f}", font_size=20)
        w2_mean = Text(f"W2 mean: {np.mean(W2):.4f}", font_size=20)
        w2_std = Text(f"W2 std: {np.std(W2):.4f}", font_size=20)

        stats.add(w1_mean, w1_std, w2_mean, w2_std)
        stats.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        stats.to_corner(UR)

        self.play(FadeIn(stats))
        self.wait(3)

        # Fade out everything
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )


class WeightHeatmap(Scene):
    def construct(self):
        # Load the trained model weights
        try:
            with open('model_weights.pkl', 'rb') as f:
                weights_data = pickle.load(f)
            W1 = weights_data['W1']  # Shape: (2, 100)
            W2 = weights_data['W2']  # Shape: (100, 1)
        except FileNotFoundError:
            self.add(Text("Error: model_weights.pkl not found!", color=RED))
            return

        # Get actual network size
        n_hidden_neurons = W1.shape[1]

        # Title
        title = Text("Weight Heatmaps", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Create heatmap for W1
        w1_title = Text(f"Input → Hidden Layer Weights (2×{n_hidden_neurons})", font_size=24)
        w1_title.move_to(UP * 2.5 + LEFT * 3)

        # Normalize W1 for visualization
        W1_norm = (W1 - W1.min()) / (W1.max() - W1.min())

        # Create grid for W1 (show as 2 rows, n_hidden columns)
        n_display_w1 = min(n_hidden_neurons, 50)  # Limit to 50 for visibility
        cell_width = min(0.1, 5.0 / n_display_w1)  # Adjust width based on count
        cell_height = 0.3
        w1_grid = VGroup()

        for i in range(2):
            for j in range(n_display_w1):
                color_value = W1_norm[i, j]
                color = interpolate_color(BLUE, RED, color_value)

                rect = Rectangle(
                    width=cell_width,
                    height=cell_height,
                    fill_color=color,
                    fill_opacity=0.8,
                    stroke_width=0.5
                )
                rect.move_to(
                    LEFT * 3 +
                    RIGHT * (j * cell_width - n_display_w1 * cell_width / 2) +
                    UP * (0.5 - i * cell_height)
                )
                w1_grid.add(rect)

        self.play(Write(w1_title))
        self.play(FadeIn(w1_grid), run_time=1.5)

        # Create heatmap for W2
        w2_title = Text(f"Hidden → Output Weights ({n_hidden_neurons}×1)", font_size=24)
        w2_title.move_to(UP * 2.5 + RIGHT * 3)

        # Normalize W2 for visualization
        W2_norm = (W2 - W2.min()) / (W2.max() - W2.min())

        # Create grid for W2 (arrange in grid pattern)
        w2_grid = VGroup()

        # Calculate grid dimensions (try for square-ish layout)
        grid_cols = int(np.ceil(np.sqrt(n_hidden_neurons)))
        grid_rows = int(np.ceil(n_hidden_neurons / grid_cols))

        cell_width2 = min(0.1, 2.5 / grid_cols)
        cell_height2 = min(0.1, 2.5 / grid_rows)

        for i in range(n_hidden_neurons):
            color_value = W2_norm[i, 0]
            color = interpolate_color(BLUE, RED, color_value)

            rect = Rectangle(
                width=cell_width2,
                height=cell_height2,
                fill_color=color,
                fill_opacity=0.8,
                stroke_width=0.3
            )
            # Arrange in grid
            row = i // grid_cols
            col = i % grid_cols
            rect.move_to(
                RIGHT * 3 +
                RIGHT * (col * cell_width2 - grid_cols * cell_width2 / 2) +
                UP * (grid_rows * cell_height2 / 2 - row * cell_height2)
            )
            w2_grid.add(rect)

        self.play(Write(w2_title))
        self.play(FadeIn(w2_grid), run_time=1.5)

        # Add color scale legend
        legend_title = Text("Color Scale:", font_size=20)
        legend_title.to_corner(DL).shift(UP * 0.5)

        # Create color bar
        color_bar = VGroup()
        for i in range(20):
            color_value = i / 19
            color = interpolate_color(BLUE, RED, color_value)
            segment = Rectangle(
                width=0.3,
                height=0.1,
                fill_color=color,
                fill_opacity=0.8,
                stroke_width=0.5
            )
            segment.move_to(LEFT * 5.5 + RIGHT * (i * 0.3) + DOWN * 2.5)
            color_bar.add(segment)

        min_label = Text("Min", font_size=16).next_to(color_bar[0], DOWN)
        max_label = Text("Max", font_size=16).next_to(color_bar[-1], DOWN)

        self.play(Write(legend_title))
        self.play(FadeIn(color_bar), FadeIn(min_label), FadeIn(max_label))

        self.wait(3)


class SampleCalculation(Scene):
    def construct(self):
        # Load the trained model weights
        try:
            with open('model_weights.pkl', 'rb') as f:
                weights_data = pickle.load(f)
            W1 = weights_data['W1']  # Shape: (2, n_hidden)
            b1 = weights_data['b1']  # Shape: (n_hidden,)
            W2 = weights_data['W2']  # Shape: (n_hidden, 1)
            b2 = weights_data['b2']  # Shape: (1,)
        except FileNotFoundError:
            self.add(Text("Error: model_weights.pkl not found!", color=RED))
            return

        n_hidden = W1.shape[1]

        # Title
        title = Text("Sample Calculation: f(1, 2)", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Show expected output
        expected = np.sqrt(1**2 + 2**2)
        expected_text = Text(f"Expected: √(1² + 2²) = √5 ≈ {expected:.4f}", font_size=24, color=YELLOW)
        expected_text.next_to(title, DOWN)
        self.play(FadeIn(expected_text))
        self.wait(1)

        # Step 1: Show input
        step1_title = Text("Step 1: Input Layer", font_size=28, color=BLUE)
        step1_title.move_to(UP * 2)

        input_eq = MathTex(r"x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}", font_size=40)
        input_eq.move_to(UP * 1)

        self.play(Write(step1_title))
        self.play(Write(input_eq))
        self.wait(1.5)

        # Step 2: Calculate hidden layer (show first 3 neurons for clarity)
        self.play(FadeOut(step1_title), FadeOut(input_eq))

        step2_title = Text("Step 2: Hidden Layer (ReLU)", font_size=28, color=GREEN)
        step2_title.move_to(UP * 2.5)
        self.play(Write(step2_title))

        # Calculate z = W1^T @ x + b1
        x_input = np.array([1, 2])
        z_hidden = W1.T @ x_input + b1
        a_hidden = np.maximum(0, z_hidden)  # ReLU activation

        # Show calculation for first 3 neurons
        calc_group = VGroup()
        show_neurons = min(3, n_hidden)

        for i in range(show_neurons):
            w1_val = W1[0, i]
            w2_val = W1[1, i]
            b_val = b1[i]
            z_val = z_hidden[i]
            a_val = a_hidden[i]

            calc_text = Text(
                f"h{i+1}: z = {w1_val:.2f}×1 + {w2_val:.2f}×2 + {b_val:.2f} = {z_val:.2f}",
                font_size=18
            )
            relu_text = Text(
                f"    a = ReLU({z_val:.2f}) = {a_val:.2f}",
                font_size=18,
                color=GREEN if a_val > 0 else RED
            )

            neuron_calc = VGroup(calc_text, relu_text).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            calc_group.add(neuron_calc)

        calc_group.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        calc_group.move_to(UP * 0.5)

        self.play(*[FadeIn(calc) for calc in calc_group])
        self.wait(2)

        # Show ellipsis if there are more neurons
        if n_hidden > show_neurons:
            ellipsis = Text(f"... ({n_hidden - show_neurons} more neurons)", font_size=18, color=GRAY)
            ellipsis.next_to(calc_group, DOWN, buff=0.3)
            self.play(FadeIn(ellipsis))
            self.wait(1)

        # Step 3: Output layer
        self.play(*[FadeOut(mob) for mob in [step2_title, calc_group] + ([ellipsis] if n_hidden > show_neurons else [])])

        step3_title = Text("Step 3: Output Layer (Linear)", font_size=28, color=RED)
        step3_title.move_to(UP * 2.5)
        self.play(Write(step3_title))

        # Calculate output: y = W2^T @ a + b2
        y_output = W2.T @ a_hidden + b2
        prediction = y_output[0]

        # Show output calculation
        output_text = Text("Output calculation:", font_size=20)
        output_text.move_to(UP * 1.5)

        # Build equation showing sum of weighted activations
        sum_terms = []
        for i in range(min(3, n_hidden)):
            sum_terms.append(f"{W2[i, 0]:.2f}×{a_hidden[i]:.2f}")

        if n_hidden > 3:
            equation_str = f"y = {' + '.join(sum_terms)} + ... + {b2[0]:.2f}"
        else:
            equation_str = f"y = {' + '.join(sum_terms)} + {b2[0]:.2f}"

        equation = Text(equation_str, font_size=16)
        equation.next_to(output_text, DOWN, buff=0.3)

        result_text = Text(f"y = {prediction:.4f}", font_size=24, color=YELLOW)
        result_text.next_to(equation, DOWN, buff=0.5)

        self.play(Write(output_text))
        self.play(Write(equation))
        self.wait(1)
        self.play(Write(result_text))
        self.wait(1.5)

        # Comparison
        self.play(FadeOut(step3_title), FadeOut(output_text), FadeOut(equation))

        comparison_title = Text("Comparison", font_size=32, color=PURPLE)
        comparison_title.move_to(UP * 2)

        pred_line = Text(f"Predicted:  {prediction:.4f}", font_size=28)
        actual_line = Text(f"Actual:     {expected:.4f}", font_size=28)
        error = abs(prediction - expected)
        error_pct = (error / expected) * 100
        error_line = Text(f"Error:      {error:.4f} ({error_pct:.2f}%)", font_size=28,
                         color=GREEN if error_pct < 5 else YELLOW if error_pct < 10 else RED)

        comparison = VGroup(pred_line, actual_line, error_line)
        comparison.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        comparison.move_to(ORIGIN)

        self.play(FadeOut(result_text))
        self.play(Write(comparison_title))
        self.play(*[Write(line) for line in comparison])
        self.wait(3)

        # Final message
        if error_pct < 5:
            msg = Text("Excellent prediction! ✓", font_size=24, color=GREEN)
        elif error_pct < 10:
            msg = Text("Good prediction", font_size=24, color=YELLOW)
        else:
            msg = Text("Needs more training", font_size=24, color=RED)

        msg.to_edge(DOWN)
        self.play(FadeIn(msg))
        self.wait(2)


if __name__ == "__main__":
    # This allows running with: python animate_weights.py
    pass
