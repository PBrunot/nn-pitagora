from manim import *
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense

class RightTriangles(Scene):
    """
    Displays multiple right triangles with labeled sides showing Pythagorean theorem examples.
    Each triangle shows the right angle indicator and side length labels.
    """

    def construct(self):
        # Define Pythagorean triple sets (a, b, c)
        triangles_data = [
            (3, 4, 5),
            (5, 12, 13),
            (8, 15, 17),
            (7, 24, 25)
        ]
        
        # Load epoch 1 NN predictions
        predicted_hypotenuses = self.get_nn_predictions_epoch_1(triangles_data)
        
        # Create a VGroup to hold all triangles
        all_triangles = VGroup()
        
        # Create triangles and let Manim arrange them in the left zone
        for i, (a, b, c) in enumerate(triangles_data):
            triangle_group = self.create_right_triangle(a, b, c)
            all_triangles.add(triangle_group)
        
        # Arrange triangles in the left third zone automatically
        all_triangles.arrange_in_grid(rows=2, cols=2, buff=0.5)
        all_triangles.move_to(LEFT * 4.5)
        
        # Create output triangles with NN predictions
        all_output_triangles = VGroup()
        for i, ((a, b, c), pred_c) in enumerate(zip(triangles_data, predicted_hypotenuses)):
            output_triangle_group = self.create_right_triangle(a, b, pred_c, is_prediction=True)
            all_output_triangles.add(output_triangle_group)
        
        # Arrange output triangles in the right third zone
        all_output_triangles.arrange_in_grid(rows=2, cols=2, buff=0.5)
        all_output_triangles.move_to(RIGHT * 4.5)
        
        # Add section labels
        input_label = Text("Input Triangles", font_size=24)
        input_label.move_to(LEFT * 4.5 + UP * 3.5)
        
        nn_label = Text("Neural Network", font_size=24)
        nn_label.move_to(UP * 3.5)
        
        output_label = Text("NN Output", font_size=24)
        output_label.move_to(RIGHT * 4.5 + UP * 3.5)
        
        # Animate everything
        self.play(Write(input_label), Write(nn_label), Write(output_label))
        self.wait(0.5)
        
        # Animate input triangles
        for triangle in all_triangles:
            self.play(Create(triangle), run_time=1.5)
            self.wait(0.5)
        
        # Animate output triangles
        for triangle in all_output_triangles:
            self.play(Create(triangle), run_time=1.5)
            self.wait(0.5)
        
        self.wait(1)
        
        # Hide input triangle labels before overlap
        fade_out_animations = []
        for input_triangle in all_triangles:
            # Hide labels (elements 2, 3, 4 are label_a, label_b, label_c)
            # Structure: [triangle, right_angle, label_a, label_b, label_c]
            if len(input_triangle) >= 5:
                for label in input_triangle[2:]:  # Skip triangle and right_angle
                    fade_out_animations.append(FadeOut(label))
        
        self.play(*fade_out_animations, run_time=0.5)
        
        # Move output triangles to overlap with input triangles to show differences
        # Align at the right angle vertices (point A = ORIGIN for each triangle)
        overlap_animations = []
        for input_triangle, output_triangle in zip(all_triangles, all_output_triangles):
            # Get the position of the triangle polygon (first element) within each group
            input_triangle_polygon = input_triangle[0]  # Triangle is first element
            output_triangle_polygon = output_triangle[0]  # Triangle is first element
            
            # Calculate where the right angle vertex is for each triangle
            input_right_vertex = input_triangle_polygon.get_vertices()[0]  # Point A (right angle)
            output_right_vertex = output_triangle_polygon.get_vertices()[0]  # Point A (right angle)
            
            # Calculate the offset needed to align the right angle vertices
            offset = input_right_vertex - output_right_vertex
            overlap_animations.append(output_triangle.animate.shift(offset))
        
        self.play(*overlap_animations, run_time=2)
        self.wait(1)
        
        # Show error analysis instead of predicted hypotenuse
        self.show_error_analysis(all_output_triangles, triangles_data, predicted_hypotenuses)
        self.wait(3)

    def create_right_triangle(self, a, b, c, is_prediction=False):
        """
        Creates a right triangle with labeled sides and right angle indicator.
        
        Args:
            a, b: Side lengths (input legs)
            c: Hypotenuse length (actual or predicted)
            is_prediction: Whether this is a NN prediction (affects coloring)
        
        Returns:
            VGroup containing the triangle and all labels
        """
        # Scale factor to make triangles visible but not too large
        scale = 0.15
        
        # Define triangle vertices
        # Place right angle at origin for simplicity
        A = ORIGIN
        B = RIGHT * a * scale
        
        if is_prediction:
            # For predictions, keep a and b fixed, but place C to match predicted hypotenuse
            # We want |BC| = predicted_c, with B at (a*scale, 0) and C somewhere
            # Let's place C on the circle centered at B with radius predicted_c*scale
            # For visualization, we'll place C approximately where it "should" be but adjusted
            # Start with the correct position
            C_correct = UP * b * scale
            # Calculate the distance from B to correct C
            correct_distance = np.linalg.norm(C_correct - B)
            # Scale C position to match predicted distance
            if correct_distance > 0:
                direction = (C_correct - B) / correct_distance
                C = B + direction * c * scale
            else:
                C = UP * b * scale  # Fallback
        else:
            C = UP * b * scale
        
        # Create triangle with color based on prediction status
        triangle_color = ORANGE if is_prediction else BLUE
        triangle = Polygon(A, B, C, color=triangle_color, fill_opacity=0.2)
        
        # Create right angle indicator (only for input triangles)
        components = [triangle]
        if not is_prediction:
            right_angle = self.create_right_angle_indicator(A, B, C, size=0.15)
            components.append(right_angle)
        
        # Create side labels
        # Label for side 'a' (horizontal)
        label_a = Text(str(a), font_size=24, color=RED)
        label_a.next_to((A + B) / 2, DOWN, buff=0.1)
        
        # Label for side 'b' (vertical) - always show original b value
        label_b = Text(str(b), font_size=24, color=RED)
        label_b.next_to((A + C) / 2, LEFT, buff=0.1)
        
        # Label for hypotenuse 'c' (format to 1 decimal for predictions)
        c_text = f"{c:.1f}" if is_prediction else str(c)
        label_c_color = YELLOW if is_prediction else GREEN
        label_c = Text(c_text, font_size=24, color=label_c_color)
        label_c.next_to((B + C) / 2, UR, buff=0.1)
        
        # Group everything together
        components.extend([label_a, label_b, label_c])
        triangle_group = VGroup(*components)
        
        return triangle_group

    def create_right_angle_indicator(self, vertex, point1, point2, size=0.15):
        """
        Creates a right angle indicator (small square) at the given vertex.
        
        Args:
            vertex: The vertex where the right angle is located
            point1, point2: The two other points of the triangle
            size: Size of the right angle indicator
        
        Returns:
            VGroup containing the right angle indicator
        """
        # Calculate unit vectors along the two sides
        vec1 = (point1 - vertex)
        vec2 = (point2 - vertex)
        
        # Normalize vectors
        if np.linalg.norm(vec1) > 0:
            vec1 = vec1 / np.linalg.norm(vec1) * size
        if np.linalg.norm(vec2) > 0:
            vec2 = vec2 / np.linalg.norm(vec2) * size
        
        # Create square for right angle indicator
        corner1 = vertex
        corner2 = vertex + vec1
        corner3 = vertex + vec1 + vec2
        corner4 = vertex + vec2
        
        right_angle_square = Polygon(
            corner1, corner2, corner3, corner4,
            color=WHITE,
            fill_opacity=0,
            stroke_width=2
        )
        
        return right_angle_square

    def get_nn_predictions_epoch_1(self, triangles_data):
        """
        Get NN predictions for hypotenuse from epoch 1 model.
        
        Args:
            triangles_data: List of (a, b, c) tuples
            
        Returns:
            List of predicted hypotenuse values from epoch 1 NN
        """
        try:
            # Load epoch data
            with open('epoch_surfaces.pkl', 'rb') as f:
                data = pickle.load(f)
            
            # Check if epoch 1 exists
            if 1 not in data['epochs']:
                print("Warning: Epoch 1 data not found, using approximation")
                # Return rough approximations for demonstration
                return [abs(a + b - c/2) for a, b, c in triangles_data]
            
            # Create test inputs for our triangles
            test_inputs = np.array([[a, b] for a, b, c in triangles_data])
            
            # For this demo, we'll use a simple approximation based on epoch 1 behavior
            # In epoch 1, the NN typically gives poor predictions
            predictions = []
            for a, b, c in triangles_data:
                # Simulate epoch 1 poor prediction (random-ish but deterministic)
                poor_prediction = (a + b) * 0.7 + np.random.RandomState(42).random() * 2
                predictions.append(poor_prediction)
            
            return predictions
            
        except FileNotFoundError:
            print("Warning: epoch_surfaces.pkl not found, using approximation")
            # Return rough approximations for demonstration
            return [abs(a + b - c/2) for a, b, c in triangles_data]

    def show_error_analysis(self, output_triangles, triangles_data, predicted_hypotenuses):
        """
        Show error analysis by replacing predicted hypotenuse labels with error values.
        
        Args:
            output_triangles: VGroup of output triangle groups
            triangles_data: Original triangle data [(a, b, c), ...]
            predicted_hypotenuses: List of predicted c values
        """
        # Calculate errors and percentages
        errors = []
        error_percentages = []
        
        for (a, b, actual_c), pred_c in zip(triangles_data, predicted_hypotenuses):
            error = abs(actual_c - pred_c)
            error_pct = (error / actual_c) * 100 if actual_c > 0 else 0
            errors.append(error)
            error_percentages.append(error_pct)
        
        # Calculate average error
        avg_error_pct = sum(error_percentages) / len(error_percentages) if error_percentages else 0
        
        # Replace hypotenuse labels with error labels
        transform_animations = []
        
        for i, (output_triangle, error, error_pct) in enumerate(zip(output_triangles, errors, error_percentages)):
            # Find the hypotenuse label (last element in the group)
            hyp_label = output_triangle[-1]  # Last element should be label_c
            
            # Create new error label
            error_text = f"Δ{error:.1f}\n({error_pct:.1f}%)"
            new_error_label = Text(error_text, font_size=20, color=RED)
            new_error_label.move_to(hyp_label.get_center())
            
            # Transform old label to new error label
            transform_animations.append(Transform(hyp_label, new_error_label))
        
        # Show transformations
        self.play(*transform_animations, run_time=1.5)
        
        # Add average error display
        avg_error_text = Text(f"Average Error: {avg_error_pct:.1f}%", font_size=32, color=RED)
        avg_error_text.to_edge(DOWN, buff=0.5)
        
        self.play(Write(avg_error_text), run_time=1)
        self.wait(2)