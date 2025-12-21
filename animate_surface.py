from manim import *
import numpy as np
import pickle

class EvolvingNNSurfaceSimple(ThreeDScene):
    """
    Animates the neural network learning to approximate f(a,b) = sqrt(a^2 + b^2)
    Shows progression through epochs 1, 3, 10, 30, 50 with smooth morphing animation.
    """

    def construct(self):
        # Load epoch data
        with open('epoch_surfaces.pkl', 'rb') as f:
            data = pickle.load(f)

        a_range = data['a_range']
        b_range = data['b_range']
        Z_actual = data['Z_actual']
        all_epochs = sorted(data['epochs'].keys())

        # Select specific epochs to show: 1, 3, 10, 30, 50
        target_epochs = [1, 3, 10, 30, 50]
        # Filter to only include epochs that exist in the data
        epochs = [e for e in target_epochs if e in all_epochs]

        # Scale for visualization - larger scale fills screen better
        xy_scale = 0.6
        z_scale = 0.6  # Same scale for all axes

        # Set up camera for optimal viewing
        # phi=60 gives a lower angle that emphasizes the curvature of the surface
        # theta=-110 provides a view similar to the reference screenshot
        # Distance is set closer to fill more of the screen
        self.set_camera_orientation(
            phi=60 * DEGREES,
            theta=-110 * DEGREES,
            distance=8  # Closer camera for larger view
        )
        self.begin_ambient_camera_rotation(rate=0.06)  # Slower rotation for better viewing

        # Add axes with different z-scale, optimized ranges
        axes = ThreeDAxes(
            x_range=[0, 10 * xy_scale, 2 * xy_scale],
            y_range=[0, 10 * xy_scale, 2 * xy_scale],
            z_range=[0, 14 * z_scale, 3 * z_scale],  # Adjusted for better fit
            x_length=10 * xy_scale,
            y_length=10 * xy_scale,
            z_length=14 * z_scale
        )

        labels = axes.get_axis_labels(
            x_label=MathTex("a").scale(0.8),
            y_label=MathTex("b").scale(0.8),
            z_label=MathTex("f(a,b)=\\sqrt{a^2+b^2}").scale(0.7)
        )

        # Title
        title = Text("NN Learning: Pythagorean Theorem", font_size=36)
        title.to_corner(UP)

        self.add_fixed_in_frame_mobjects(title)
        self.play(Create(axes), Write(labels), Write(title))

        # Create actual surface (blue wireframe) with amplified z
        def actual_func(u, v):
            a = u * 10
            b = v * 10
            z = np.sqrt(a**2 + b**2)
            return np.array([a * xy_scale, b * xy_scale, z * z_scale])

        actual_surface = Surface(
            actual_func,
            u_range=[0, 1],
            v_range=[0, 1],
            resolution=(30, 30),  # Higher resolution for smoother surface
            fill_opacity=0.15,  # More transparent to see NN surface through it
            stroke_width=0.5,  # Thinner stroke to not obscure view
            stroke_color=BLUE,
            fill_color=BLUE,
            stroke_opacity=0.6
        )

        self.play(Create(actual_surface), run_time=2)
        self.wait(2)

        # Create surfaces for selected epochs
        surfaces = []
        for epoch in epochs:
            Z_pred = data['epochs'][epoch]

            # Interpolate surface with amplified z
            def make_surface_func(Z_data, a_r, b_r, z_scl):
                def surf_func(u, v):
                    # Bilinear interpolation
                    x_idx = u * (len(a_r) - 1)
                    y_idx = v * (len(b_r) - 1)

                    x0, x1 = int(x_idx), min(int(x_idx) + 1, len(a_r) - 1)
                    y0, y1 = int(y_idx), min(int(y_idx) + 1, len(b_r) - 1)

                    fx, fy = x_idx - x0, y_idx - y0

                    z = (1-fx)*(1-fy)*Z_data[y0,x0] + fx*(1-fy)*Z_data[y0,x1] + \
                        (1-fx)*fy*Z_data[y1,x0] + fx*fy*Z_data[y1,x1]

                    a = u * 10
                    b = v * 10
                    return np.array([a * xy_scale, b * xy_scale, z * z_scl])
                return surf_func

            # Color interpolation from red to green based on progress
            color_progress = epochs.index(epoch) / (len(epochs) - 1) if len(epochs) > 1 else 0
            surf_color = interpolate_color(RED, GREEN, color_progress)

            surf = Surface(
                make_surface_func(Z_pred, a_range, b_range, z_scale),
                u_range=[0, 1],
                v_range=[0, 1],
                resolution=(30, 30),  # Higher resolution for smoother surface
                fill_opacity=0.65,  # Slightly more opaque to stand out
                stroke_width=0.8,  # Visible wireframe
                stroke_color=surf_color,
                fill_color=surf_color,
                stroke_opacity=0.8
            )
            surfaces.append((epoch, surf))

        # Animate transitions
        current_surf = surfaces[0][1]
        epoch_label = Text(f"Epoch {surfaces[0][0]}", font_size=32, color=YELLOW).to_corner(UP + LEFT)

        self.add_fixed_in_frame_mobjects(epoch_label)
        self.play(Create(current_surf), Write(epoch_label))
        self.wait(2)  # Longer pause to view initial poor approximation

        for i in range(1, len(surfaces)):
            epoch, next_surf = surfaces[i]
            new_label = Text(f"Epoch {epoch}", font_size=32, color=YELLOW).to_corner(UP + LEFT)

            # Remove old label from fixed frame objects and fade it out
            self.remove(epoch_label)

            # Add new label and animate transformation
            self.add_fixed_in_frame_mobjects(new_label)
            self.play(
                Transform(current_surf, next_surf),
                FadeOut(epoch_label),
                FadeIn(new_label),
                run_time=3  # Slightly longer to appreciate the morphing
            )

            # Update label reference for next iteration
            epoch_label = new_label

            self.wait(2)  # Pause to view each epoch's result

        # Final message
        final_text = Text("Training Complete!", font_size=24, color=GREEN)
        final_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(final_text)
        self.play(Write(final_text))

        self.stop_ambient_camera_rotation()
        self.wait(3)
