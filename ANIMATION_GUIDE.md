# 3D Surface Animation Guide

## Interactive Plotly Visualization

### Usage
1. Run the training script:
   ```bash
   source .venv/bin/activate
   python nn_pitagora.py
   ```

2. Open `3d_comparison_epochs.html` in your browser

### Features
- **Blue surface**: Actual function f(a,b) = √(a² + b²)
- **Colored surfaces**: NN predictions at different epochs with gradient from red (early) to green (converged)
- **Interactive controls**:
  - Click and drag to rotate
  - Scroll to zoom
  - Click legend items to toggle surfaces on/off

## Manim 3D Surface Animation

### Installation
Install Manim in your virtual environment:
```bash
source .venv/bin/activate
pip install manim
```

### Generate Animation

Smoother morphing animation showing epochs 1, 3, 10, 30, 50 with rotating camera:
```bash
manim -pqh animate_surface.py EvolvingNNSurfaceSimple
```

### Command Flags
- `-p`: Play the video after rendering
- `-q`: Quality
  - `-ql`: Low quality (480p, fast)
  - `-qm`: Medium quality (720p)
  - `-qh`: High quality (1080p)
  - `-qk`: 4K quality (2160p)

### Output
Videos are saved in `media/videos/animate_surface/1080p60/`

## Animation Features

### EvolvingNNSurfaceSimple
- **Optimized camera positioning** for maximum surface visibility
  - 60° angled view emphasizes surface curvature
  - -110° theta angle for optimal perspective
  - Closer camera distance fills screen
  - Slow rotation (0.06 rate) for easy viewing
- **Blue semi-transparent surface**: Target function (opacity 0.15)
- **Colored evolving surface**: NN prediction morphing through epochs (opacity 0.65)
- **Color gradient**: Red (epoch 1) → Green (epoch 50) showing learning progress
- **Smooth transitions** between epochs 1 → 3 → 10 → 30 → 50 (3 seconds each)
- **Proportional scaling** across all axes for accurate representation
- **Higher resolution surfaces** (30×30) for smooth appearance
- **Clean epoch counter** (yellow, size 32) - old text fades out before new appears
- **Large surfaces** filling most of screen for clear visibility
- Ideal for presentations and demonstrations

## How the NN Learning Process Works

### Epoch 1 (Dark Red)
- Random initial weights
- Poor approximation, surface is far from target
- High error across the entire domain

### Epoch 3 (Red-Orange)
- Weights starting to adjust
- Surface beginning to take shape
- Error still significant

### Epoch 10 (Orange-Yellow)
- Weights have learned basic pattern
- Surface conforming to correct shape
- Error decreasing noticeably

### Epoch 30 (Yellow-Green)
- Substantial learning complete
- Surface closely follows target
- Minor deviations remain

### Epoch 50 (Green)
- Well-trained network
- Surface nearly overlaps with actual function
- Minimal error, excellent generalization

## Customization

### Different Epochs
Edit `nn_pitagora.py` line 54:
```python
epochs_to_plot = [1, 3, 10, 30, 50]  # Change these values
```

The animation will automatically use epochs 1, 3, 10, 30, 50 (or whatever exists in the data).

### Overall Scale
Edit `animate_surface.py` lines 27-28 to change surface size:
```python
xy_scale = 0.6  # Change to make surfaces larger/smaller
z_scale = 0.6   # Should match xy_scale for proportional representation
```

Note: For accurate representation, keep xy_scale and z_scale equal. You can amplify the z-axis for dramatic effect, but this will distort the true shape of the function.

### Camera Position
Edit camera settings in `animate_surface.py` (lines 34-38):
```python
self.set_camera_orientation(
    phi=60 * DEGREES,     # Vertical angle (higher = more top-down)
    theta=-110 * DEGREES,  # Horizontal angle
    distance=8            # Lower = closer (larger view)
)
```

### Camera Rotation Speed
Edit rotation rate in `animate_surface.py` (line 40):
```python
self.begin_ambient_camera_rotation(rate=0.06)  # Lower = slower
```

### Animation Speed
Edit `animate_surface.py` transition time (line 153):
```python
self.play(Transform(...), run_time=3)  # Adjust duration
```

### Surface Opacity
Edit opacity values in `animate_surface.py`:
```python
# Actual surface (line 77)
fill_opacity=0.15  # Target function transparency

# NN surface (line 121)
fill_opacity=0.65  # NN prediction transparency
```

### Resolution
Edit `resolution` parameter for surface quality vs performance:
```python
resolution=(30, 30)  # Lower = faster, Higher = smoother (lines 76, 120)
```

## Tips for Best Results

1. **Run training first**: Always run `nn_pitagora.py` before creating animations to generate `epoch_surfaces.pkl`

2. **Start with low quality**: Test with `-ql` flag first to preview quickly:
   ```bash
   manim -pql animate_surface.py EvolvingNNSurfaceSimple
   ```

3. **Camera positioning is optimized**: The 60° phi angle and -110° theta angle with distance=8 emphasize surface curvature

4. **Proportional scaling**: All axes use 0.6 scale for accurate representation of the function's true shape

5. **Surfaces fill the screen**: The 0.6 scale with closer camera makes surfaces occupy most of the frame

6. **Slower rotation helps**: The 0.06 rotation rate allows viewers to appreciate the morphing without motion blur

7. **Longer transitions show detail**: 3-second morphs between epochs let viewers see the gradual convergence

8. **Contrast is optimized**: Blue target (15% opacity) vs colored NN surface (65% opacity) clearly shows the gap

9. **No text overlap**: The animation properly fades out old epoch labels before showing new ones

10. **Yellow epoch labels stand out**: High-contrast yellow text is visible against all surface colors
