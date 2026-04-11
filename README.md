# Synthetic Vision Calibrator

An ultra high-performance, real-time 3D camera calibration simulator built for Apple Silicon. This engine uses parallel processing and mathematically accurate 3D rigid body projections to simulate camera distortion, sensor noise, and perspective warping, allowing you to visualize and understand OpenCV's core camera calibration matrix extraction in real-time.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)

---

## 🔥 Features
- **60 FPS Pre-Computation Engine**: Harnesses the `ThreadPoolExecutor` to multi-thread the complex intrinsic matrix math across all available cores before playback begins, preventing UI stuttering.
- **True 3D Physics Projection**: Casts a flawless checkerboard into absolute `Z=0` 3D space, and mathematically simulates Pitch, Yaw, Roll, and Z-Axis zooms using `cv2.projectPoints()` rather than simple 2D stretching.
- **Cyber-HUD Dashboard**: Includes a dynamic wireframe rendering and live, color-coded telemetry stream so the audience can see exactly how 3D world coordinates are mapped to 2D pixel sensors.
- **Interactive Matrix Dashboards**: Uses a "Presenter Pause" gatekeeper before calculating the final intrinsic `[K]` matrix, distortion coefficients, and global system Reprojection Error.
- **Hacker-Style "Digital Twin" Mapping**: Creates a randomized visual memory bank representation mapping exactly to the real-time arrays being populated by `findChessboardCorners()`.

## 🛠 Prerequisites

Run the following command to ensure you have the required computer vision dependencies installed:
```bash
pip install opencv-python numpy
```

## 🚀 Execution & Command Line Hooks

The environment exposes several physics and noise traits through a Command Line Interface. You can tweak these values to see precisely how they propagate through the simulation and impact the final Reprojection Error.

```bash
python animation.py [OPTIONS]
```

### Options:
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--noise` | Float | `4.0` | Controls the variance of the Gaussian distribution injected into the sensor. Increasing this makes corner detection radically more volatile. |
| `--focal` | Float | `1200.0` | The simulated intrinsic focal length `(fx, fy)`. Overwriting this fundamentally alters the generated matrix. |
| `--speed` | Float | `1.0` | Multiplies the pan/tilt/zoom vector of the camera. Higher speeds cause aggressive perspective shifts but reduce the number of high-quality flat scans. |

### Example CLI Experiments:
**High-Noise Stress Test** (Watch the Reprojection Error skyrocket):
```bash
python animation.py --noise 15.0 
```

**Macro-Lens Cinematic Mode**:
```bash
python animation.py --focal 2400.0 --speed 0.5
```

## 🎮 Controls
* **`ESC` / `Q`**: Abort the simulation instantly.
* **`E`**: Proceed past the Data Collection dashboard to extract the final matrix and distortion outputs.

---
*Developed for Advanced Agentic UI Visualization.*
