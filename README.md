# CUDA SDF Real-Time Raymarcher

A high-performance C++20 and CUDA Signed Distance Field (SDF) raymarcher running on NVIDIA RTX hardware, rendered through an OpenGL/GLFW interactive window with full orbit camera controls.

---

## Features

- **Raymarching (Sphere Tracing)**: Mathematically exact, pixel-perfect rendering directly from analytical distance functions.
- **CUDA Kernel Acceleration**: Parallel per-pixel ray evaluation evaluated across GPU streaming multiprocessors (Ampere `sm_86`).
- **Surface Normals & Shading**:
  - Analytical normals computed via central differences ($\nabla f(\mathbf{p})$).
  - Blinn-Phong specular highlighting and ambient lighting.
  - Soft raymarched shadows with penumbra estimation.
  - Ambient occlusion along surface normal vectors.
  - Reinhard tone mapping and gamma correction ($2.2$).
- **Interactive Orbit Camera**:
  - Smooth mouse drag orbit rotation (azimuth / elevation).
  - Mouse wheel zoom.
  - Right-click camera panning.
- **Multiple Switchable Scenes**:
  1. **Scene 1**: Morphing smooth CSG (Sphere + Box + Torus blend with polynomial smooth minimum).
  2. **Scene 2**: Mechanical CSG (Cube + Sphere intersection with orthogonal cylindrical drills).
  3. **Scene 3**: Mathematical Lattice (Triply periodic minimal surface / Gyroid bounded inside a sphere).

---

## Controls

| Key / Mouse Action | Function |
| :--- | :--- |
| **Left Click + Drag** | Orbit rotate camera |
| **Right Click + Drag** | Pan camera target |
| **Scroll Wheel** | Zoom in / out |
| **Keys `1`, `2`, `3`** | Switch between SDF scenes |
| **Spacebar** | Pause / resume animation |
| **Key `R`** | Reset camera to default view |
| **Esc** | Exit application |

---

## Building & Running

### Prerequisites

- NVIDIA GPU with CUDA Toolkit ($\ge 12.0$)
- C++20 compatible compiler (`g++` or `clang++`)
- CMake ($\ge 3.22$)
- OpenGL, GLEW, GLFW3, GLM

```bash
# Ubuntu / WSL dependencies
sudo apt update
sudo apt install -y build-essential cmake libglfw3-dev libglew-dev libglm-dev libgl1-mesa-dev
```

### Build

```bash
mkdir -p build
cmake -B build -S .
cmake --build build -j
```

### Run

```bash
./build/cuda_sdf_viewer
```
