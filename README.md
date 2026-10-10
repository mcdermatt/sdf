# CUDA Destructible SDF Outdoor World

A real-time Signed Distance Field (SDF) renderer and destructible physics sandbox running on an NVIDIA GeForce RTX 3090, featuring procedural rolling terrain, soil strata, foliage, a log cabin, physical volumetric clouds at sunset, and an interactive ray gun.

---

## Features

- **Procedural Rolling Hills**: Multi-octave Perlin noise gradient heightfield blended into an organic landscape.
- **Stratified Soil Layers**:
  - **Grass**: Top layer (~1 inch thick), dynamically omitted on slopes steeper than $30^\circ$.
  - **Dirt**: Sub-surface soil (4 to 6 inches thick), eroded away on slopes steeper than $40^\circ$.
  - **Rock**: Bedrock underlying the dirt and exposed on steep cliffs and in blast craters.
- **Log Cabin & Pine Forest**:
  - Authentic interlocking log courses, gable roof, plank door, and fieldstone chimney.
  - Multi-tier procedural pine trees surrounding the cabin clearing.
- **Destructible Terrain via Ray Gun**:
  - Visible first-person sci-fi ray gun on the right side of the screen with recoil and walk bobbing.
  - Left-clicking fires a high-energy laser beam and smoothly carves out a crater directly in the analytical SDF.
  - Craters immediately reveal the grass $\rightarrow$ dirt $\rightarrow$ rock strata along the crater walls.
  - Real-time $128 \times 128$ spatial grid on the GPU ensures continuous 60+ FPS performance even with many carved holes.
- **Physical Volumetric Clouds & Sunset Skybox**:
  - Sunset sun low on the western horizon casting warm golden light and elongated soft shadows.
  - Volumetric cloud layer raymarched with light marching toward the sun for self-shadowing and silver lining (Mie forward scattering).
- **First-Person Controller**:
  - Captured mouse look (un-inverted yaw & pitch).
  - WASD surface walking with terrain collision.
  - Left Shift sprint and Spacebar jump with gravity.

---

## Controls

| Input | Action |
| :--- | :--- |
| **Mouse Move** | Look around (first-person camera, un-inverted) |
| **Left Click** | Fire Ray Gun & carve craters into the SDF |
| **W / A / S / D** | Walk along the ground |
| **Left Shift** | Sprint |
| **Spacebar** | Jump |
| **R** | Respawn at the cabin clearing |
| **Esc** | Release mouse cursor / Exit |

---

## Building & Running

```bash
# Build
cmake -B build -S .
cmake --build build -j

# Run
./build/cuda_sdf_viewer
```
