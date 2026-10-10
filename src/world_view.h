#pragma once
// Plain-old-data description of the world. Passed BY VALUE into the CUDA kernel
// (kernel parameters live in constant memory, so tree/cabin data is broadcast to
// every thread for free) and also used by host-side physics.
#include <cuda_runtime.h>

constexpr int   MAX_TREES     = 16;

// Ray-gun carve edits are bucketed into a 2D grid over the XZ plane so each SDF
// evaluation only has to test the handful of carves near that point.
constexpr int   EDIT_GRID_N   = 128;                    // cells per side
constexpr float EDIT_CELL     = 4.0f;                   // meters per cell
constexpr float EDIT_GRID_MIN = -0.5f * EDIT_GRID_N * EDIT_CELL;
constexpr int   EDIT_CELL_CAP = 64;                     // max carves referenced per cell
constexpr int   MAX_EDITS     = 16384;
constexpr float CARVE_SMOOTH  = 0.08f;                  // smooth-subtraction blend (m)

struct TreeDesc {
    float x, y, z;     // base position (y = ground height)
    float height;
    float radius;      // foliage radius at the bottom cone
};

struct WorldView {
    TreeDesc trees[MAX_TREES];
    int num_trees;
    float cabin_x, cabin_y, cabin_z;   // cabin_y = flattened ground pad height

    // Carve edits (device pointers on the GPU, host pointers on the CPU).
    const float4* edits;               // xyz = center, w = radius
    const int* cell_counts;            // EDIT_GRID_N^2
    const int* cell_items;             // EDIT_GRID_N^2 * EDIT_CELL_CAP edit indices
};
