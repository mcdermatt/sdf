#include "world_manager.h"
#include "sdf_common.cuh"
#include <iostream>
#include <algorithm>
#include <cmath>

WorldManager::WorldManager() {}

WorldManager::~WorldManager() {
    cleanup();
}

void WorldManager::init() {
    // 1. Cabin position
    host_world_.cabin_x = 0.0f;
    host_world_.cabin_z = 0.0f;
    // Query initial raw terrain height to place cabin pad
    host_world_.cabin_y = terrainRaw(0.0f, 0.0f);

    // 2. Scatter trees naturally around the cabin
    // Distribute 14 trees in a grove around the clearing
    host_world_.num_trees = 14;

    const float tree_offsets[14][2] = {
        { -12.5f,  -8.0f },
        { -15.0f,   6.5f },
        {  13.0f,  -9.5f },
        {  14.5f,   7.0f },
        {  -8.0f,  16.0f },
        {   9.0f,  15.5f },
        {  -7.5f, -17.0f },
        {   8.5f, -18.0f },
        { -21.0f,   1.0f },
        {  22.0f,  -2.0f },
        { -18.0f, -14.0f },
        {  19.0f,  13.0f },
        { -17.0f,  17.0f },
        {  18.0f, -15.0f }
    };

    for (int i = 0; i < host_world_.num_trees; ++i) {
        float tx = host_world_.cabin_x + tree_offsets[i][0];
        float tz = host_world_.cabin_z + tree_offsets[i][1];
        float ty = terrainHeight(tx, tz, host_world_);

        // Semi-random tree height and foliage radius based on hash
        float rand_h = hash2f((int)(tx * 10.0f), (int)(tz * 10.0f));
        float h = 6.0f + rand_h * 3.5f;     // 6m to 9.5m tall
        float r = 1.6f + rand_h * 0.6f;     // 1.6m to 2.2m radius

        host_world_.trees[i].x = tx;
        host_world_.trees[i].y = ty;
        host_world_.trees[i].z = tz;
        host_world_.trees[i].height = h;
        host_world_.trees[i].radius = r;
    }

    // 3. Allocate Spatial Hash Grid for Carve Edits
    cell_counts_.assign(EDIT_GRID_N * EDIT_GRID_N, 0);
    cell_items_.assign(EDIT_GRID_N * EDIT_GRID_N * EDIT_CELL_CAP, 0);

    // Allocate GPU memory
    cudaMalloc(&d_edits_, MAX_EDITS * sizeof(float4));
    cudaMalloc(&d_cell_counts_, EDIT_GRID_N * EDIT_GRID_N * sizeof(int));
    cudaMalloc(&d_cell_items_, EDIT_GRID_N * EDIT_GRID_N * EDIT_CELL_CAP * sizeof(int));

    cudaMemset(d_cell_counts_, 0, EDIT_GRID_N * EDIT_GRID_N * sizeof(int));
}

void WorldManager::cleanup() {
    if (d_edits_) { cudaFree(d_edits_); d_edits_ = nullptr; }
    if (d_cell_counts_) { cudaFree(d_cell_counts_); d_cell_counts_ = nullptr; }
    if (d_cell_items_) { cudaFree(d_cell_items_); d_cell_items_ = nullptr; }
}

WorldView WorldManager::getDeviceWorldView() const {
    WorldView dev = host_world_;
    dev.edits = d_edits_;
    dev.cell_counts = d_cell_counts_;
    dev.cell_items = d_cell_items_;
    return dev;
}

WorldView WorldManager::getHostWorldView() const {
    WorldView host = host_world_;
    host.edits = edits_.data();
    host.cell_counts = cell_counts_.data();
    host.cell_items = cell_items_.data();
    return host;
}

float WorldManager::getGroundHeight(float x, float z) const {
    // Exact terrain height plus check if a carve hollowed out beneath it
    float h = terrainHeight(x, z, host_world_);
    WorldView hw = getHostWorldView();
    // Step downward from initial terrain surface to find true ground if carved
    for (float y = h + 0.1f; y >= h - 12.0f; y -= 0.15f) {
        float d = mapWorld(make_float3(x, y, z), hw);
        if (d <= 0.05f) {
            return y;
        }
    }
    return h;
}

bool WorldManager::raycast(float3 ro, float3 rd, float max_dist, float3& out_hit) const {
    WorldView hw = getHostWorldView();
    float t = 0.25f;

    for (int i = 0; i < 220 && t < max_dist; ++i) {
        float3 pos = ro + rd * t;
        float d = mapWorld(pos, hw);

        if (d < 0.005f * (1.0f + t * 0.02f)) {
            out_hit = pos;
            return true;
        }
        t += fmaxf(d, 0.04f);
    }
    return false;
}

void WorldManager::addCarve(float3 center, float radius) {
    if (edits_.size() >= MAX_EDITS) return;

    int new_idx = (int)edits_.size();
    float4 new_edit = make_float4(center.x, center.y, center.z, radius);
    edits_.push_back(new_edit);

    // Update spatial grid cells affected by this sphere
    float eff_r = radius + CARVE_SMOOTH;
    int min_cx = std::clamp((int)floorf((center.x - eff_r - EDIT_GRID_MIN) / EDIT_CELL), 0, EDIT_GRID_N - 1);
    int max_cx = std::clamp((int)floorf((center.x + eff_r - EDIT_GRID_MIN) / EDIT_CELL), 0, EDIT_GRID_N - 1);
    int min_cz = std::clamp((int)floorf((center.z - eff_r - EDIT_GRID_MIN) / EDIT_CELL), 0, EDIT_GRID_N - 1);
    int max_cz = std::clamp((int)floorf((center.z + eff_r - EDIT_GRID_MIN) / EDIT_CELL), 0, EDIT_GRID_N - 1);

    for (int cz = min_cz; cz <= max_cz; ++cz) {
        for (int cx = min_cx; cx <= max_cx; ++cx) {
            int cell = cz * EDIT_GRID_N + cx;
            int count = cell_counts_[cell];
            if (count < EDIT_CELL_CAP) {
                cell_items_[cell * EDIT_CELL_CAP + count] = new_idx;
                cell_counts_[cell]++;
            }
        }
    }

    // Upload new edit and updated cell grid to GPU
    cudaMemcpy(d_edits_ + new_idx, &new_edit, sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cell_counts_, cell_counts_.data(), EDIT_GRID_N * EDIT_GRID_N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cell_items_, cell_items_.data(), EDIT_GRID_N * EDIT_GRID_N * EDIT_CELL_CAP * sizeof(int), cudaMemcpyHostToDevice);
}
