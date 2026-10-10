#pragma once
#include "world_view.h"
#include <vector>

class WorldManager {
public:
    WorldManager();
    ~WorldManager();

    void init();
    void cleanup();

    // Returns a WorldView with device pointers for passing into CUDA kernel
    WorldView getDeviceWorldView() const;

    // Returns a WorldView with host pointers for host-side physics / raycast
    WorldView getHostWorldView() const;

    // Raycast on host to find hit point for ray gun
    bool raycast(float3 ro, float3 rd, float max_dist, float3& out_hit) const;

    // Add a carve sphere and update GPU spatial grid
    void addCarve(float3 center, float radius);

    float getGroundHeight(float x, float z) const;
    int getCarveCount() const { return (int)edits_.size(); }

private:
    void rebuildGrid();

    WorldView host_world_;

    std::vector<float4> edits_;
    std::vector<int> cell_counts_;
    std::vector<int> cell_items_;

    // Device allocations
    float4* d_edits_ = nullptr;
    int* d_cell_counts_ = nullptr;
    int* d_cell_items_ = nullptr;
};
