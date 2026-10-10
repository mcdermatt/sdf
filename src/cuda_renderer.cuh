#pragma once
#include <cuda_runtime.h>
#include "world_view.h"
#include "frame_params.h"

void launch_cuda_render(
    uchar4* d_output,
    int width,
    int height,
    const FrameParams& params,
    const WorldView& world
);
