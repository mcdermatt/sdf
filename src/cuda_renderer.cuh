#pragma once
#include <cuda_runtime.h>
#include "camera.h"

void launch_cuda_render(
    uchar4* d_output,
    int width,
    int height,
    const CameraUniforms& cam,
    float time,
    int scene_id
);
