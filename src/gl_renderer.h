#pragma once
#include <GL/glew.h>
#include <cuda_runtime.h>
#include "camera.h"

class GLRenderer {
public:
    GLRenderer();
    ~GLRenderer();

    bool init(int width, int height);
    void resize(int width, int height);
    void render(const CameraUniforms& cam, float time, int scene_id);
    void cleanup();

private:
    void initQuadShader();

    int width_ = 0;
    int height_ = 0;
    GLuint texture_ = 0;
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    GLuint shader_program_ = 0;

    uchar4* d_pixels_ = nullptr; // CUDA device buffer on GPU
    uchar4* h_pixels_ = nullptr; // CUDA pinned host memory for high-speed PCIe copy
};
