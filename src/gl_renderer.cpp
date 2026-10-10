#include "gl_renderer.h"
#include "cuda_renderer.cuh"
#include <iostream>

static const char* vertex_shader_src = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

static const char* fragment_shader_src = R"(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D screenTexture;

void main() {
    FragColor = texture(screenTexture, TexCoord);
}
)";

GLRenderer::GLRenderer() {}

GLRenderer::~GLRenderer() {
    cleanup();
}

void GLRenderer::initQuadShader() {
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertex_shader_src, nullptr);
    glCompileShader(vs);

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragment_shader_src, nullptr);
    glCompileShader(fs);

    shader_program_ = glCreateProgram();
    glAttachShader(shader_program_, vs);
    glAttachShader(shader_program_, fs);
    glLinkProgram(shader_program_);

    glDeleteShader(vs);
    glDeleteShader(fs);

    float quad_vertices[] = {
        // positions   // texcoords
        -1.0f,  1.0f,  0.0f, 0.0f,
        -1.0f, -1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 1.0f,

        -1.0f,  1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 1.0f,
         1.0f,  1.0f,  1.0f, 0.0f
    };

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    glBindVertexArray(0);
}

bool GLRenderer::init(int width, int height) {
    width_ = width;
    height_ = height;

    initQuadShader();

    // Setup OpenGL texture
    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Allocate GPU Device Memory
    size_t buffer_size = width_ * height_ * sizeof(uchar4);
    cudaError_t err = cudaMalloc(&d_pixels_, buffer_size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Allocate High-Speed Pinned Host Memory for DMA transfers
    err = cudaHostAlloc(&h_pixels_, buffer_size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        std::cerr << "cudaHostAlloc failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}

void GLRenderer::resize(int width, int height) {
    if (width <= 0 || height <= 0 || (width == width_ && height == height_)) return;

    width_ = width;
    height_ = height;

    if (d_pixels_) {
        cudaFree(d_pixels_);
        d_pixels_ = nullptr;
    }
    if (h_pixels_) {
        cudaFreeHost(h_pixels_);
        h_pixels_ = nullptr;
    }

    size_t buffer_size = width_ * height_ * sizeof(uchar4);
    cudaMalloc(&d_pixels_, buffer_size);
    cudaHostAlloc(&h_pixels_, buffer_size, cudaHostAllocDefault);

    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void GLRenderer::render(const FrameParams& params, const WorldView& world) {
    // 1. Run CUDA Raymarching Kernel on RTX 3090
    launch_cuda_render(d_pixels_, width_, height_, params, world);

    // 2. High-speed DMA copy to host pinned buffer
    size_t buffer_size = width_ * height_ * sizeof(uchar4);
    cudaMemcpy(h_pixels_, d_pixels_, buffer_size, cudaMemcpyDeviceToHost);

    // 3. Upload texture to OpenGL
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, h_pixels_);

    // 4. Blit to screen
    glViewport(0, 0, width_, height_);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shader_program_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glUniform1i(glGetUniformLocation(shader_program_, "screenTexture"), 0);

    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void GLRenderer::cleanup() {
    if (d_pixels_) {
        cudaFree(d_pixels_);
        d_pixels_ = nullptr;
    }
    if (h_pixels_) {
        cudaFreeHost(h_pixels_);
        h_pixels_ = nullptr;
    }
    if (texture_) {
        glDeleteTextures(1, &texture_);
        texture_ = 0;
    }
    if (vao_) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    if (vbo_) {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    if (shader_program_) {
        glDeleteProgram(shader_program_);
        shader_program_ = 0;
    }
}
