#include "cuda_renderer.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Vector math utilities for float3 and float2
__device__ inline float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ inline float3 operator*(float s, float3 a) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ inline float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ inline float3 operator/(float3 a, float s) { return make_float3(a.x / s, a.y / s, a.z / s); }
__device__ inline float3 operator/(float3 a, float3 b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
__device__ inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float length(float3 a) { return sqrtf(dot(a, a)); }
__device__ inline float length(float2 a) { return sqrtf(a.x * a.x + a.y * a.y); }
__device__ inline float3 normalize(float3 a) { float l = length(a); return l > 1e-6f ? a / l : make_float3(0,0,0); }
__device__ inline float3 abs3(float3 a) { return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z)); }
__device__ inline float3 max3(float3 a, float3 b) { return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
__device__ inline float clampf(float x, float a, float b) { return fmaxf(a, fminf(b, x)); }
__device__ inline float3 clamp3(float3 v, float a, float b) {
    return make_float3(clampf(v.x, a, b), clampf(v.y, a, b), clampf(v.z, a, b));
}

// -------------------------------------------------------------
// SDF Primitives
// -------------------------------------------------------------
__device__ inline float sdSphere(float3 p, float r) {
    return length(p) - r;
}

__device__ inline float sdBox(float3 p, float3 b) {
    float3 q = abs3(p) - b;
    return length(max3(q, make_float3(0,0,0))) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f);
}

__device__ inline float sdTorus(float3 p, float r_major, float r_minor) {
    float q_x = length(make_float2(p.x, p.z)) - r_major;
    return sqrtf(q_x * q_x + p.y * p.y) - r_minor;
}

__device__ inline float sdCylinder(float3 p, float r, float h) {
    float d_xz = length(make_float2(p.x, p.z)) - r;
    float d_y = fabsf(p.y) - h;
    return fminf(fmaxf(d_xz, d_y), 0.0f) + length(max3(make_float3(d_xz, d_y, 0.0f), make_float3(0,0,0)));
}

// -------------------------------------------------------------
// CSG Operations
// -------------------------------------------------------------
__device__ inline float opUnion(float d1, float d2) { return fminf(d1, d2); }
__device__ inline float opSubtract(float d1, float d2) { return fmaxf(-d1, d2); }
__device__ inline float opIntersect(float d1, float d2) { return fmaxf(d1, d2); }

__device__ inline float opSmoothUnion(float d1, float d2, float k) {
    float h = clampf(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
    return (d2 * (1.0f - h) + d1 * h) - k * h * (1.0f - h);
}

__device__ inline float opSmoothSubtract(float d1, float d2, float k) {
    float h = clampf(0.5f - 0.5f * (d2 + d1) / k, 0.0f, 1.0f);
    return (d2 * (1.0f - h) - d1 * h) + k * h * (1.0f - h);
}

// -------------------------------------------------------------
// Scene Evaluation
// -------------------------------------------------------------
__device__ float mapScene(float3 p, float time, int scene_id) {
    float floor_d = p.y + 1.25f;

    if (scene_id == 1) {
        // Scene 1: Smooth morphing CSG
        float3 p_obj = p;
        p_obj.y -= 0.15f * sinf(time * 2.0f);

        float sphere = sdSphere(p_obj, 1.0f);
        float box = sdBox(p_obj, make_float3(0.75f, 0.75f, 0.75f));
        float torus = sdTorus(p_obj, 1.35f, 0.18f + 0.04f * sinf(time * 3.0f));

        float blend_k = 0.35f + 0.2f * sinf(time * 1.5f);
        float obj = opSmoothUnion(sphere, box, blend_k);
        obj = opSmoothUnion(obj, torus, 0.25f);

        return opUnion(obj, floor_d);
    } else if (scene_id == 2) {
        // Scene 2: Mechanical CSG
        float3 p_obj = p;
        p_obj.y -= 0.1f;

        float box = sdBox(p_obj, make_float3(0.85f, 0.85f, 0.85f));
        float sphere = sdSphere(p_obj, 1.15f);
        float base = opIntersect(box, sphere);

        float cyl_y = sdCylinder(p_obj, 0.42f, 2.0f);
        float cyl_x = sdCylinder(make_float3(p_obj.y, p_obj.x, p_obj.z), 0.42f, 2.0f);
        float cyl_z = sdCylinder(make_float3(p_obj.x, p_obj.z, p_obj.y), 0.42f, 2.0f);

        float obj = opSubtract(cyl_y, base);
        obj = opSubtract(cyl_x, obj);
        obj = opSubtract(cyl_z, obj);

        return opUnion(obj, floor_d);
    } else {
        // Scene 3: Mathematical Lattice / Gyroid in a sphere
        float3 p_obj = p;
        p_obj.y -= 0.1f;

        float bound = sdSphere(p_obj, 1.45f);

        float scale = 3.2f;
        float3 sp = p_obj * scale;
        float gyroid = fabsf(sinf(sp.x) * cosf(sp.y) + sinf(sp.y) * cosf(sp.z) + sinf(sp.z) * cosf(sp.x)) / scale - 0.06f;

        float obj = opIntersect(gyroid, bound);
        return opUnion(obj, floor_d);
    }
}

// -------------------------------------------------------------
// Normals, Shadows, and Ambient Occlusion
// -------------------------------------------------------------
__device__ float3 calcNormal(float3 p, float time, int scene_id) {
    const float eps = 0.0005f;
    float3 n = make_float3(
        mapScene(p + make_float3(eps, 0.0f, 0.0f), time, scene_id) - mapScene(p - make_float3(eps, 0.0f, 0.0f), time, scene_id),
        mapScene(p + make_float3(0.0f, eps, 0.0f), time, scene_id) - mapScene(p - make_float3(0.0f, eps, 0.0f), time, scene_id),
        mapScene(p + make_float3(0.0f, 0.0f, eps), time, scene_id) - mapScene(p - make_float3(0.0f, 0.0f, eps), time, scene_id)
    );
    return normalize(n);
}

__device__ float calcSoftShadow(float3 ro, float3 rd, float mint, float maxt, float k, float time, int scene_id) {
    float res = 1.0f;
    float t = mint;
    for (int i = 0; i < 32 && t < maxt; ++i) {
        float h = mapScene(ro + rd * t, time, scene_id);
        if (h < 0.001f) return 0.0f;
        res = fminf(res, k * h / t);
        t += clampf(h, 0.02f, 0.25f);
    }
    return clampf(res, 0.0f, 1.0f);
}

__device__ float calcAO(float3 p, float3 n, float time, int scene_id) {
    float occ = 0.0f;
    float sca = 1.0f;
    for (int i = 0; i < 5; ++i) {
        float h = 0.01f + 0.12f * (float)i / 4.0f;
        float d = mapScene(p + n * h, time, scene_id);
        occ += (h - d) * sca;
        sca *= 0.95f;
    }
    return clampf(1.0f - 3.0f * occ, 0.0f, 1.0f);
}

// -------------------------------------------------------------
// Main Raymarch Kernel
// -------------------------------------------------------------
__global__ void renderKernel(
    uchar4* d_output,
    int width,
    int height,
    CameraUniforms cam,
    float time,
    int scene_id
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Camera ray generation
    float u_coord = (2.0f * (x + 0.5f) / width - 1.0f) * cam.fov_scale * cam.aspect;
    float v_coord = (1.0f - 2.0f * (y + 0.5f) / height) * cam.fov_scale;

    float3 ro = make_float3(cam.ro_x, cam.ro_y, cam.ro_z);
    float3 cam_u = make_float3(cam.u_x, cam.u_y, cam.u_z);
    float3 cam_v = make_float3(cam.v_x, cam.v_y, cam.v_z);
    float3 cam_w = make_float3(cam.w_x, cam.w_y, cam.w_z);

    float3 rd = normalize(cam_u * u_coord + cam_v * v_coord + cam_w);

    // Dynamic background sky gradient
    float3 col = make_float3(0.06f, 0.08f, 0.12f) + make_float3(0.10f, 0.14f, 0.22f) * (0.5f * rd.y + 0.5f);

    float t = 0.0f;
    const float t_max = 40.0f;
    bool hit = false;

    // Raymarching loop
    for (int i = 0; i < 128; ++i) {
        float3 pos = ro + rd * t;
        float d = mapScene(pos, time, scene_id);
        if (d < 0.001f) {
            hit = true;
            break;
        }
        t += d;
        if (t > t_max) break;
    }

    if (hit) {
        float3 p = ro + rd * t;
        float3 n = calcNormal(p, time, scene_id);

        float3 mat_col;
        if (p.y < -1.23f) {
            // Anti-aliased procedural checkerboard floor
            float check = fmodf(floorf(p.x * 1.5f) + floorf(p.z * 1.5f), 2.0f);
            mat_col = (check > 0.5f || check < -0.5f) ? make_float3(0.18f, 0.20f, 0.24f) : make_float3(0.32f, 0.35f, 0.42f);
        } else {
            if (scene_id == 1) {
                mat_col = make_float3(0.95f, 0.48f, 0.18f); // Warm copper
            } else if (scene_id == 2) {
                mat_col = make_float3(0.18f, 0.72f, 0.95f); // Tech cyan
            } else {
                mat_col = make_float3(0.92f, 0.25f, 0.75f); // Neon magenta
            }
        }

        // Key light + Fill light
        float3 light_dir = normalize(make_float3(0.7f, 1.2f, 0.6f));
        float diff = fmaxf(dot(n, light_dir), 0.0f);
        float shadow = calcSoftShadow(p + n * 0.006f, light_dir, 0.02f, 15.0f, 16.0f, time, scene_id);
        float ao = calcAO(p, n, time, scene_id);

        // Blinn-phong specular
        float3 half_vec = normalize(light_dir - rd);
        float spec = powf(fmaxf(dot(n, half_vec), 0.0f), 32.0f);

        float3 ambient = make_float3(0.07f, 0.09f, 0.14f) * ao;
        float3 diffuse = mat_col * diff * shadow;
        float3 specular = make_float3(1.0f, 1.0f, 1.0f) * spec * shadow * 0.5f;

        col = ambient + diffuse + specular;

        // Atmospheric depth fog
        float fog = 1.0f - expf(-t * 0.035f);
        float3 fog_col = make_float3(0.06f, 0.08f, 0.12f);
        col = col * (1.0f - fog) + fog_col * fog;
    }

    // Tone mapping and gamma correction
    col = col / (col + make_float3(1.0f, 1.0f, 1.0f));
    col = make_float3(powf(col.x, 1.0f / 2.2f), powf(col.y, 1.0f / 2.2f), powf(col.z, 1.0f / 2.2f));
    col = clamp3(col, 0.0f, 1.0f);

    d_output[y * width + x] = make_uchar4(
        (unsigned char)(col.x * 255.0f),
        (unsigned char)(col.y * 255.0f),
        (unsigned char)(col.z * 255.0f),
        255
    );
}

void launch_cuda_render(
    uchar4* d_output,
    int width,
    int height,
    const CameraUniforms& cam,
    float time,
    int scene_id
) {
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    renderKernel<<<blocks, threads>>>(d_output, width, height, cam, time, scene_id);
    cudaDeviceSynchronize();
}
