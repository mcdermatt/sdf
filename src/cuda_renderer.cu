#include "cuda_renderer.cuh"
#include "sdf_common.cuh"
#include "frame_params.h"
#include <device_launch_parameters.h>

// =============================================================
// Ray Gun in Camera View Space
// =============================================================
HD float sdGunBody(float3 p, float recoil, int* is_glow) {
    // Gun model centered around local origin
    // Apply recoil: pushes gun backwards along -Z and tilts muzzle slightly up
    float3 q = p;
    q.z += recoil * 0.06f;
    q.y -= recoil * 0.02f;

    // Receiver / main body
    float body = sdRoundBox(q - make_float3(0.0f, 0.0f, 0.0f), make_float3(0.04f, 0.055f, 0.15f), 0.015f);

    // Barrel
    float barrel = sdCylZ(q - make_float3(0.0f, 0.035f, 0.16f), 0.028f, 0.14f);

    // Upper heatsink / top rail
    float rail = sdBox(q - make_float3(0.0f, 0.075f, 0.02f), make_float3(0.025f, 0.012f, 0.13f));

    // Angled Grip
    float3 q_grip = q - make_float3(0.0f, -0.09f, -0.08f);
    // Slight tilt: rotate in YZ plane
    float cos_a = 0.955f, sin_a = 0.296f; // ~17 degrees
    float3 q_rot = make_float3(q_grip.x, q_grip.y * cos_a + q_grip.z * sin_a, -q_grip.y * sin_a + q_grip.z * cos_a);
    float grip = sdRoundBox(q_rot, make_float3(0.028f, 0.075f, 0.038f), 0.01f);

    // Glowing energy chamber ring & muzzle emitter
    float energy_ring = sdTorusZ(q - make_float3(0.0f, 0.035f, 0.08f), 0.038f, 0.012f);
    float emitter = sdCylZ(q - make_float3(0.0f, 0.035f, 0.29f), 0.020f, 0.015f);
    float glow_parts = fminf(energy_ring, emitter);

    float solid = fminf(fminf(body, barrel), fminf(rail, grip));

    if (glow_parts < solid) {
        if (is_glow) *is_glow = 1;
        return glow_parts;
    } else {
        if (is_glow) *is_glow = 0;
        return solid;
    }
}

// =============================================================
// Sunset Sky & Volumetric Clouds
// =============================================================
HD float3 evalSkyColor(float3 rd, float3 sun_dir, float time) {
    float sun_dot = clampf(dot(rd, sun_dir), 0.0f, 1.0f);

    // Horizon to zenith gradient
    float elevation = clampf(rd.y, 0.0f, 1.0f);
    float3 zenith = make_float3(0.12f, 0.25f, 0.52f);       // Deep evening blue
    float3 horizon = make_float3(0.96f, 0.54f, 0.22f);      // Golden orange sunset
    float3 sunset_glow = make_float3(1.0f, 0.72f, 0.35f);   // Bright amber halo around sun

    float3 sky = mix3(horizon, zenith, powf(elevation, 0.45f));
    sky += sunset_glow * powf(sun_dot, 5.0f) * 0.65f;

    // Sun disk
    if (sun_dot > 0.9985f) {
        sky += make_float3(3.5f, 2.8f, 1.8f);
    } else {
        sky += make_float3(1.2f, 0.8f, 0.4f) * powf(sun_dot, 120.0f) * 1.5f;
    }

    return sky;
}

HD float sampleCloudDensity(float3 p, float time) {
    float wind = time * 0.6f;
    float3 cp = p * 0.015f + make_float3(wind * 0.08f, 0.0f, wind * 0.04f);

    float n = perlin2(cp.x * 2.5f, cp.z * 2.5f) * 0.55f;
    n += perlin2(cp.x * 5.0f + 13.1f, cp.z * 5.0f - 7.3f) * 0.30f;
    n += perlin2(cp.x * 10.0f - 5.2f, cp.z * 10.0f + 19.4f) * 0.15f;

    // Height fade: clouds between altitude 60m and 110m
    float h_fade = smoothstepf(60.0f, 75.0f, p.y) * smoothstepf(115.0f, 95.0f, p.y);
    float density = clampf((n - 0.32f) * 2.2f, 0.0f, 1.0f) * h_fade;
    return density;
}

HD float3 renderVolumetricClouds(float3 ro, float3 rd, float3 sun_dir, float3 bg_color, float time, float max_dist) {
    if (rd.y < 0.015f) return bg_color;

    float t_bottom = (60.0f - ro.y) / rd.y;
    float t_top    = (115.0f - ro.y) / rd.y;

    if (t_top < 0.0f) return bg_color;
    float t_start = fmaxf(t_bottom, 0.0f);
    float t_end   = fminf(t_top, max_dist);
    if (t_start >= t_end) return bg_color;

    // Raymarch cloud layer in 16 steps
    const int NUM_STEPS = 16;
    float step_len = (t_end - t_start) / (float)NUM_STEPS;
    float t = t_start + step_len * 0.5f;

    float transmittance = 1.0f;
    float3 cloud_color = make_float3(0.0f, 0.0f, 0.0f);
    float sun_dot = clampf(dot(rd, sun_dir), 0.0f, 1.0f);

    // Mie forward scattering peak (silver lining)
    float mie_phase = 0.45f + 0.55f * powf(sun_dot, 6.0f);

    float3 sun_light = make_float3(1.15f, 0.70f, 0.35f);     // Sunset sun color
    float3 ambient_sky = make_float3(0.20f, 0.30f, 0.50f);   // Blue ambient sky fill

    for (int i = 0; i < NUM_STEPS; ++i) {
        float3 pos = ro + rd * t;
        float d = sampleCloudDensity(pos, time);

        if (d > 0.001f) {
            // Light sample toward sun (self-shadowing)
            float3 sun_pos = pos + sun_dir * 12.0f;
            float sun_sample = sampleCloudDensity(sun_pos, time);
            float shadow = expf(-sun_sample * 2.4f);

            // Cloud lighting: sunset direct light + ambient fill
            float3 step_light = sun_light * (shadow * mie_phase * 1.8f) + ambient_sky * 0.45f;

            // Optical thickness
            float opt_thick = d * step_len * 0.12f;
            float step_trans = expf(-opt_thick);

            cloud_color += (step_light * (1.0f - step_trans)) * transmittance;
            transmittance *= step_trans;

            if (transmittance < 0.02f) break;
        }

        t += step_len;
    }

    return bg_color * transmittance + cloud_color;
}

// =============================================================
// Surface Normals, Shadows & Ambient Occlusion
// =============================================================
HD float3 calcNormal(float3 p, const WorldView& w) {
    const float eps = 0.003f;
    float3 n = make_float3(
        mapWorld(p + make_float3(eps, 0.0f, 0.0f), w) - mapWorld(p - make_float3(eps, 0.0f, 0.0f), w),
        mapWorld(p + make_float3(0.0f, eps, 0.0f), w) - mapWorld(p - make_float3(0.0f, eps, 0.0f), w),
        mapWorld(p + make_float3(0.0f, 0.0f, eps), w) - mapWorld(p - make_float3(0.0f, 0.0f, eps), w)
    );
    return normalize(n);
}

HD float calcSoftShadow(float3 ro, float3 rd, float mint, float maxt, float k, const WorldView& w) {
    float res = 1.0f;
    float t = mint;
    for (int i = 0; i < 28 && t < maxt; ++i) {
        float h = mapWorld(ro + rd * t, w);
        if (h < 0.002f) return 0.0f;
        res = fminf(res, k * h / t);
        t += clampf(h, 0.04f, 0.40f);
    }
    return clampf(res, 0.0f, 1.0f);
}

HD float calcAO(float3 p, float3 n, const WorldView& w) {
    float occ = 0.0f;
    float sca = 1.0f;
    for (int i = 0; i < 5; ++i) {
        float h = 0.03f + 0.25f * (float)i / 4.0f;
        float d = mapWorld(p + n * h, w);
        occ += (h - d) * sca;
        sca *= 0.85f;
    }
    return clampf(1.0f - 2.8f * occ, 0.0f, 1.0f);
}

// =============================================================
// Material Shading
// =============================================================
HD float3 getMaterialAlbedo(int mat_id, float3 p, const WorldView& w) {
    // If it's terrain, resolve into grass / dirt / rock based on depth and slope
    if (mat_id == MAT_TERRAIN) {
        mat_id = terrainLayer(p, w);
    }

    switch (mat_id) {
        case MAT_GRASS: {
            float n = perlin2(p.x * 2.0f, p.z * 2.0f);
            return mix3(make_float3(0.18f, 0.42f, 0.10f), make_float3(0.28f, 0.52f, 0.14f), 0.5f + 0.5f * n);
        }
        case MAT_DIRT: {
            float n = perlin2(p.x * 3.5f, p.z * 3.5f);
            return mix3(make_float3(0.32f, 0.20f, 0.12f), make_float3(0.42f, 0.28f, 0.16f), 0.5f + 0.5f * n);
        }
        case MAT_ROCK: {
            float n = perlin2(p.x * 1.5f + p.y * 2.0f, p.z * 1.5f);
            return mix3(make_float3(0.36f, 0.35f, 0.34f), make_float3(0.48f, 0.46f, 0.44f), 0.5f + 0.5f * n);
        }
        case MAT_BARK:
            return make_float3(0.24f, 0.14f, 0.08f);
        case MAT_LEAVES: {
            float n = perlin2(p.x * 4.0f, p.z * 4.0f);
            return mix3(make_float3(0.10f, 0.26f, 0.11f), make_float3(0.16f, 0.36f, 0.16f), 0.5f + 0.5f * n);
        }
        case MAT_LOG: {
            float ring = sinf(p.y * 22.0f) * 0.05f;
            return make_float3(0.52f + ring, 0.32f + ring, 0.17f + ring);
        }
        case MAT_ROOF:
            return make_float3(0.34f, 0.22f, 0.15f);
        case MAT_PLANK:
            return make_float3(0.58f, 0.40f, 0.24f);
        case MAT_STONE:
            return make_float3(0.44f, 0.42f, 0.40f);
        default:
            return make_float3(0.5f, 0.5f, 0.5f);
    }
}

// Distance from point to line segment
HD float distToSegment(float3 p, float3 a, float3 b) {
    float3 pa = p - a, ba = b - a;
    float h = clampf(dot(pa, ba) / dot(ba, ba), 0.0f, 1.0f);
    return length(pa - ba * h);
}

// =============================================================
// Primary CUDA Raymarching Kernel
// =============================================================
__global__ void renderKernel(
    uchar4* d_output,
    int width,
    int height,
    FrameParams params,
    WorldView world
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Camera view ray
    float u_coord = (2.0f * (x + 0.5f) / (float)width - 1.0f) * params.cam.fov_scale * params.cam.aspect;
    float v_coord = (1.0f - 2.0f * (y + 0.5f) / (float)height) * params.cam.fov_scale;

    float3 ro = make_float3(params.cam.ro_x, params.cam.ro_y, params.cam.ro_z);
    float3 cam_u = make_float3(params.cam.u_x, params.cam.u_y, params.cam.u_z);
    float3 cam_v = make_float3(params.cam.v_x, params.cam.v_y, params.cam.v_z);
    float3 cam_w = make_float3(params.cam.w_x, params.cam.w_y, params.cam.w_z);

    float3 rd = normalize(cam_u * u_coord + cam_v * v_coord + cam_w);
    float3 sun_dir = normalize(make_float3(params.sun_x, params.sun_y, params.sun_z));

    // ---------------------------------------------------------
    // 1. Raymarch Gun in Camera Space
    // ---------------------------------------------------------
    // Ray in camera local coordinate system: origin (0,0,0), direction:
    float3 rd_cam = normalize(make_float3(u_coord, v_coord, 1.0f));
    float3 gun_anchor = make_float3(params.gun_x, params.gun_y, params.gun_z);

    bool gun_hit = false;
    float t_gun = 0.10f;
    int gun_is_glow = 0;

    for (int i = 0; i < 48; ++i) {
        float3 p_local = rd_cam * t_gun - gun_anchor;
        int is_glow = 0;
        float d = sdGunBody(p_local, params.recoil, &is_glow);
        if (d < 0.001f) {
            gun_hit = true;
            gun_is_glow = is_glow;
            break;
        }
        t_gun += d;
        if (t_gun > 1.2f) break;
    }

    // ---------------------------------------------------------
    // 2. Raymarch World SDF
    // ---------------------------------------------------------
    float t_world = 0.2f;
    const float t_world_max = 240.0f;
    bool world_hit = false;
    int hit_mat = MAT_TERRAIN;

    for (int i = 0; i < 160; ++i) {
        float3 pos = ro + rd * t_world;
        int mat_candidate = MAT_TERRAIN;
        float d = mapWorld(pos, world, &mat_candidate);

        if (d < 0.002f * (1.0f + t_world * 0.04f)) {
            world_hit = true;
            hit_mat = mat_candidate;
            break;
        }
        t_world += d;
        if (t_world > t_world_max) break;
    }

    // ---------------------------------------------------------
    // 3. Shading Composition
    // ---------------------------------------------------------
    float3 final_col = make_float3(0.0f, 0.0f, 0.0f);

    if (gun_hit) {
        // Gun Surface Shading
        float3 p_local = rd_cam * t_gun - gun_anchor;
        const float eps = 0.001f;
        float3 gun_n = normalize(make_float3(
            sdGunBody(p_local + make_float3(eps, 0.0f, 0.0f), params.recoil, nullptr) - sdGunBody(p_local - make_float3(eps, 0.0f, 0.0f), params.recoil, nullptr),
            sdGunBody(p_local + make_float3(0.0f, eps, 0.0f), params.recoil, nullptr) - sdGunBody(p_local - make_float3(0.0f, eps, 0.0f), params.recoil, nullptr),
            sdGunBody(p_local + make_float3(0.0f, 0.0f, eps), params.recoil, nullptr) - sdGunBody(p_local - make_float3(0.0f, 0.0f, eps), params.recoil, nullptr)
        ));

        // Transform gun normal to world space
        float3 world_n = cam_u * gun_n.x + cam_v * gun_n.y + cam_w * gun_n.z;

        if (gun_is_glow) {
            // Glowing neon cyan / plasma energy core with pulse
            float pulse = 0.8f + 0.2f * sinf(params.time * 8.0f);
            final_col = make_float3(0.15f, 0.85f, 1.0f) * (2.2f * pulse + params.recoil * 3.0f);
        } else {
            // Gun chassis: dark brushed titanium alloy
            float3 albedo = make_float3(0.18f, 0.20f, 0.22f);
            float diff = fmaxf(dot(world_n, sun_dir), 0.0f);
            float3 half_v = normalize(sun_dir - rd);
            float spec = powf(fmaxf(dot(world_n, half_v), 0.0f), 32.0f);

            float3 sun_light = make_float3(1.2f, 0.8f, 0.45f);
            float3 amb = make_float3(0.15f, 0.18f, 0.24f);

            final_col = albedo * (amb + sun_light * diff) + make_float3(1.0f, 0.9f, 0.8f) * spec * 0.7f;
        }
    } else if (world_hit) {
        // World Surface Shading
        float3 p = ro + rd * t_world;
        float3 n = calcNormal(p, world);
        float3 albedo = getMaterialAlbedo(hit_mat, p, world);

        // Lighting
        float diff = fmaxf(dot(n, sun_dir), 0.0f);
        float shadow = calcSoftShadow(p + n * 0.02f, sun_dir, 0.05f, 35.0f, 12.0f, world);
        float ao = calcAO(p, n, world);

        // Warm sunset direct sunlight + blue sky ambient
        float3 sun_color = make_float3(1.25f, 0.75f, 0.38f);
        float3 sky_ambient = make_float3(0.14f, 0.20f, 0.32f);

        // Blinn-phong specular
        float3 half_v = normalize(sun_dir - rd);
        float spec = powf(fmaxf(dot(n, half_v), 0.0f), 24.0f) * 0.35f;

        // Hemispherical ground bounce light
        float ground_bounce = clampf(-n.y * 0.5f + 0.5f, 0.0f, 1.0f);
        float3 bounce_col = make_float3(0.18f, 0.12f, 0.08f) * ground_bounce;

        float3 lit = albedo * (sky_ambient * ao + sun_color * (diff * shadow) + bounce_col) + sun_color * (spec * shadow);

        // Atmospheric distance fog (fading smoothly into sunset sky)
        float fog = 1.0f - expf(-t_world * 0.007f);
        float3 fog_col = evalSkyColor(rd, sun_dir, params.time);
        final_col = mix3(lit, fog_col, fog);
    } else {
        // Skybox + Clouds
        float3 sky = evalSkyColor(rd, sun_dir, params.time);
        final_col = renderVolumetricClouds(ro, rd, sun_dir, sky, params.time, t_world_max);
    }

    // ---------------------------------------------------------
    // 4. Ray Gun Beam Laser Glow Effect
    // ---------------------------------------------------------
    if (params.beam > 0.01f) {
        // Gun muzzle position in world space
        float3 muzzle_cam = gun_anchor + make_float3(0.0f, 0.035f, 0.32f);
        float3 muzzle_world = ro + cam_u * muzzle_cam.x + cam_v * muzzle_cam.y + cam_w * muzzle_cam.z;
        float3 target_world = make_float3(params.beam_x, params.beam_y, params.beam_z);

        // Closest approach of pixel ray to laser line segment
        float dist_to_beam = distToSegment(ro + rd * fminf(t_world, 8.0f), muzzle_world, target_world);
        float beam_glow = params.beam / (dist_to_beam * 22.0f + 0.08f);
        beam_glow = clampf(beam_glow, 0.0f, 3.0f);

        float3 beam_color = make_float3(0.2f, 0.85f, 1.0f) * beam_glow;
        final_col += beam_color;
    }

    // ---------------------------------------------------------
    // 5. Tonemapping and Gamma
    // ---------------------------------------------------------
    // ACES / Reinhard tonemap
    final_col = final_col / (final_col + make_float3(1.0f, 1.0f, 1.0f));
    final_col = make_float3(powf(final_col.x, 1.0f / 2.2f), powf(final_col.y, 1.0f / 2.2f), powf(final_col.z, 1.0f / 2.2f));
    final_col = clamp3(final_col, 0.0f, 1.0f);

    d_output[y * width + x] = make_uchar4(
        (unsigned char)(final_col.x * 255.0f),
        (unsigned char)(final_col.y * 255.0f),
        (unsigned char)(final_col.z * 255.0f),
        255
    );
}

void launch_cuda_render(
    uchar4* d_output,
    int width,
    int height,
    const FrameParams& params,
    const WorldView& world
) {
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    renderKernel<<<blocks, threads>>>(d_output, width, height, params, world);
    cudaDeviceSynchronize();
}
