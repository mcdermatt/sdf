#pragma once
// Plain structs shared between C++ host code and the CUDA kernel.

struct CameraUniforms {
    float ro_x, ro_y, ro_z;   // eye position
    float u_x, u_y, u_z;      // camera right
    float v_x, v_y, v_z;      // camera up
    float w_x, w_y, w_z;      // camera forward
    float fov_scale;          // tan(fov_y / 2)
    float aspect;
};

struct FrameParams {
    CameraUniforms cam;
    float time;                         // seconds, drives cloud wind + glow pulse
    float sun_x, sun_y, sun_z;          // normalized direction toward the sun

    float gun_x, gun_y, gun_z;          // gun anchor in camera space (incl. bob/recoil)
    float recoil;                       // 0..1, decays after each shot
    float beam;                         // 0..1 beam intensity
    float beam_x, beam_y, beam_z;       // beam end point (world)
    float blast_radius;
};
