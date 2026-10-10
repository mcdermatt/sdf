#include "player.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

Player::Player()
    : pos_(0.0f, 5.0f, 0.0f), vel_(0.0f),
      yaw_(glm::radians(-45.0f)), pitch_(glm::radians(-5.0f)) {}

void Player::init(float start_x, float start_z, const WorldManager& world) {
    pos_.x = start_x;
    pos_.z = start_z;
    pos_.y = world.getGroundHeight(start_x, start_z) + eye_height_;
    vel_ = glm::vec3(0.0f);
    yaw_ = 0.0f; // Directly facing cabin (-Z direction)
    pitch_ = glm::radians(-3.0f);
    rotate(0.0f, 0.0f);
}

void Player::rotate(float delta_yaw, float delta_pitch) {
    yaw_ += delta_yaw;
    // Un-inverted mouse look with safe pitch clamp [-88, 88 degrees]
    pitch_ = std::clamp(pitch_ + delta_pitch, glm::radians(-88.0f), glm::radians(88.0f));

    // Calculate forward, right, up vectors (yaw=0 points North towards -Z)
    forward_.x = std::cos(pitch_) * std::sin(yaw_);
    forward_.y = std::sin(pitch_);
    forward_.z = std::cos(pitch_) * -std::cos(yaw_);
    forward_ = glm::normalize(forward_);

    glm::vec3 world_up(0.0f, 1.0f, 0.0f);
    right_ = glm::normalize(glm::cross(forward_, world_up));
    up_ = glm::normalize(glm::cross(right_, forward_));
}

void Player::setMovementInput(float forward, float strafe, bool sprint, bool jump) {
    input_forward_ = forward;
    input_strafe_ = strafe;
    input_sprint_ = sprint;
    input_jump_ = jump;
}

void Player::update(float dt, const WorldManager& world) {
    dt = std::clamp(dt, 0.001f, 0.05f);

    // 1. Horizontal movement vector along ground plane
    glm::vec3 flat_forward = glm::normalize(glm::vec3(forward_.x, 0.0f, forward_.z));
    glm::vec3 flat_right = glm::normalize(glm::vec3(right_.x, 0.0f, right_.z));

    glm::vec3 wish_dir = flat_forward * input_forward_ + flat_right * input_strafe_;
    float move_len = glm::length(wish_dir);
    if (move_len > 1e-4f) {
        wish_dir /= move_len;
    }

    float speed = input_sprint_ ? 7.5f : 4.0f; // Sprint vs Walk speed (m/s)
    glm::vec3 target_horiz_vel = wish_dir * (speed * std::min(move_len, 1.0f));

    // Smooth acceleration / friction
    float accel = on_ground_ ? 18.0f : 4.0f;
    vel_.x = glm::mix(vel_.x, target_horiz_vel.x, std::min(1.0f, dt * accel));
    vel_.z = glm::mix(vel_.z, target_horiz_vel.z, std::min(1.0f, dt * accel));

    // 2. Vertical movement & Jump
    if (on_ground_ && input_jump_) {
        vel_.y = 7.0f; // Jump impulse
        on_ground_ = false;
    }

    // Apply gravity
    vel_.y -= 22.0f * dt;

    // Apply velocity
    pos_ += vel_ * dt;

    // 3. Ground Collision & Terrain Surface Walking
    float ground_y = world.getGroundHeight(pos_.x, pos_.z) + eye_height_;
    if (pos_.y <= ground_y) {
        pos_.y = ground_y;
        vel_.y = 0.0f;
        on_ground_ = true;
    } else {
        on_ground_ = false;
    }

    // 4. Head and Gun Bobbing
    float horiz_speed = std::sqrt(vel_.x * vel_.x + vel_.z * vel_.z);
    if (on_ground_ && horiz_speed > 0.3f) {
        float bob_freq = input_sprint_ ? 13.0f : 9.0f;
        bob_phase_ += dt * bob_freq;
        current_bob_x_ = std::sin(bob_phase_ * 0.5f) * 0.015f;
        current_bob_y_ = std::abs(std::sin(bob_phase_)) * 0.018f;
    } else {
        current_bob_x_ = glm::mix(current_bob_x_, 0.0f, dt * 8.0f);
        current_bob_y_ = glm::mix(current_bob_y_, 0.0f, dt * 8.0f);
    }

    // 5. Gun Recoil and Beam decay
    gun_recoil_ = std::max(0.0f, gun_recoil_ - dt * 6.5f);
    gun_beam_intensity_ = std::max(0.0f, gun_beam_intensity_ - dt * 9.0f);
}

void Player::fireRayGun(WorldManager& world) {
    gun_recoil_ = 1.0f;
    gun_beam_intensity_ = 1.0f;

    float3 ro = make_float3(pos_.x, pos_.y, pos_.z);
    float3 rd = make_float3(forward_.x, forward_.y, forward_.z);

    float3 hit_pos;
    if (world.raycast(ro, rd, 120.0f, hit_pos)) {
        beam_target_ = glm::vec3(hit_pos.x, hit_pos.y, hit_pos.z);
        // Carve out a crater in the SDF (1.35m blast radius)
        world.addCarve(hit_pos, 1.35f);
    } else {
        beam_target_ = pos_ + forward_ * 90.0f;
    }
}

FrameParams Player::getFrameParams(int width, int height, float time) const {
    FrameParams params;

    // Camera uniforms
    params.cam.ro_x = pos_.x;
    params.cam.ro_y = pos_.y;
    params.cam.ro_z = pos_.z;

    params.cam.u_x = right_.x;
    params.cam.u_y = right_.y;
    params.cam.u_z = right_.z;

    params.cam.v_x = up_.x;
    params.cam.v_y = up_.y;
    params.cam.v_z = up_.z;

    params.cam.w_x = forward_.x;
    params.cam.w_y = forward_.y;
    params.cam.w_z = forward_.z;

    params.cam.fov_scale = std::tan(fov_radians_ * 0.5f);
    params.cam.aspect = static_cast<float>(width) / std::max(static_cast<float>(height), 1.0f);

    params.time = time;

    // Sunset sun direction: low in the western sky (~14 degrees elevation)
    params.sun_x = -0.72f;
    params.sun_y =  0.24f;
    params.sun_z = -0.65f;

    // Gun view-space anchor on right side of screen with walk bobbing
    params.gun_x = 0.26f + current_bob_x_;
    params.gun_y = -0.19f - current_bob_y_;
    params.gun_z = 0.44f;

    params.recoil = gun_recoil_;
    params.beam = gun_beam_intensity_;
    params.beam_x = beam_target_.x;
    params.beam_y = beam_target_.y;
    params.beam_z = beam_target_.z;
    params.blast_radius = 1.35f;

    return params;
}
