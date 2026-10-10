#pragma once
#include <glm/glm.hpp>
#include "frame_params.h"
#include "world_manager.h"

class Player {
public:
    Player();

    void init(float start_x, float start_z, const WorldManager& world);
    void update(float dt, const WorldManager& world);

    // Mouse look
    void rotate(float delta_yaw, float delta_pitch);

    // Movement input
    void setMovementInput(float forward, float strafe, bool sprint, bool jump);

    // Action input
    void fireRayGun(WorldManager& world);

    // Build frame parameters for the CUDA renderer
    FrameParams getFrameParams(int width, int height, float time) const;

    glm::vec3 getPosition() const { return pos_; }
    glm::vec3 getForward() const { return forward_; }
    bool isGrounded() const { return on_ground_; }

private:
    glm::vec3 pos_;
    glm::vec3 vel_;
    float yaw_;    // Radians (horizontal)
    float pitch_;  // Radians (vertical)

    glm::vec3 forward_;
    glm::vec3 right_;
    glm::vec3 up_;

    // Movement state
    float input_forward_ = 0.0f;
    float input_strafe_ = 0.0f;
    bool input_sprint_ = false;
    bool input_jump_ = false;
    bool on_ground_ = false;

    // Head / gun bobbing
    float bob_phase_ = 0.0f;
    float current_bob_x_ = 0.0f;
    float current_bob_y_ = 0.0f;

    // Ray gun state
    float gun_recoil_ = 0.0f;
    float gun_beam_intensity_ = 0.0f;
    glm::vec3 beam_target_ = glm::vec3(0.0f);

    float fov_radians_ = glm::radians(65.0f);
    const float eye_height_ = 1.70f;
};
