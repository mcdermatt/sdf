#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

OrbitCamera::OrbitCamera(float radius, float theta, float phi)
    : default_radius_(radius), default_theta_(theta), default_phi_(phi),
      radius_(radius), theta_(theta), phi_(phi),
      target_(0.0f, 0.0f, 0.0f), fov_radians_(glm::radians(45.0f)) {}

void OrbitCamera::rotate(float delta_theta, float delta_phi) {
    theta_ += delta_theta;
    phi_ = std::clamp(phi_ + delta_phi, -1.50f, 1.50f);
}

void OrbitCamera::pan(float delta_x, float delta_y) {
    glm::vec3 eye = getEye();
    glm::vec3 w = glm::normalize(target_ - eye);
    glm::vec3 u = glm::normalize(glm::cross(w, glm::vec3(0.0f, 1.0f, 0.0f)));
    glm::vec3 v = glm::cross(u, w);

    float scale = radius_ * 0.002f;
    target_ += (-u * delta_x + v * delta_y) * scale;
}

void OrbitCamera::zoom(float factor) {
    radius_ = std::clamp(radius_ * factor, 0.5f, 50.0f);
}

void OrbitCamera::reset() {
    radius_ = default_radius_;
    theta_ = default_theta_;
    phi_ = default_phi_;
    target_ = glm::vec3(0.0f, 0.0f, 0.0f);
}

glm::vec3 OrbitCamera::getEye() const {
    float x = target_.x + radius_ * std::cos(phi_) * std::sin(theta_);
    float y = target_.y + radius_ * std::sin(phi_);
    float z = target_.z + radius_ * std::cos(phi_) * std::cos(theta_);
    return glm::vec3(x, y, z);
}

CameraUniforms OrbitCamera::getUniforms(int width, int height) const {
    glm::vec3 eye = getEye();
    glm::vec3 w = glm::normalize(target_ - eye);
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::vec3 u = glm::normalize(glm::cross(w, up));
    glm::vec3 v = glm::cross(u, w);

    CameraUniforms uniforms;
    uniforms.ro_x = eye.x; uniforms.ro_y = eye.y; uniforms.ro_z = eye.z;
    uniforms.u_x = u.x;   uniforms.u_y = u.y;   uniforms.u_z = u.z;
    uniforms.v_x = v.x;   uniforms.v_y = v.y;   uniforms.v_z = v.z;
    uniforms.w_x = w.x;   uniforms.w_y = w.y;   uniforms.w_z = w.z;
    uniforms.fov_scale = std::tan(fov_radians_ * 0.5f);
    uniforms.aspect = static_cast<float>(width) / std::max(static_cast<float>(height), 1.0f);

    return uniforms;
}
