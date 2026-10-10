#pragma once
#include <glm/glm.hpp>

struct CameraUniforms {
    float ro_x, ro_y, ro_z;
    float u_x, u_y, u_z;
    float v_x, v_y, v_z;
    float w_x, w_y, w_z;
    float fov_scale;
    float aspect;
};

class OrbitCamera {
public:
    OrbitCamera(float radius = 5.0f, float theta = 0.8f, float phi = 0.4f);

    void rotate(float delta_theta, float delta_phi);
    void pan(float delta_x, float delta_y);
    void zoom(float factor);
    void reset();

    glm::vec3 getEye() const;
    glm::vec3 getTarget() const { return target_; }
    CameraUniforms getUniforms(int width, int height) const;

private:
    float default_radius_;
    float default_theta_;
    float default_phi_;

    float radius_;
    float theta_;
    float phi_;
    glm::vec3 target_;
    float fov_radians_;
};
