#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"
#include "../gl.h"
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

class Camera {
public:
    Camera() = default;

    __host__ __device__ void rayThrough(float x, float y, ray &r) const;

    void setEye(const glm::vec3 &eye);

    void setLook(float, float, float, float);

    void setLook(const glm::vec3 &viewDir, const glm::vec3 &upDir);

    void setFOV(float);

    void setAspectRatio(float);

    double getAspectRatio() const { return aspectRatio; }

private:
    void update();

    float normalizedHeight = 1;
    float aspectRatio = 1;
    glm::mat3 m = glm::dmat3x3(1);
    glm::vec3 eye = glm::vec3(0, 0, 0);
    glm::vec3 u = glm::vec3(1, 0, 0);
    glm::vec3 v = glm::vec3(0, 1, 0);
    glm::vec3 look = glm::vec3(0, 1, -1);
};

#endif
