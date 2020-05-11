#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"
#include "../gl.h"
#include "../vec.h"
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

class Camera {
public:
    Camera() = default;

    __host__ __device__ void rayThrough(float x, float y, ray &r) const;

    void setEye(const f4 &e) { eye = e; };

    void setLook(float, float, float, float);

    void setLook(const f4 &viewDir, const f4 &upDir);

    void setFOV(float);

    void setAspectRatio(float);

    double getAspectRatio() const { return aspectRatio; }

private:
    void update();

    float normalizedHeight = 1;
    float aspectRatio = 1;
    glm::mat3 m = glm::mat3x3(1);
    f4 eye;
    f4 u = {1, 0, 0};
    f4 v = {0, 1, 0};
    f4 look = {0, 1, -1};
};

#endif
