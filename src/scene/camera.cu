#include <iostream>
#include "camera.h"

#define PI 3.14159265359

__host__ __device__
void Camera::rayThrough(float x, float y, ray &r) const
// Ray through normalized window point x,y.  In normalized coordinates
// the camera's x and y vary both vary from 0 to 1.
{
    x -= 0.5;
    y -= 0.5;
    auto dir = f4m::normalize(look + x * u + y * v);
    r.setPosition(eye);
    r.setDirection(dir);
}

void Camera::setLook(float r, float i, float j, float k) {
    m[0][0] = 1.0 - 2.0 * (i * i + j * j);
    m[0][1] = 2.0 * (r * i - j * k);
    m[0][2] = 2.0 * (j * r + i * k);

    m[1][0] = 2.0 * (r * i + j * k);
    m[1][1] = 1.0 - 2.0 * (j * j + r * r);
    m[1][2] = 2.0 * (i * j - r * k);

    m[2][0] = 2.0 * (j * r - i * k);
    m[2][1] = 2.0 * (i * j + r * k);
    m[2][2] = 1.0 - 2.0 * (i * i + r * r);

    m = glm::transpose(m);
    update();
}

void
Camera::setLook(const f4 &viewDir, const f4 &upDir) {
    auto z = -viewDir;
    const auto &y = upDir;
    auto x = f4m::cross(y, z);
    m = glm::mat3x3(
            glm::vec3(x[0], x[1], x[2]),
            glm::vec3(y[0], y[1], y[2]),
            glm::vec3(z[0], z[1], z[2]));
    update();
}

void
Camera::setFOV(float fov) {
    fov /= (180.0f / 3.141592f);
    normalizedHeight = 2 * tan(fov / 2);
    update();
}

void
Camera::setAspectRatio(float ar) {
    aspectRatio = ar;
    update();
}

void
Camera::update() {
    u = m * glm::vec3(1, 0, 0) * normalizedHeight * aspectRatio;
    v = m * glm::vec3(0, 1, 0) * normalizedHeight;
    look = m * glm::vec3(0, 0, -1);
}
