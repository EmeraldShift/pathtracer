#pragma once

#include "geometry.h"
#include "material.h"

class Sphere : public Geometry {
public:
    Sphere(const Material &mat, glm::vec3 position, float radius)
            : mat(mat), position(position), radius(radius) {
        bounds = BoundingBox(
                glm::vec3(position[0] - radius, position[1] - radius, position[2] - radius),
                glm::vec3(position[0] + radius, position[1] + radius, position[2] + radius));
        isSphere = true;
    }

    __host__ __device__ static bool intersect(void *obj, ray &r, isect &i);

    Sphere *clone() const override;

private:
    Material mat;
    glm::vec3 position;
    double radius;
};
