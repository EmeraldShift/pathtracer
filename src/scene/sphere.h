#pragma once

#include "geometry.h"
#include "material.h"

class Sphere : public Geometry {
public:
    Sphere(const Material &mat, glm::dvec3 position, double radius)
            : mat(mat), position(position), radius(radius) {
        bounds = BoundingBox(
                glm::dvec3(position[0] - radius, position[1] - radius, position[2] - radius),
                glm::dvec3(position[0] + radius, position[1] + radius, position[2] + radius));
        isSphere = true;
    }

    CUDA_CALLABLE_MEMBER static bool intersect(void *obj, ray &r, isect &i);

    Sphere *clone() const override;

private:
    Material mat;
    glm::dvec3 position;
    double radius;
};