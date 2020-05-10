#pragma once

#include <glm/vec3.hpp>

class ray;

class BoundingBox {
    glm::vec3 bmin;
    glm::vec3 bmax;

public:
    BoundingBox();

    BoundingBox(glm::vec3 bMin, glm::vec3 bMax);

    glm::vec3 getMin() const { return bmin; }

    glm::vec3 getMax() const { return bmax; }

    void setMin(glm::vec3 bMin) {
        bmin = bMin;
    }

    void setMax(glm::vec3 bMax) {
        bmax = bMax;
    }

    __host__ __device__ bool intersect(const ray &r, float tMax = 1.0e308) const;

    BoundingBox &operator=(const BoundingBox &target);

    void merge(const BoundingBox &bBox);
};
