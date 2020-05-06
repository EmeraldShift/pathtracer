#pragma once

#include <glm/vec3.hpp>

class ray;

class BoundingBox {
    glm::dvec3 bmin;
    glm::dvec3 bmax;

public:
    BoundingBox();

    BoundingBox(glm::dvec3 bMin, glm::dvec3 bMax);

    glm::dvec3 getMin() const { return bmin; }

    glm::dvec3 getMax() const { return bmax; }

    void setMin(glm::dvec3 bMin) {
        bmin = bMin;
    }

    void setMax(glm::dvec3 bMax) {
        bmax = bMax;
    }

    bool intersect(const ray &r, double tMax = 1.0e308) const;

    BoundingBox &operator=(const BoundingBox &target);

    void merge(const BoundingBox &bBox);
};
