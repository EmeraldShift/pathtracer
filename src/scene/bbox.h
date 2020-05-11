#pragma once

#include <glm/vec3.hpp>
#include "../vec.h"

class ray;

class BoundingBox {
    f4 bmin;
    f4 bmax;

public:
    BoundingBox();

    BoundingBox(f4 bMin, f4 bMax);

    f4 getMin() const { return bmin; }

    f4 getMax() const { return bmax; }

    void setMin(f4 bMin) {
        bmin = bMin;
    }

    void setMax(f4 bMax) {
        bmax = bMax;
    }

    __host__ __device__ bool intersect(const ray &r, float tMax = 1.0e38f) const;

    BoundingBox &operator=(const BoundingBox &target) = default;

    void merge(const BoundingBox &bBox);
};
