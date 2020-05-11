#pragma once

#include "material.h"

class Sphere {
public:
    Sphere(const Material &mat, f4 position, float radius)
            : mat(mat), position(position), radius(radius) {}

    __host__ __device__ bool intersect(const ray &r, isect &i) const;

private:
    Material mat;
    f4 position;
    float radius;

    friend class Geometry;
};
