#pragma once

#include "material.h"
#include "../vec.h"
#include <memory>

class MaterialObject;

// I don't know if even an inline function can be called from CUDA, so
// this is just to be extra safe
#define invert(dd) f4(1.0f / dd[0], 1.0f / dd[1], 1.0f / dd[2])
#define RAY_EPSILON 0.00001f

constexpr float ISECT_NO_HIT = 12345678.9f;

class isect {
public:
    __host__ __device__ isect() = default;

    __host__ __device__ isect(const isect &other) = default;

    __host__ __device__ ~isect() = default;

    __host__ __device__ void setT(float tt) { t = tt; }

    __host__ __device__ float getT() const { return t; }

    __host__ __device__ void setN(const f4 &n) { N = n; }

    __host__ __device__ f4 getN() const { return N; }

    __host__ __device__ void setMaterial(const Material &m) {
        material = m;
    }

    __host__ __device__ const Material &getMaterial() const { return material; }

    __host__ __device__ void setUVCoordinates(const float2 &coords) {
        uvCoordinates = coords;
    }

    __host__ __device__ float2 getUVCoordinates() const { return uvCoordinates; }

private:
    float t = ISECT_NO_HIT;
    f4 N = 0;
    float2 uvCoordinates = {};
    Material material;
};

class ray {
public:
    __host__ __device__ ray(const f4 &p, const f4 &d) {
        pos = p;
        dir = d;
        invdir = 1.0 / dir;
    };

    __host__ __device__ ray(const ray &r) {
        pos = r.pos;
        dir = r.dir;
        invdir = r.invdir;
    };

    __host__ __device__ ray() = default;

    __host__ __device__ ray(const ray &other) = default;

    __host__ __device__ ~ray() = default;

    __host__ __device__ ray &operator=(const ray &other) = default;

    __host__ __device__ f4 at(float t) const { return pos + (t * dir); }

    __host__ __device__ f4 at(const isect &i) const { return at(i.getT()); }

    __host__ __device__ f4 getPosition() const { return pos; }

    __host__ __device__ f4 getDirection() const { return dir; }

    __host__ __device__ f4 getInverseDirection() const { return invdir; }

    __host__ __device__ void setPosition(const f4 &pp) { pos = pp; }

    __host__ __device__ void setDirection(const f4 &dd) {
        dir = dd;
        invdir = invert(dd);
    }

private:
    f4 pos = 0;
    f4 dir = 0;
    f4 invdir = 0;
};
