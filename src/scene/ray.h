#pragma once

#include "../gl.h"
#include "material.h"
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <memory>

class MaterialObject;

// I don't know if even an inline function can be called frmo CUDA, so
// this is just to be extra safe
#define invert(dd) glm::dvec3(1.0 / dd[0], 1.0 / dd[1], 1.0 / dd[2])
#define RAY_EPSILON 0.00000001

constexpr double ISECT_NO_HIT = 12345678.9;

class isect {
public:
    __host__ __device__ isect() = default;

    __host__ __device__ isect(const isect &other) = default;

    __host__ __device__ ~isect() = default;

    __host__ __device__ void setT(double tt) { t = tt; }

    __host__ __device__ double getT() const { return t; }

    __host__ __device__ void setN(const glm::dvec3 &n) { N = n; }

    __host__ __device__ glm::dvec3 getN() const { return N; }

    __host__ __device__ void setMaterial(const Material &m) {
        material = m;
    }

    __host__ __device__ const Material &getMaterial() const { return material; }

    __host__ __device__ void setUVCoordinates(const glm::dvec2 &coords) {
        uvCoordinates = coords;
    }

    __host__ __device__ glm::dvec2 getUVCoordinates() const { return uvCoordinates; }

private:
    double t = ISECT_NO_HIT;
    glm::dvec3 N = glm::dvec3();
    glm::dvec2 uvCoordinates = glm::dvec2();
    Material material;
};

class ray {
public:
    __host__ __device__ ray(const glm::dvec3 &p, const glm::dvec3 &d) {
        pos = p;
        dir = d;
        invdir = invert(dir);
    };

    __host__ __device__ ray() = default;

    __host__ __device__ ray(const ray &other) = default;

    __host__ __device__ ~ray() = default;

    __host__ __device__ ray &operator=(const ray &other) = default;

    __host__ __device__ glm::dvec3 at(double t) const { return pos + (t * dir); }

    __host__ __device__ glm::dvec3 at(const isect &i) const { return at(i.getT()); }

    __host__ __device__ glm::dvec3 getPosition() const { return pos; }

    __host__ __device__ glm::dvec3 getDirection() const { return dir; }

    __host__ __device__ glm::dvec3 getInverseDirection() const { return invdir; }

    __host__ __device__ void setPosition(const glm::dvec3 &pp) { pos = pp; }

    __host__ __device__ void setDirection(const glm::dvec3 &dd) {
        dir = dd;
        invdir = invert(dd);
    }

private:
    glm::dvec3 pos = glm::dvec3();
    glm::dvec3 dir = glm::dvec3();
    glm::dvec3 invdir = glm::dvec3();
};
