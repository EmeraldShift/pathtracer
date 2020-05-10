#pragma once

#include "../gl.h"
#include "material.h"
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <memory>

class MaterialObject;

// I don't know if even an inline function can be called from CUDA, so
// this is just to be extra safe
#define invert(dd) glm::vec3(1.0 / dd[0], 1.0 / dd[1], 1.0 / dd[2])
#define RAY_EPSILON 0.00000001f

constexpr float ISECT_NO_HIT = 12345678.9f;

class isect {
public:
    __host__ __device__ isect() = default;

    __host__ __device__ isect(const isect &other) = default;

    __host__ __device__ ~isect() = default;

    __host__ __device__ void setT(float tt) { t = tt; }

    __host__ __device__ float getT() const { return t; }

    __host__ __device__ void setN(const glm::vec3 &n) { N = n; }

    __host__ __device__ glm::vec3 getN() const { return N; }

    __host__ __device__ void setMaterial(const Material &m) {
        material = m;
    }

    __host__ __device__ const Material &getMaterial() const { return material; }

    __host__ __device__ void setUVCoordinates(const glm::vec2 &coords) {
        uvCoordinates = coords;
    }

    __host__ __device__ glm::vec2 getUVCoordinates() const { return uvCoordinates; }

private:
    float t = ISECT_NO_HIT;
    glm::vec3 N = glm::vec3();
    glm::vec2 uvCoordinates = glm::vec2();
    Material material;
};

class ray {
public:
    __host__ __device__ ray(const glm::vec3 &p, const glm::vec3 &d) {
        pos = p;
        dir = d;
        invdir = invert(dir);
    };

    __host__ __device__ ray() = default;

    __host__ __device__ ray(const ray &other) = default;

    __host__ __device__ ~ray() = default;

    __host__ __device__ ray &operator=(const ray &other) = default;

    __host__ __device__ glm::vec3 at(float t) const { return pos + (t * dir); }

    __host__ __device__ glm::vec3 at(const isect &i) const { return at(i.getT()); }

    __host__ __device__ glm::vec3 getPosition() const { return pos; }

    __host__ __device__ glm::vec3 getDirection() const { return dir; }

    __host__ __device__ glm::vec3 getInverseDirection() const { return invdir; }

    __host__ __device__ void setPosition(const glm::vec3 &pp) { pos = pp; }

    __host__ __device__ void setDirection(const glm::vec3 &dd) {
        dir = dd;
        invdir = invert(dd);
    }

private:
    glm::vec3 pos = glm::vec3();
    glm::vec3 dir = glm::vec3();
    glm::vec3 invdir = glm::vec3();
};
