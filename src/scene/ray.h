#pragma once

#include "../gl.h"
#include "material.h"
#include "../gpu/cuda.h"
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <memory>

class MaterialObject;

// I don't know if even an inline function can be called frmo CUDA, so
// this is just to be extra safe
#define invert(dd) glm::dvec3(1.0 / dd[0], 1.0 / dd[1], 1.0 / dd[2])

constexpr double ISECT_NO_HIT = 12345678.9;

class isect {
public:
    CUDA_CALLABLE_MEMBER isect() = default;

    CUDA_CALLABLE_MEMBER isect(const isect &other) = default;

    CUDA_CALLABLE_MEMBER ~isect() = default;

    CUDA_CALLABLE_MEMBER void setT(double tt) { t = tt; }

    CUDA_CALLABLE_MEMBER double getT() const { return t; }

    CUDA_CALLABLE_MEMBER void setN(const glm::dvec3 &n) { N = n; }

    CUDA_CALLABLE_MEMBER glm::dvec3 getN() const { return N; }

    CUDA_CALLABLE_MEMBER void setMaterial(const Material &m) {
        material = m;
    }

    CUDA_CALLABLE_MEMBER const Material &getMaterial() const { return material; }

    CUDA_CALLABLE_MEMBER void setUVCoordinates(const glm::dvec2 &coords) {
        uvCoordinates = coords;
    }

    CUDA_CALLABLE_MEMBER glm::dvec2 getUVCoordinates() const { return uvCoordinates; }

private:
    double t = ISECT_NO_HIT;
    glm::dvec3 N = glm::dvec3();
    glm::dvec2 uvCoordinates = glm::dvec2();
    Material material;
};

class ray {
public:
    CUDA_CALLABLE_MEMBER ray(const glm::dvec3 &pos, const glm::dvec3 &dir)
            : pos(pos), dir(dir), invdir(invert(dir)) {
    };

    CUDA_CALLABLE_MEMBER ray(const ray &other) = default;

    CUDA_CALLABLE_MEMBER ~ray() = default;

    CUDA_CALLABLE_MEMBER ray &operator=(const ray &other) = default;

    CUDA_CALLABLE_MEMBER glm::dvec3 at(double t) const { return pos + (t * dir); }

    CUDA_CALLABLE_MEMBER glm::dvec3 at(const isect &i) const { return at(i.getT()); }

    CUDA_CALLABLE_MEMBER glm::dvec3 getPosition() const { return pos; }

    CUDA_CALLABLE_MEMBER glm::dvec3 getDirection() const { return dir; }

    CUDA_CALLABLE_MEMBER glm::dvec3 getInverseDirection() const { return invdir; }

    CUDA_CALLABLE_MEMBER void setPosition(const glm::dvec3 &pp) { pos = pp; }

    CUDA_CALLABLE_MEMBER void setDirection(const glm::dvec3 &dd) {
        dir = dd;
        invdir = invert(dd);
    }

private:
    glm::dvec3 pos;
    glm::dvec3 dir;
    glm::dvec3 invdir;
};

const double RAY_EPSILON = 0.00000001;
