//
// ray.h
//
// The low-level classes used by ray tracing: the ray and isect classes.
//

#ifndef __RAY_H__
#define __RAY_H__

// who the hell cares if my identifiers are longer than 255 characters:
#pragma warning(disable : 4786)

#include "../gl.h"
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <memory>
#include "material.h"


class MaterialSceneObject;

class isect;

/*
 * ray_thread_id: a thread local variable for statistical purpose.
 */
extern thread_local unsigned int ray_thread_id;

// A ray has a position where the ray starts, and a direction (which should
// always be normalized!)

static glm::dvec3 invert(glm::dvec3 dd) {
    return glm::dvec3(1.0 / dd[0], 1.0 / dd[1], 1.0 / dd[2]);
}

class ray {
public:
    ray(const glm::dvec3 &pos, const glm::dvec3 &dir);

    ray(const ray &other);

    ~ray();

    ray &operator=(const ray &other);

    glm::dvec3 at(double t) const { return pos + (t * dir); }

    glm::dvec3 at(const isect &i) const;

    glm::dvec3 getPosition() const { return pos; }

    glm::dvec3 getDirection() const { return dir; }

    glm::dvec3 getInverseDirection() const { return invdir; }

    void setPosition(const glm::dvec3 &pp) { pos = pp; }

    void setDirection(const glm::dvec3 &dd) {
        dir = dd;
        invdir = invert(dd);
    }

private:
    glm::dvec3 pos;
    glm::dvec3 dir;
    glm::dvec3 invdir;
};

// The description of an intersection point.

constexpr double ISECT_NO_HIT = 12345678.9;

class isect {
public:
    isect() : t(ISECT_NO_HIT), N() {}

    isect(const isect &other) {
        copyFromOther(other);
    }

    ~isect() {}

    isect &operator=(const isect &other) {
        copyFromOther(other);
        return *this;
    }

    // Get/Set Time of flight
    void setT(double tt) { t = tt; }

    double getT() const { return t; }

    // Get/Set surface normal at this intersection.
    void setN(const glm::dvec3 &n) { N = n; }

    glm::dvec3 getN() const { return N; }

    void setMaterial(const Material &m) {
        material = m;
    }
    const Material &getMaterial() const { return material; }

    void setUVCoordinates(const glm::dvec2 &coords) {
        uvCoordinates = coords;
    }

    glm::dvec2 getUVCoordinates() const { return uvCoordinates; }

    void setBary(const glm::dvec3 &weights) { bary = weights; }

    void setBary(const double alpha, const double beta, const double gamma) {
        setBary(glm::dvec3(alpha, beta, gamma));
    }

private:
    void copyFromOther(const isect &other) {
        if (this == &other)
            return;
        t = other.t;
        N = other.N;
        bary = other.bary;
        uvCoordinates = other.uvCoordinates;
        material = other.material;
    }

    double t;
    glm::dvec3 N;
    glm::dvec2 uvCoordinates;
    glm::dvec3 bary;

    // if this intersection has its own material
    // (as opposed to one in its associated object)
    // as in the case where the material was interpolated
    Material material;
};

const double RAY_EPSILON = 0.00000001;

#endif // __RAY_H__
