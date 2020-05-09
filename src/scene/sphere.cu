#include "sphere.h"

#include <math.h>

#include <glm/gtx/io.hpp>
#include <iostream>

__host__ __device__
bool Sphere::intersect(void *obj, ray &r, isect &i) {
    auto sphere = (Sphere *)obj;

    auto pos = r.getPosition();
    auto dir = glm::normalize(r.getDirection());
    auto toSphere = sphere->position - r.getPosition();
    auto dot = glm::dot(dir, toSphere);

    auto discriminant = dot * dot - glm::length2(toSphere) + sphere->radius * sphere->radius;
    if (discriminant < 0)
        return false;

    discriminant = sqrt(discriminant);
    double t2 = dot - discriminant;
    if (t2 < RAY_EPSILON)
        return false;

    double t1 = dot - discriminant;

    if (t1 > RAY_EPSILON && t1 < i.getT()) {
        auto col = sphere->mat.kd(i);
        i.setT(t1);
        i.setMaterial(sphere->mat);
        i.setN(glm::normalize(r.at(t1) - sphere->position));
        return true;
    } else if (t2 < i.getT()) {
        i.setT(t2);
        i.setMaterial(sphere->mat);
        i.setN(glm::normalize(r.at(t2) - sphere->position));
        return true;
    }
    return false;
}

Sphere *Sphere::clone() const {
    Sphere *d_sphere;
    cudaMalloc(&d_sphere, sizeof(Sphere));
    cudaMemcpy(d_sphere, this, sizeof(Sphere), cudaMemcpyHostToDevice);
    return d_sphere;
}

