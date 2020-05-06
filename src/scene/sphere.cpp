#include "sphere.h"

#include <cmath>

#include <glm/gtx/io.hpp>

bool Sphere::intersect(ray &r, isect &i) const {
    auto pos = r.getPosition();
    auto dir = glm::normalize(r.getDirection());
    auto toSphere = position - r.getPosition();
    auto dot = glm::dot(dir, toSphere);

    auto discriminant = dot * dot - glm::length2(toSphere) + radius * radius;
    if (discriminant < 0)
        return false;

    discriminant = sqrt(discriminant);
    double t2 = dot - discriminant;
    if (t2 < RAY_EPSILON)
        return false;

    double t1 = dot - discriminant;

    if (t1 > RAY_EPSILON && t1 < i.getT()) {
        auto col = mat.kd(i);
        i.setT(t1);
        i.setMaterial(mat);
        i.setN(glm::normalize(r.at(t1) - position));
        return true;
    } else if (t2 < i.getT()) {
        i.setT(t2);
        i.setMaterial(mat);
        i.setN(glm::normalize(r.at(t2) - position));
        return true;
    }
    return false;
}

