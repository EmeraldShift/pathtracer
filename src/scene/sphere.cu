#include "sphere.h"
#include "ray.h"

__host__ __device__
bool Sphere::intersect(const ray &r, isect &i) const {
    auto pos = r.getPosition();
    auto dir = f4m::normalize(r.getDirection());
    auto toSphere = position - r.getPosition();
    auto dot = f4m::dot(dir, toSphere);

    auto discriminant = dot * dot - f4m::length2(toSphere) + radius * radius;
    if (discriminant < 0)
        return false;

    discriminant = sqrtf(discriminant);
    auto t2 = dot + discriminant;
    if (t2 < RAY_EPSILON)
        return false;

    auto t1 = dot - discriminant;

    if (t1 > RAY_EPSILON && t1 < i.getT()) {
        auto col = mat.kd(i);
        i.setT(t1);
        i.setMaterial(mat);
        i.setN(f4m::normalize(r.at(t1) - position));
        return true;
    } else if (t2 < i.getT()) {
        i.setT(t2);
        i.setMaterial(mat);
        i.setN(f4m::normalize(r.at(t2) - position));
        return true;
    }
    return false;
}