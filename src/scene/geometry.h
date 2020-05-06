#pragma once

#include "bbox.h"
#include "ray.h"

#include <vector>

class Geometry {
public:
    Geometry() = default;

    const BoundingBox &getBoundingBox() const { return bounds; }

    virtual bool intersect(ray &r, isect &i) const = 0;

    static bool compare(Geometry *const lhs, Geometry *const rhs, int i) {
        return ((lhs->bounds.getMax() + lhs->bounds.getMin()) / 2.0)[i]
               < ((rhs->bounds.getMax() + rhs->bounds.getMin()) / 2.0)[i];
    }

    static double spread(std::vector<Geometry *> &v, int i) {
        return ((v[v.size() - 1]->bounds.getMax() + v[v.size() - 1]->bounds.getMin()) / 2.0)[i]
               - ((v[0]->bounds.getMax() + v[0]->bounds.getMin()) / 2.0)[i];
    }

protected:
    BoundingBox bounds;
};