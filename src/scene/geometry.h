#pragma once

#include "bbox.h"
#include "ray.h"
#include "sphere.h"
#include "trimesh.h"

#include <vector>

class Geometry {
public:
    Geometry(const Sphere &sphere) : obj(sphere), isSphere(true) {
        bounds = {sphere.position - sphere.radius, sphere.position + sphere.radius};
    }

    Geometry(const TrimeshFace &face) : obj(face), isSphere(false) {
        bounds.setMax(f4m::max(face.vertices[0], face.vertices[1]));
        bounds.setMin(f4m::min(face.vertices[0], face.vertices[1]));
        bounds.setMax(f4m::max(bounds.getMax(), face.vertices[2]));
        bounds.setMin(f4m::min(bounds.getMin(), face.vertices[2]));
    }

    const BoundingBox &getBoundingBox() const { return bounds; }

    __host__ __device__
    bool intersect(const ray &r, isect &i) const {
        return isSphere ? obj.sphere.intersect(r, i) : obj.face.intersect(r, i);
    }

    Geometry *clone() const;

private:
    bool isSphere = false;
    BoundingBox bounds;

    union obj {
        obj(const Sphere &sphere) : sphere(sphere) {}

        obj(const TrimeshFace &face) : face(face) {}

        Sphere sphere;
        TrimeshFace face;
    } obj;
};