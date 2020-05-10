#pragma once

#include "bbox.h"
#include "ray.h"

#include <vector>

class Geometry {
public:
    Geometry() = default;

    const BoundingBox &getBoundingBox() const { return bounds; }

    virtual Geometry *clone() const = 0;

    bool isSphere = false;

protected:
    BoundingBox bounds;
};