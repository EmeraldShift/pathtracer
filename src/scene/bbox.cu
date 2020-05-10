#include "ray.h"
#include "bbox.h"

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a < b ? b : a)

BoundingBox::BoundingBox() : bmin(glm::vec3(0)), bmax(glm::vec3(0)) {
}

BoundingBox::BoundingBox(glm::vec3 bMin, glm::vec3 bMax)
        : bmin(bMin), bmax(bMax) {
}

__host__ __device__
bool BoundingBox::intersect(const ray &r, float tMax /* = 1.0e308 */) const {
    auto p = r.getPosition();
    auto d = r.getDirection();
    auto n = r.getInverseDirection();
    float tMin = -1.0e308;//, tMax = 1.0e308;

    float t10 = (bmin[0] - p[0]) * n[0];
    float t11 = (bmin[1] - p[1]) * n[1];
    float t12 = (bmin[2] - p[2]) * n[2];
    float t20 = (bmax[0] - p[0]) * n[0];
    float t21 = (bmax[1] - p[1]) * n[1];
    float t22 = (bmax[2] - p[2]) * n[2];
    tMin = max(tMin, min(t10, t20));
    tMin = max(tMin, min(t11, t21));
    tMin = max(tMin, min(t12, t22));
    tMax = min(tMax, max(t10, t20));
    tMax = min(tMax, max(t11, t21));
    tMax = min(tMax, max(t12, t22));
    return tMin <= tMax && tMax > 0;
}

BoundingBox &BoundingBox::operator=(const BoundingBox &target) {
    bmin = target.bmin;
    bmax = target.bmax;
    return *this;
}

void BoundingBox::merge(const BoundingBox &bBox) {
    for (int axis = 0; axis < 3; axis++) {
        if (bBox.bmin[axis] < bmin[axis])
            bmin[axis] = bBox.bmin[axis];
        if (bBox.bmax[axis] > bmax[axis])
            bmax[axis] = bBox.bmax[axis];
    }
}
