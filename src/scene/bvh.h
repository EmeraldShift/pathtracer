#pragma once

#include "geometry.h"
#include "ray.h"
#include "bbox.h"
#include "sphere.h"
#include "trimesh.h"
#include <set>
#include <queue>
#include <iostream>

/**
 * A cluster class to represent a collection of spatially grouped objects
 * in the bounding volume hierarchy.
 * @tparam Obj The type of primitive to store
 */
struct Cluster {
    BoundingBox bbox;
    Cluster *left, *right;
    Geometry *obj;

    bool intersect(ray &r, isect &i) const;

    Cluster *clone() const;
};

////////////////////////
/// Helper Functions ///
////////////////////////

/**
 * A tree-like spatial acceleration data structure to facilitate ray
 * intersection tests.
 * @tparam Obj The type object contained in the hierarchy
 * @tparam compareFunction A function to compare two objects
 * @tparam spreadFunction A function to find the spread across a
 * list of objects
 */
class BoundedVolumeHierarchy {
    Cluster *root = nullptr;

public:
    void construct(std::vector<Geometry *> &objs);

    bool traverse(ray &r, isect &i) const;

    __host__ __device__ bool traverseIterative(ray &r, isect &i) const;

    BoundedVolumeHierarchy clone() const;
};
