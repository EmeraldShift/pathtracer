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

 //64 BYTES
struct Cluster {
    BoundingBox bbox;
    Cluster *left, *right;
    Geometry *obj;
    int left_size, right_size;

    int flatten();

    bool intersect(ray &r, isect &i) const;

    Cluster *clone() const;
    int flatten_move(Cluster* tree, int subtree, int type);
    int calculate_size();
    void update_children(Cluster* tree);
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
    int size = 0;
    int root_index = 0;

public:
    void construct(std::vector<Geometry *> &objs);

    bool traverse(ray &r, isect &i) const;

    void flatten();
    BoundedVolumeHierarchy flatten_clone() const;

    __host__ __device__ bool traverseIterative(ray &r, isect &i) const;

    BoundedVolumeHierarchy clone() const;

private:
    void flatten_move(int type);
    void calculate_size(int type);
    void update_children();
};
