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
    void update_children_forGPU(Cluster* GPUtree, Geometry* GPUshapes);
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
    BoundedVolumeHierarchy flatten_clone(const std::vector<Geometry> &objs) const;

    __host__ __device__ bool traverseIterative(ray &r, isect &i) const;

    BoundedVolumeHierarchy clone() const;

private:
    void flatten_move(int type);
    void calculate_size(int type);
    void update_children();
};

/**
 * Compare two objects across a given axis.
 * @tparam Obj The type of object to compare
 * @tparam compareFunction The function used to perform the comparison
 * @tparam i The axis across which to measure
 * @param lhs The first object
 * @param rhs The second object
 * @return The logical difference between the two objects,
 * according to the comparison function
 */
template<int i>
static bool compare(const Geometry lhs, const Geometry rhs) {
    return ((lhs.getBoundingBox().getMax() + lhs.getBoundingBox().getMin()) / 2.0f)[i]
           < ((rhs.getBoundingBox().getMax() + rhs.getBoundingBox().getMin()) / 2.0f)[i];
}
