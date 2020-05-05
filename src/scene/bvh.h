#pragma once

#include "../gl.h"
#include <set>
#include <queue>
#include <iostream>
#include "ray.h"
#include "bbox.h"

/**
 * A cluster class to represent a collection of spatially grouped objects
 * in the bounding volume hierarchy.
 * @tparam Obj The type of primitive to store
 */
template<typename Obj>
struct Cluster {
    BoundingBox bbox;
    Cluster *left, *right;
    Obj obj;

    bool intersect(ray &r, isect &i) {
        if (obj) {
            isect cur;
            if (obj->intersect(r, cur) && cur.getT() < i.getT()) {
                i = cur;
                return true;
            }
            return false;
        } else {
            double d = i.getT();
            return (left->bbox.intersect(r, d + RAY_EPSILON) && left->intersect(r, i))
                   | (right->bbox.intersect(r, d + RAY_EPSILON) && right->intersect(r, i));
        }
    }
};

////////////////////////
/// Helper Functions ///
////////////////////////

/**
 * Find the spread (max-min) of a given vector, across a given axis.
 * @tparam Obj The type of object stored in the vector
 * @tparam spreadFunction The function used to find the spread
 * @tparam i The axis across which to measure
 * @param v The vector to measure
 * @return The spread of the vector
 */
template<typename Obj, double(*spreadFunction)(std::vector<Obj> &, int), int i>
static double spread(std::vector<Obj> &v) {
    return spreadFunction(v, i);
}

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
template<typename Obj, bool (*compareFunction)(const Obj, const Obj, int), int i>
static bool compare(const Obj lhs, const Obj rhs) {
    return compareFunction(lhs, rhs, i);
}

static double area(BoundingBox &bb) {
    auto min = bb.getMin();
    auto max = bb.getMax();
    auto x = max[0] - min[0];
    auto y = max[1] - min[1];
    auto z = max[2] - min[2];
    auto a = 2.0 * (x * y + y * z + z * x);
    return a == 0 ? 1e308 : a; // Default empty bbox to infinite area
}

static double sah(int n_l, double s_l, int n_r, double s_r, double s_p) {
    return 1 + n_l * (s_l / s_p) + n_r * (s_r / s_p);
}

/**
 * Encapsulates a set of objects, which have already been sorted on each axis,
 * into a Cluster instance which contains them. This function recursively
 * organizes large sets of objects into clusters which contain other clusters.
 * @tparam Obj The type of object to store
 * @tparam spreadFunction The function used to find the spread
 * @param xvec The set of objects sorted by x-axis location
 * @param yvec The set of objects sorted by y-axis location
 * @param zvec The set of objects sorted by z-axis location
 * @return A new cluster containing all the objects
 */
template<typename Obj, double (*spreadFunction)(std::vector<Obj> &, int)>
static Cluster<Obj> *genCluster(std::vector<Obj> &xvec,
                                std::vector<Obj> &yvec,
                                std::vector<Obj> &zvec) {
    auto c = new Cluster<Obj>();

    // Leaf node
    if (xvec.size() == 1) {
        c->obj = xvec[0];
        c->bbox = BoundingBox(xvec[0]->getBoundingBox().getMin(), xvec[0]->getBoundingBox().getMax());
        return c;
    }

    auto bestAxis = -1;
    auto bestSplit = -1;
    auto size = xvec.size();
    auto bestCost = size;
    double *leftArea = new double[size];
    std::vector<Obj> vecs[] = {xvec, yvec, zvec};
    for (int axis = 0; axis < 3; axis++) {
        BoundingBox bb;
        for (int i = 0; i < size; i++) {
            bb.merge(vecs[axis][i]->getBoundingBox());
            leftArea[i] = area(bb);
        }

        // leftArea[i] contains the (i+1) leftmost items

        bb = BoundingBox();
        for (int i = size - 1; i > 0; i--) {
            bb.merge(xvec[i]->getBoundingBox());
            double cost = sah(i, leftArea[i - 1], size - i, area(bb), leftArea[size]);
            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestSplit = i;
            }
        }
    }
    delete[] leftArea;

    // Defaults if no split better than leaf
    if (bestAxis == -1) {
        // Find max spreads
        auto xspread = spread<Obj, spreadFunction, 0>(xvec);
        auto yspread = spread<Obj, spreadFunction, 1>(yvec);
        auto zspread = spread<Obj, spreadFunction, 2>(zvec);

        // Pick widest
        auto splitset = 2;
        if (yspread > zspread)
            splitset = 1;
        if (xspread > yspread && xspread > zspread)
            splitset = 0;

        bestAxis = splitset;
        bestSplit = size / 2;
    }

    auto &splitset = vecs[bestAxis];
    std::vector<Obj> xleft, yleft, zleft, xright, yright, zright;
    for (size_t i = 0; i < bestSplit; i++) {
        auto f = splitset[i];
        xleft.push_back(f);
        yleft.push_back(f);
        zleft.push_back(f);
    }
    for (size_t i = bestSplit; i < size; i++) {
        auto f = splitset[i];
        xright.push_back(f);
        yright.push_back(f);
        zright.push_back(f);
    }

    c->left = genCluster<Obj, spreadFunction>(xleft, yleft, zleft);
    c->right = genCluster<Obj, spreadFunction>(xright, yright, zright);
    c->bbox = BoundingBox(glm::min(c->left->bbox.getMin(), c->right->bbox.getMin()),
                          glm::max(c->left->bbox.getMax(), c->right->bbox.getMax()));
    return c;
}

/**
 * A tree-like spatial acceleration data structure to facilitate ray
 * intersection tests.
 * @tparam Obj The type object contained in the hierarchy
 * @tparam compareFunction A function to compare two objects
 * @tparam spreadFunction A function to find the spread across a
 * list of objects
 */
template<typename Obj,
        bool (*compareFunction)(const Obj a, const Obj b, int),
        double (*spreadFunction)(std::vector<Obj> &, int)>
class BoundedVolumeHierarchy {
    Cluster<Obj> *root = nullptr;

public:
    void construct(std::vector<Obj> &objs) {
        if (objs.size() == 0)
            return;

        // Create sorted sets
        auto xset = std::multiset<Obj, bool (*)(const Obj, const Obj)>(compare<Obj, compareFunction, 0>);
        auto yset = std::multiset<Obj, bool (*)(const Obj, const Obj)>(compare<Obj, compareFunction, 1>);
        auto zset = std::multiset<Obj, bool (*)(const Obj, const Obj)>(compare<Obj, compareFunction, 2>);
        for (auto f : objs) {
            xset.insert(f);
            yset.insert(f);
            zset.insert(f);
        }

        // Copy into vectors
        auto xvec = std::vector<Obj>();
        auto yvec = std::vector<Obj>();
        auto zvec = std::vector<Obj>();
        std::copy(xset.begin(), xset.end(), std::back_inserter(xvec));
        std::copy(yset.begin(), yset.end(), std::back_inserter(yvec));
        std::copy(zset.begin(), zset.end(), std::back_inserter(zvec));

        root = genCluster<Obj, spreadFunction>(xvec, yvec, zvec);
    }

    bool traverse(ray &r, isect &i) const {
        return root == nullptr ? false : root->intersect(r, i);
    }
};
