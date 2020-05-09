#include "bvh.h"
#include "sphere.h"
#include "trimesh.h"

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
static bool compare(const Geometry *lhs, const Geometry *rhs) {
    return ((lhs->getBoundingBox().getMax() + lhs->getBoundingBox().getMin()) / 2.0)[i]
           < ((rhs->getBoundingBox().getMax() + rhs->getBoundingBox().getMin()) / 2.0)[i];
}

/**
 * Find the spread (max-min) of a given vector, across a given axis.
 * @tparam Obj The type of object stored in the vector
 * @tparam spreadFunction The function used to find the spread
 * @tparam i The axis across which to measure
 * @param v The vector to measure
 * @return The spread of the vector
 */
static double spread(std::vector<Geometry *> &v, int i) {
    return ((v[v.size() - 1]->getBoundingBox().getMax() + v[v.size() - 1]->getBoundingBox().getMin()) / 2.0)[i]
           - ((v[0]->getBoundingBox().getMax() + v[0]->getBoundingBox().getMin()) / 2.0)[i];
}

static double area(BoundingBox &bb) {
    auto min = bb.getMin();
    auto max = bb.getMax();
    auto x = max[0] - min[0];
    auto y = max[1] - min[1];
    auto z = max[2] - min[2];
    auto a = 2.0 * (x * y + y * z + z * x);
    return a < RAY_EPSILON ? 1e308 : a; // Default empty bbox to infinite area
}

static double sah(int n_l, double s_l, int n_r, double s_r, double s_p) {
    return 1 + 3.0 * (n_l * (s_l / s_p) + n_r * (s_r / s_p));
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
static Cluster *genCluster(std::vector<Geometry *> &xvec,
                           std::vector<Geometry *> &yvec,
                           std::vector<Geometry *> &zvec) {
    auto c = new Cluster();

    // Leaf node
    if (xvec.size() == 1) {
        c->obj = xvec[0];
        c->bbox = BoundingBox(xvec[0]->getBoundingBox().getMin(), xvec[0]->getBoundingBox().getMax());
        return c;
    }

    auto bestAxis = -1;
    auto bestSplit = -1;
    auto size = xvec.size();
    auto bestCost = 3.0 * size;
    auto leftArea = new double[size];
    auto allBounds = BoundingBox();
    for (int i = 0; i < size; i++)
        allBounds.merge(xvec[i]->getBoundingBox());

    std::vector<Geometry *> vecs[] = {xvec, yvec, zvec};
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
            double cost = sah(i, leftArea[i - 1], size - i, area(bb), leftArea[size - 1]);
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
        auto xspread = spread(xvec, 0);
        auto yspread = spread(yvec, 1);
        auto zspread = spread(zvec, 2);

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
    std::vector<Geometry *> xleft, yleft, zleft, xright, yright, zright;
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
    c->left = genCluster(xleft, yleft, zleft);
    c->right = genCluster(xright, yright, zright);
    c->bbox = BoundingBox(glm::min(c->left->bbox.getMin(), c->right->bbox.getMin()),
                          glm::max(c->left->bbox.getMax(), c->right->bbox.getMax()));
    return c;
}

void BoundedVolumeHierarchy::construct(std::vector<Geometry *> &objs) {
    if (objs.size() == 0)
        return;

    // Create sorted sets
    auto xset = std::multiset<Geometry *, bool (*)(const Geometry *, const Geometry *)>(compare<0>);
    auto yset = std::multiset<Geometry *, bool (*)(const Geometry *, const Geometry *)>(compare<1>);
    auto zset = std::multiset<Geometry *, bool (*)(const Geometry *, const Geometry *)>(compare<2>);
    for (auto f : objs) {
        xset.insert(f);
        yset.insert(f);
        zset.insert(f);
    }

    // Copy into vectors
    auto xvec = std::vector<Geometry *>();
    auto yvec = std::vector<Geometry *>();
    auto zvec = std::vector<Geometry *>();
    std::copy(xset.begin(), xset.end(), std::back_inserter(xvec));
    std::copy(yset.begin(), yset.end(), std::back_inserter(yvec));
    std::copy(zset.begin(), zset.end(), std::back_inserter(zvec));

    root = genCluster(xvec, yvec, zvec);
}

bool BoundedVolumeHierarchy::traverse(ray &r, isect &i) const {
    return root == nullptr ? false : root->intersect(r, i);
}

bool BoundedVolumeHierarchy::traverseIterative(ray &r, isect &i) const {
    if (root == nullptr)
        return false;
    if (root->obj)
        return root->obj->isSphere
               ? Sphere::intersect(root->obj, r, i)
               : TrimeshFace::intersect(root->obj, r, i);

    bool have_one = false;
    Cluster *stack[64];
    Cluster **sp = stack;
    *sp++ = nullptr;
    Cluster *current = root;
    while (current) {
        bool overlapL = current->left->bbox.intersect(r, i.getT());
        bool overlapR = current->right->bbox.intersect(r, i.getT());
        if (overlapL && current->left->obj) {
            have_one |= current->left->obj->isSphere
                    ? Sphere::intersect(current->left->obj, r, i)
                    : TrimeshFace::intersect(current->left->obj, r, i);
        }
        if (overlapR && current->right->obj) {
            have_one |= current->right->obj->isSphere
                        ? Sphere::intersect(current->right->obj, r, i)
                        : TrimeshFace::intersect(current->right->obj, r, i);
        }

        bool traverseL = (overlapL && !current->left->obj);
        bool traverseR = (overlapR && !current->right->obj);

        if (!traverseL && !traverseR) {
            current = *--sp;
        } else {
            if (traverseL && traverseR)
                *sp++ = current->right;
            current = traverseL ? current->left : current->right;
        }
    }
    return have_one;
}

Cluster *Cluster::clone() const {
    Cluster *d_cluster;
    cudaMalloc(&d_cluster, sizeof(Cluster));
    cudaMemcpy(d_cluster, this, sizeof(Cluster), cudaMemcpyHostToDevice);

    // Children
    Cluster *l = left ? left->clone() : nullptr;
    Cluster *r = right ? right->clone() : nullptr;
    Geometry *o = obj ? obj->clone() : nullptr;
    cudaMemcpy(&d_cluster->left, &l, sizeof(Cluster *), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_cluster->right, &r, sizeof(Cluster *), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_cluster->obj, &o, sizeof(Geometry *), cudaMemcpyHostToDevice);
    return d_cluster;
}

BoundedVolumeHierarchy BoundedVolumeHierarchy::clone() const {
    BoundedVolumeHierarchy d_bvh;
    d_bvh.root = root->clone();
    return d_bvh;
}