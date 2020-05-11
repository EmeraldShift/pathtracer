#include "bvh.h"

bool Cluster::intersect(ray &r, isect &i) const {
    if (obj) {
        return obj->intersect(r, i);
    } else {
        return (left->bbox.intersect(r, i.getT()) && left->intersect(r, i))
               | (right->bbox.intersect(r, i.getT()) && right->intersect(r, i));
    }
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
template<int i>
static bool compare(const Geometry *lhs, const Geometry *rhs) {
    return ((lhs->getBoundingBox().getMax() + lhs->getBoundingBox().getMin()) / 2.0f)[i]
           < ((rhs->getBoundingBox().getMax() + rhs->getBoundingBox().getMin()) / 2.0f)[i];
}

/**
 * Find the spread (max-min) of a given vector, across a given axis.
 * @tparam Obj The type of object stored in the vector
 * @tparam spreadFunction The function used to find the spread
 * @tparam i The axis across which to measure
 * @param v The vector to measure
 * @return The spread of the vector
 */
static float spread(std::vector<Geometry *> &v, int i) {
    return ((v[v.size() - 1]->getBoundingBox().getMax() + v[v.size() - 1]->getBoundingBox().getMin()) / 2.0f)[i]
           - ((v[0]->getBoundingBox().getMax() + v[0]->getBoundingBox().getMin()) / 2.0f)[i];
}

static float area(BoundingBox &bb) {
    auto min = bb.getMin();
    auto max = bb.getMax();
    auto x = max[0] - min[0];
    auto y = max[1] - min[1];
    auto z = max[2] - min[2];
    auto a = 2.0 * (x * y + y * z + z * x);
    return a < RAY_EPSILON ? 1e308 : a; // Default empty bbox to infinite area
}

static float sah(int n_l, float s_l, int n_r, float s_r, float s_p) {
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
    auto leftArea = new float[size];
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
            auto cost = sah(i, leftArea[i - 1], size - i, area(bb), leftArea[size - 1]);
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
    c->bbox = BoundingBox(f4m::min(c->left->bbox.getMin(), c->right->bbox.getMin()),
                          f4m::max(c->left->bbox.getMax(), c->right->bbox.getMax()));
    return c;
}

void BoundedVolumeHierarchy::construct(std::vector<Geometry *> &objs) {
    if (objs.empty())
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

__host__ __device__
bool BoundedVolumeHierarchy::traverseIterative(ray &r, isect &i) const {
    if (root == nullptr)
        return false;
    if (root->obj)
        return root->obj->intersect(r, i);
    bool have_one = false;
    Cluster *stack[64];
    Cluster **sp = stack;
    *sp++ = nullptr;
    Cluster *current = root;
    while (current) {
        bool overlapL = current->left->bbox.intersect(r, i.getT());
        bool overlapR = current->right->bbox.intersect(r, i.getT());
        if (overlapL && current->left->obj)
            have_one |= current->left->obj->intersect(r, i);
        if (overlapR && current->right->obj)
            have_one |= current->right->obj->intersect(r, i);
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

//TODO BVH flatten and then the order we're doing threads
//cache line 128 on GPU, 32 on CPU, make 32-byte alligned if possible
//make it a union of inner node and child?

/*
    Takes a non-flat tree and flattens it on the CPU, calculates L,R offset, frees node memory
    Returns a BoundedVolumeHierarchy where the nodes are in contigous memory
    Nodes are in inorder traversal, so that entire subtrees can be cached together
*/
void BoundedVolumeHierarchy::flatten() {
    //assumes BVH has been generated
    int type = 0; //TODO optargs

    //recursively calculate offsets subtrees
    calculate_size(type);

    //copy in specified order into array
    flatten_move(type);

    //update_children pointers so left, right actually point in array
    update_children();

}
void BoundedVolumeHierarchy::flatten_move(int type) {
    //malloc tree
    //TODO DIFFERENT FOR BFS
    Cluster * tree = (Cluster*)malloc(size*sizeof(Cluster));

    //get loc of root in copied array
    int root_loc = root->flatten_move(tree, 0, type);
    free(root);

    //set root pointer to place in array
    root = tree + root_loc;
    root_index = root_loc;
}


//subtree specifies the start index of subtree in array tree
//copies tree into array in order specified
//frees children after they have been copied
int Cluster::flatten_move(Cluster* tree, int subtree, int type){
    int root_loc, left_index, right_index = -1;
    int start_left, start_right;

    //determine where left, right subtrees begin
    switch (type){
        case 0:{ //inorder
            start_left = subtree;
            start_right = subtree + left_size + 1;
            root_loc = subtree + left_size;
            break;
        }
        case 1:{ //preorder
            start_left = subtree + 1;
            start_right = subtree + left_size + 1;
            root_loc = subtree;
            break;
        }
        case 2:{ //postorder
            start_left = subtree;
            start_right = subtree+left_size;
            root_loc = subtree+left_size + right_size;
            break;
        }
        case 3:{ //BFS, have to get max depth tho
            break;
        }
    }


    //update subtrees first
    //if no subtrees, left & right still nullptr
    if (left != nullptr){
        left_index = left->flatten_move(tree, start_left, type);
        free(left);
        left = tree + left_index;
    }

    if (right != nullptr){
        right_index = right->flatten_move(tree, start_right, type);
        free(right);
        right = tree + right_index;
    }

    //repurposes x_size to be the abs index of the child, for update_children
    left_size = left_index;
    right_size = right_index;

    //copy self into location in array, so can free heap memory
    memcpy(tree + root_loc, this, sizeof(Cluster));
    return root_loc;

}

void BoundedVolumeHierarchy::calculate_size(int type) {
    size = root->calculate_size();
}

//flattens subtree rooted at Cluster, returns number of elements in subtree, inclusive of root
//calculates L, R offsets 
int Cluster::calculate_size()  {
    if (left == nullptr) 
        left_size = 0;
    else
        left_size = left->calculate_size();
    
    if (right == nullptr)
        right_size = 0;
    else
        right_size = right->calculate_size();

    return 1 + left_size + right_size;

}


/*
    Assumes that every node is already flattened, updates pointers relative to the root using offsets L,F
    For CPU, this call made in flatten is sufficient
*/
void BoundedVolumeHierarchy::update_children(){
    Cluster* tree = root - root_index;
    //assumes the root is at least point to the right place
    root->update_children(tree);
}

void Cluster::update_children(Cluster* tree){
    //recall size was repurposed to be indices
    if (left != nullptr){
        left = tree + left_size;
        left->update_children(tree);
    }

    if (right != nullptr){
        right = tree + right_size;
        right->update_children(tree);
    }

}

//uses CPU pointers, breaks ties so it will be accurate for GPU
void Cluster::update_children_forGPU(Cluster* GPUtree){

    if (left != nullptr){
        left->update_children_forGPU(GPUtree);
        left = GPUtree + left_size;
    }

    if (right != nullptr){
        right->update_children_forGPU(GPUtree);
        right = GPUtree + right_size;
    }

    if (left == nullptr && right == nullptr){
        obj = obj ? obj->clone() : nullptr;
    }

}


/*
    copies a flattened BVH onto the GPU, updates the tree pointers to be relative to GPU array
*/
BoundedVolumeHierarchy BoundedVolumeHierarchy::flatten_clone() const{
    //mallocs array space  
    Cluster* d_begin;
    cudaMalloc(&d_begin, sizeof(Cluster)*size);

    //make tree relative to where it will exist in GPU
    root->update_children_forGPU(d_begin);

    //copies the tree into there
    cudaMemcpy(d_begin, root - root_index, sizeof(Cluster)*size, cudaMemcpyHostToDevice);

    BoundedVolumeHierarchy d_bvh;
    d_bvh.size = size;
    d_bvh.root_index = root_index;
    d_bvh.root = d_begin + root_index;

    return d_bvh; //ready for copy onto the device somewhere!
}
