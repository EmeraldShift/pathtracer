#include "../scene/scene.h"

Scene *Scene::clone() const {
    Scene *d_scene;
    cudaMalloc(&d_scene, sizeof(Scene));
    cudaMemcpy(d_scene, this, sizeof(Scene), cudaMemcpyHostToDevice);

    // Clone BVH
    BoundedVolumeHierarchy d_bvh = bvh.clone();
    cudaMemcpy(&d_scene->bvh, &d_bvh, sizeof(BoundedVolumeHierarchy), cudaMemcpyHostToDevice);
    return d_scene;
}