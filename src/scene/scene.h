//
// scene.h
//
// The Scene class and the geometric types that it can contain.
//

#pragma warning(disable : 4786)

#ifndef __SCENE_H__
#define __SCENE_H__

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "bbox.h"
#include "camera.h"
#include "geometry.h"
#include "material.h"
#include "ray.h"
#include "bvh.h"
#include "transform.h"

#include <glm/geometric.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/matrix.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

class Scene {
public:
    Scene() = default;

    ~Scene() = default;

    void add(Geometry *obj) {
        actual.push_back(*obj);
        objects.emplace_back(obj);
    }

    template <typename T>
    void add(std::vector<T> objs) {
        for (const auto &o : objs){
            objects.emplace_back(o);
            actual.push_back(*o);
        }
    }

    void updateObjects(){
        //since we're looking down the x dir
        std::sort(actual.begin(), actual.end(), compare<2>);
        std::sort(actual.begin(), actual.end(), compare<1>);

        //makes pos in objs same as in actual
        //makes everything in contiguous memory
        for (int i = 0; i < actual.size(); i++)
            objects[i] = &actual[i];
    }

    void constructBvh() {
        bvh.construct(objects);
        bvh.flatten();
    }

    __host__ __device__ Camera &getCamera() { return camera; }

    bool intersect(ray &r, isect &i) {
        return bvh.traverse(r, i);
    }

    __host__ __device__ bool intersectIterative(ray &r, isect &i) {
        return bvh.traverseIterative(r, i);
    }

    Scene *clone() const;

    BoundedVolumeHierarchy bvh;
    std::vector<Geometry *> objects;
    Camera camera;
    std::vector<Geometry> actual;
};

#endif // __SCENE_H__
