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
#include "../gl.h"
#include "../gpu/cuda.h"

#include <glm/geometric.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/matrix.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

class Scene {
public:
    Scene() = default;

    virtual ~Scene() = default;

    void add(Geometry *obj) {
        objects.emplace_back(obj);
    }

    template <typename T>
    void add(std::vector<T> objs) {
        for (const auto &o : objs)
            objects.emplace_back(o);
    }

    void constructBvh() {
        bvh.construct(objects);
    }

    CUDA_CALLABLE_MEMBER Camera &getCamera() { return camera; }

    bool intersect(ray &r, isect &i) {
        return bvh.traverse(r, i);
    }

    CUDA_CALLABLE_MEMBER bool intersectIterative(ray &r, isect &i) {
        return bvh.traverseIterative(r, i);
    }

    Scene *clone() const;

private:
    std::vector<Geometry *> objects;
    Camera camera;

    BoundedVolumeHierarchy bvh;
};

#endif // __SCENE_H__
