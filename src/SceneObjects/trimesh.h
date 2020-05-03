#ifndef TRIMESH_H__
#define TRIMESH_H__

#include <list>
#include <memory>
#include <vector>

#include "../scene/bvh.h"
#include "../scene/material.h"
#include "../scene/ray.h"
#include "../scene/scene.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/vec3.hpp>

class Trimesh;

class TrimeshFace : public MaterialSceneObject {
    Trimesh *parent;
    int ids[3];
    glm::dvec3 normal;
    double dist;
    glm::dvec3 uInv;
    glm::dvec3 vInv;
    glm::dvec3 nInv;


public:
    TrimeshFace(Scene *scene, Material *mat, Trimesh *parent, int a, int b,
                int c);

    BoundingBox localbounds;
    bool degen;

    int operator[](int i) const { return ids[i]; }

    glm::dvec3 getNormal() { return normal; }

    bool intersect(ray &r, isect &i) const;

    bool intersectLocal(ray &r, isect &i) const;

    bool hasBoundingBoxCapability() const override { return true; }

    BoundingBox computeLocalBoundingBox() override;

    const BoundingBox &getBoundingBox() const { return localbounds; }
};

class Trimesh : public MaterialSceneObject {
    friend class TrimeshFace;

    typedef std::vector<glm::dvec3> Normals;
    typedef std::vector<glm::dvec3> Vertices;
    typedef std::vector<TrimeshFace *> Faces;
    typedef std::vector<Material *> Materials;

    Vertices vertices;
    Faces faces;
    Normals normals;
    Materials materials;
    BoundingBox localBounds;

public:
    Trimesh(Scene *scene, Material *mat, TransformNode *transform)
            : MaterialSceneObject(scene, mat),
              displayListWithMaterials(0),
              displayListWithoutMaterials(0) {
        this->transform = transform;
        vertNorms = false;
    }

    bool vertNorms;

    bool intersectLocal(ray &r, isect &i) const;

    ~Trimesh();

    // must add vertices, normals, and materials IN ORDER
    void addVertex(const glm::dvec3 &);

    void addMaterial(Material *m);

    void addNormal(const glm::dvec3 &);

    bool addFace(int a, int b, int c);

    const char *doubleCheck();

    void generateNormals();

    bool hasBoundingBoxCapability() const { return true; }

    BoundingBox computeLocalBoundingBox() {
        BoundingBox localbounds;
        if (vertices.size() == 0)
            return localbounds;
        localbounds.setMax(vertices[0]);
        localbounds.setMin(vertices[0]);
        Vertices::const_iterator viter;
        for (viter = vertices.begin(); viter != vertices.end();
             ++viter) {
            localbounds.setMax(
                    glm::max(localbounds.getMax(), *viter));
            localbounds.setMin(
                    glm::min(localbounds.getMin(), *viter));
        }
        localBounds = localbounds;
        generateBounds();
        return localbounds;
    }

    void generateBounds() { bvh.construct(faces); };

    static bool compare(TrimeshFace *const lhs, TrimeshFace *const rhs, int i) {
        return ((lhs->getBoundingBox().getMax() + lhs->getBoundingBox().getMin()) / 2.0)[i]
               < ((rhs->getBoundingBox().getMax() + rhs->getBoundingBox().getMin()) / 2.0)[i];
    }

    static double spread(std::vector<TrimeshFace *> &v, int i) {
        return ((v[v.size() - 1]->getBoundingBox().getMax() + v[v.size() - 1]->getBoundingBox().getMin()) / 2.0)[i]
               - ((v[0]->getBoundingBox().getMax() + v[0]->getBoundingBox().getMin()) / 2.0)[i];
    }

private:
    BoundedVolumeHierarchy<TrimeshFace *, Trimesh::compare, Trimesh::spread> bvh;

protected:
    void glDrawLocal(int quality, bool actualMaterials,
                     bool actualTextures) const;

    mutable int displayListWithMaterials;
    mutable int displayListWithoutMaterials;
};

#endif // TRIMESH_H__
