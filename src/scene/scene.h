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
#include "material.h"
#include "ray.h"
#include "bvh.h"

#include "../gl.h"
#include <glm/geometric.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/matrix.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

using std::unique_ptr;

class Light;

class Scene;

inline glm::dvec3 operator*(const glm::dmat4x4 &mat, const glm::dvec3 &vec) {
    glm::dvec4 vec4(vec[0], vec[1], vec[2], 1.0);
    auto ret = mat * vec4;
    return glm::dvec3(ret[0], ret[1], ret[2]);
}

class TransformNode {
protected:
    // information about this node's transformation
    glm::dmat4x4 xform;
    glm::dmat4x4 inverse;
    glm::dmat3x3 normi;

    // information about parent & children
    TransformNode *parent;
    std::vector<TransformNode *> children;

public:
    typedef std::vector<TransformNode *>::iterator child_iter;
    typedef std::vector<TransformNode *>::const_iterator child_citer;

    ~TransformNode() {
        for (auto c : children)
            delete c;
    }

    TransformNode *createChild(const glm::dmat4x4 &xform) {
        auto child = new TransformNode(this, xform);
        children.push_back(child);
        return child;
    }

    // Coordinate-Space transformation
//    glm::dvec3 globalToLocalCoords(const glm::dvec3 &v) {
//        return inverse * v;
//    }

    glm::dvec3 localToGlobalCoords(const glm::dvec3 &v) {
        return xform * v;
    }

    glm::dvec4 localToGlobalCoords(const glm::dvec4 &v) {
        return xform * v;
    }

    glm::dvec3 localToGlobalCoordsNormal(const glm::dvec3 &v) {
        return glm::normalize(normi * v);
    }

    const glm::dmat4x4 &transform() const { return xform; }

protected:
    // protected so that users can't directly construct one of these...
    // force them to use the createChild() method.  Note that they CAN
    // directly create a TransformRoot object.
    TransformNode(TransformNode *parent, const glm::dmat4x4 &xform)
            : children() {
        this->parent = parent;
        if (parent == nullptr)
            this->xform = xform;
        else
            this->xform = parent->xform * xform;
        inverse = glm::inverse(this->xform);
        normi = glm::transpose(glm::inverse(glm::dmat3x3(this->xform)));
    }
};

class TransformRoot : public TransformNode {
public:
    TransformRoot() : TransformNode(nullptr, glm::dmat4x4(1.0)) {}
};

class MaterialSceneObject {
public:
    virtual ~MaterialSceneObject() = default;

    virtual const Material &getMaterial() const { return material; }

    virtual void setMaterial(Material m) { material = m; }

protected:
    explicit MaterialSceneObject(const Material mat) : material(mat) {
    }

    Material material;
};

// A Geometry object is anything that has extent in three dimensions.
// It may not be an actual visible scene object.  For example, hierarchical
// spatial subdivision could be expressed in terms of Geometry instances.
class Geometry : public MaterialSceneObject {
public:
    Geometry(const Material mat) : MaterialSceneObject(mat) {}

    const BoundingBox &getBoundingBox() const { return bounds; }

protected:
    BoundingBox bounds;
};

class Sphere : public Geometry {
public:
    Sphere(const Material &mat, TransformNode *txf)
            : Geometry(mat),
              position(txf->localToGlobalCoords(glm::dvec4(0, 0, 0, 1))),
              radius(txf->localToGlobalCoords(glm::dvec4(1, 0, 0, 0))[0]) {
        bounds = BoundingBox(glm::dvec3(position[0] - radius, position[1] - radius, position[2] - radius),
                             glm::dvec3(position[0] + radius, position[1] + radius, position[2] + radius));
    }

    Sphere(const Material &mat, glm::dvec3 position, double radius)
            : Geometry(mat),
              position(position), radius(radius) {
        bounds = BoundingBox(glm::dvec3(position[0] - radius, position[1] - radius, position[2] - radius),
                             glm::dvec3(position[0] + radius, position[1] + radius, position[2] + radius));
    }

    bool intersect(ray &r, isect &i) const;

    static bool compare(Sphere *const lhs, Sphere *const rhs, int i) {
        return ((lhs->getBoundingBox().getMax() + lhs->getBoundingBox().getMin()) / 2.0)[i]
               < ((rhs->getBoundingBox().getMax() + rhs->getBoundingBox().getMin()) / 2.0)[i];
    }

    static double spread(std::vector<Sphere *> &v, int i) {
        return ((v[v.size() - 1]->getBoundingBox().getMax() + v[v.size() - 1]->getBoundingBox().getMin()) / 2.0)[i]
               - ((v[0]->getBoundingBox().getMax() + v[0]->getBoundingBox().getMin()) / 2.0)[i];
    }

private:
    glm::dvec3 position;
    double radius;
};

class Trimesh;
class TrimeshFace : public Geometry {
    Trimesh *parent;
    int ids[3];
    glm::dvec3 normal;
    double dist;
    glm::dvec3 uInv;
    glm::dvec3 vInv;
    glm::dvec3 nInv;

public:
    TrimeshFace(const Material &mat, Trimesh *parent, int a, int b, int c);

    bool degen;

    int operator[](int i) const { return ids[i]; }

    glm::dvec3 getNormal() { return normal; }

    bool intersect(ray &r, isect &i) const;

    static bool compare(TrimeshFace *const lhs, TrimeshFace *const rhs, int i) {
        return ((lhs->getBoundingBox().getMax() + lhs->getBoundingBox().getMin()) / 2.0)[i]
               < ((rhs->getBoundingBox().getMax() + rhs->getBoundingBox().getMin()) / 2.0)[i];
    }

    static double spread(std::vector<TrimeshFace *> &v, int i) {
        return ((v[v.size() - 1]->getBoundingBox().getMax() + v[v.size() - 1]->getBoundingBox().getMin()) / 2.0)[i]
               - ((v[0]->getBoundingBox().getMax() + v[0]->getBoundingBox().getMin()) / 2.0)[i];
    }

private:
    BoundingBox computeLocalBoundingBox();
};

class Trimesh : public MaterialSceneObject {
    friend class TrimeshFace;

    typedef std::vector<glm::dvec3> Normals;
    typedef std::vector<glm::dvec3> Vertices;
    typedef std::vector<TrimeshFace *> Faces;
    typedef std::vector<Material> Materials;

    Vertices vertices;
    Faces faces;
    Normals normals;
    Materials materials;

public:
    explicit Trimesh(const Material &mat)
            : MaterialSceneObject(mat),
              displayListWithMaterials(0),
              displayListWithoutMaterials(0) {
    }

    ~Trimesh() override;

    // must add vertices, normals, and materials IN ORDER
    void addVertex(const glm::dvec3 &);

    void addMaterial(const Material &m);

    void addNormal(const glm::dvec3 &);

    bool addFace(int a, int b, int c);

    Faces getFaces() { return faces; };

    const char *doubleCheck();

    void generateNormals();

    static void genFace(Trimesh *mesh, glm::dvec3 a, glm::dvec3 b, glm::dvec3 c) {
        mesh->addVertex(a);
        mesh->addVertex(b);
        mesh->addVertex(c);
    }

    static Trimesh *genCube(Material &mat, glm::dvec3 v[8]) {
        auto mesh = new Trimesh(mat);
        genFace(mesh, v[0], v[1], v[2]);
        genFace(mesh, v[3], v[2], v[1]);
        genFace(mesh, v[4], v[6], v[5]);
        genFace(mesh, v[7], v[5], v[6]);
        genFace(mesh, v[0], v[2], v[4]);
        genFace(mesh, v[6], v[4], v[2]);
        genFace(mesh, v[1], v[5], v[3]);
        genFace(mesh, v[7], v[3], v[5]);
        genFace(mesh, v[0], v[1], v[4]);
        genFace(mesh, v[5], v[4], v[1]);
        genFace(mesh, v[2], v[6], v[3]);
        genFace(mesh, v[7], v[3], v[6]);
        for (int i = 0; i < 12; i++)
            mesh->addFace(i * 3, i * 3 + 1, i * 3 + 2);
        return mesh;
    }

    static Trimesh *fromBox(TransformNode *transform, Material &mat) {
        auto p000 = transform->localToGlobalCoords(glm::dvec3(-0.5, -0.5, -0.5));
        auto p001 = transform->localToGlobalCoords(glm::dvec3(-0.5, -0.5, 0.5));
        auto p010 = transform->localToGlobalCoords(glm::dvec3(-0.5, 0.5, -0.5));
        auto p011 = transform->localToGlobalCoords(glm::dvec3(-0.5, 0.5, 0.5));
        auto p100 = transform->localToGlobalCoords(glm::dvec3(0.5, -0.5, -0.5));
        auto p101 = transform->localToGlobalCoords(glm::dvec3(0.5, -0.5, 0.5));
        auto p110 = transform->localToGlobalCoords(glm::dvec3(0.5, 0.5, -0.5));
        auto p111 = transform->localToGlobalCoords(glm::dvec3(0.5, 0.5, 0.5));
        glm::dvec3 vertices[8] = {p000, p001, p010, p011, p100, p101, p110, p111};
        return genCube(mat, vertices);
    }

    static Trimesh *fromSquare(Material &mat, TransformNode *transform) {
        auto p000 = transform->localToGlobalCoords(glm::dvec3(-0.5, -0.5, -0.001));
        auto p001 = transform->localToGlobalCoords(glm::dvec3(-0.5, -0.5, 0.001));
        auto p010 = transform->localToGlobalCoords(glm::dvec3(-0.5, 0.5, -0.001));
        auto p011 = transform->localToGlobalCoords(glm::dvec3(-0.5, 0.5, 0.001));
        auto p100 = transform->localToGlobalCoords(glm::dvec3(0.5, -0.5, -0.001));
        auto p101 = transform->localToGlobalCoords(glm::dvec3(0.5, -0.5, 0.001));
        auto p110 = transform->localToGlobalCoords(glm::dvec3(0.5, 0.5, -0.001));
        auto p111 = transform->localToGlobalCoords(glm::dvec3(0.5, 0.5, 0.001));
        glm::dvec3 vertices[8] = {p000, p001, p010, p011, p100, p101, p110, p111};
        return genCube(mat, vertices);
    }

protected:

    mutable int displayListWithMaterials;
    mutable int displayListWithoutMaterials;
};

class Trimesh;
class TrimeshFace;

class Scene {
public:
    TransformRoot transformRoot;

    Scene();

    virtual ~Scene();

    void add(Sphere *sphere);

    void add(Trimesh *mesh);

    bool intersect(ray &r, isect &i);

    Camera &getCamera() { return camera; }

    // For efficiency reasons, we'll store texture maps in a cache
    // in the Scene.  This makes sure they get deleted when the scene
    // is destroyed.
    TextureMap *getTexture(string name);

    void addAmbient(const glm::dvec3 &ambient) {
        ambientIntensity += ambient;
    }

    void constructBvh() {
        sBvh.construct(spheres);
        tBvh.construct(triangles);
    }

private:
    std::vector<Sphere *> spheres;
    std::vector<TrimeshFace *> triangles;
    Camera camera;

    // This is the total amount of ambient light in the scene
    // (used as the I_a in the Phong shading model)
    glm::dvec3 ambientIntensity;

    typedef std::map<std::string, std::unique_ptr<TextureMap>> tmap;
    tmap textureCache;

    BoundedVolumeHierarchy<Sphere *, Sphere::compare, Sphere::spread> sBvh;
    BoundedVolumeHierarchy<TrimeshFace *, TrimeshFace::compare, TrimeshFace::spread> tBvh;

public:
    // This is used for debugging purposes only.
    mutable std::vector<std::pair<ray *, isect *>> intersectCache;
};

#endif // __SCENE_H__
